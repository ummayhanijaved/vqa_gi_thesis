"""
=============================================================================
EXPLAINABILITY MODULE
Thesis: Advancing Medical AI with Explainable VQA on GI Imaging

Produces for each image+question pair:
    1. GradCAM heatmap overlay        — WHERE the model looked
    2. Disease vector bar chart       — WHAT diseases were detected
    3. Routing decision confidence    — HOW the question was understood
    4. Textual explanation            — WHY this answer was given
    5. Full explainability report     — combined single figure

USAGE:
    # Single image
    python explainability.py \
        --image ./data/kvasir_raw/images/clb0kvxvm90y4074yf50vf5nq.jpg \
        --question "Is there a polyp visible?"

    # Batch over test set (saves report per sample)
    python explainability.py --mode batch --n_samples 20

    # Full analysis figures for thesis
    python explainability.py --mode thesis
=============================================================================
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, os.path.expanduser("~"))
from preprocessing import build_image_transform, TextPreprocessor
from stage1_disease_classifier import TreeNetDiseaseClassifier
from stage3_multimodal_fusion import FusionExtractor, CFG as S3_CFG
from stage4_answer_generation import (
    Stage4AnswerGenerator, load_vocabulary, CFG as S4_CFG
)

LOG_DIR = "./logs/explainability"
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QTYPE_NAMES  = ["yes/no","single-choice","multiple-choice",
                "color","location","numerical count"]
QTYPE_COLORS = ["#2196F3","#4CAF50","#FF9800",
                "#9C27B0","#F44336","#00BCD4"]

DISEASE_NAMES = [
    "Polyp (Pedunculated)","Polyp (Sessile)","Polyp (Hyperplastic)",
    "Esophagitis","Gastritis","Ulcerative Colitis","Crohn's Disease",
    "Barrett's Esophagus","Gastric Ulcer","Duodenal Ulcer",
    "Erosion","Hemorrhoids","Diverticulosis",
    "Normal Cecum","Normal Pylorus","Normal Z-Line",
    "Ileocecal Valve","Retroflex Rectum","Retroflex Stomach",
    "Dyed Lifted Polyp","Dyed Resection Margin",
    "Foreign Body","Instrument"
]
DISEASE_SHORT = [
    "P-Ped","P-Sess","P-Hyp","Esoph","Gastri","UC","Crohn",
    "Barrett","Gas-Ulc","Duo-Ulc","Erosion","Hemor","Diverti",
    "N-Cecum","N-Pylor","N-Z-Line","Ileocec","Retro-R","Retro-S",
    "Dyed-LP","Dyed-RM","For-Body","Instrum"
]


# ─────────────────────────────────────────────────────────────────────────
# GradCAM implementation for ResNet50 backbone
# ─────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet50.
    Hooks into the final conv layer (layer4) to produce spatial heatmaps.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """
        Args:
            img_tensor   : (1, 3, 224, 224) — preprocessed image
            target_class : which output neuron to backprop from
                           (None = argmax = predicted class)
        Returns:
            cam : (224, 224) numpy array in [0, 1]
        """
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE).float()
        img_tensor.requires_grad_(True)

        # Forward pass through backbone only (get feature map)
        with torch.enable_grad():
            # Get features via visual encoder
            features = self.model(img_tensor)  # (1, n_diseases)
            if target_class is None:
                target_class = features.argmax(dim=1).item()
            score = features[0, target_class]
            self.model.zero_grad()
            score.backward()

        # Pool gradients over spatial dimensions
        gradients  = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]       # (C, H, W)
        weights    = gradients.mean(dim=(1, 2)) # (C,)

        # Weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU + normalise
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input resolution
        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (224, 224))
        return cam_np


class GradCAMExtractor:
    """
    Wraps Stage 1 TreeNet for GradCAM.
    Structure discovered: model.backbone is nn.Sequential where
        [0..3]  = stem (conv1, bn1, relu, maxpool)
        [4]     = layer1,  [5] = layer2
        [6]     = layer3,  [7] = layer4   ← target
        [8]     = avgpool
    The disease MLP head follows separately.
    """

    def __init__(self, stage1_ckpt: str = "./checkpoints/stage1_best.pt"):
        from stage1_disease_classifier import TreeNetDiseaseClassifier

        self.full_model = TreeNetDiseaseClassifier().to(DEVICE)
        ckpt = torch.load(stage1_ckpt, map_location=DEVICE)
        self.full_model.load_state_dict(ckpt["model_state"])
        self.full_model.eval()
        for p in self.full_model.parameters():
            p.requires_grad_(True)

        # backbone is Sequential — layer4 is the last block before avgpool
        # Find it: last Sequential child that itself contains Bottleneck blocks
        seq = self.full_model.backbone  # nn.Sequential
        target_layer = self._find_layer4_conv3(seq)
        print(f"   GradCAM target: backbone[{self._layer4_idx}]"
              f"[-1].conv3  →  {type(target_layer).__name__}"
              f"  out_ch={target_layer.out_channels}")

        # Build a simple forward wrapper that returns disease logits
        self._grad_model = _TreeNetGradWrapper(self.full_model).to(DEVICE)

        # Re-find the same layer inside the wrapper's reference
        target_in_wrapper = self._find_layer4_conv3(
            self._grad_model.backbone)
        self.gradcam = GradCAM(self._grad_model, target_in_wrapper)

    def _find_layer4_conv3(self, seq: nn.Sequential) -> nn.Conv2d:
        """
        In ResNet50-as-Sequential, layer4 is the last child that
        is itself a Sequential of Bottleneck modules (has conv3).
        Walk backwards to find it.
        """
        children = list(seq.children())
        for idx in range(len(children)-1, -1, -1):
            child = children[idx]
            # layer4 is a Sequential of Bottleneck blocks
            if isinstance(child, nn.Sequential):
                blocks = list(child.children())
                if blocks and hasattr(blocks[-1], "conv3"):
                    self._layer4_idx = idx
                    return blocks[-1].conv3
        # Fallback: last Conv2d in backbone that isn't in MLP head
        last_conv = None
        for name, mod in seq.named_modules():
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
        self._layer4_idx = -1
        return last_conv

    def get_cam(self, img_tensor: torch.Tensor,
                target_class: int = None) -> np.ndarray:
        return self.gradcam.generate(img_tensor, target_class)


class _TreeNetGradWrapper(nn.Module):
    """
    Thin wrapper around TreeNet that makes forward() return disease logits.
    Shares the same parameter tensors — no copying.
    """
    def __init__(self, treenet):
        super().__init__()
        self.backbone = treenet.backbone   # the Sequential
        # Find the disease head — could be named various things
        self._head = self._find_head(treenet)

    @staticmethod
    def _find_head(model) -> nn.Module:
        """Find the MLP head that produces 23-D disease logits."""
        # Try common attribute names
        for attr in ["disease_head", "head", "classifier",
                     "mlp_head", "fc_head", "disease_mlp",
                     "mlp", "output_head"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        # Fallback: last non-backbone child module
        children = dict(model.named_children())
        for key in reversed(list(children.keys())):
            if key != "backbone":
                return children[key]
        raise RuntimeError("Cannot find disease head in TreeNet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)   # (B, 2048) after avgpool+flatten
        return self._head(feat)   # (B, 23)


# ─────────────────────────────────────────────────────────────────────────
# Full Explainability Predictor
# ─────────────────────────────────────────────────────────────────────────
class ExplainabilityPredictor:
    """
    Complete explainability pipeline:
        Image + Question
            → GradCAM heatmap (spatial attention)
            → Disease vector  (pathology detected)
            → Route + confidence (question understanding)
            → Answer          (final prediction)
            → Textual explanation
    """
    def __init__(self):
        print("🔍  Loading ExplainabilityPredictor ...")

        self.img_tfm   = build_image_transform("test")
        self.text_prep = TextPreprocessor()

        # Stage 1 — for GradCAM
        self.gradcam_ext = GradCAMExtractor()

        # Stage 3 — fusion extractor
        self.extractor = FusionExtractor(S4_CFG["stage3_ckpt"])

        # Stage 4 — answer heads
        self.vocab = load_vocabulary()
        ckpt       = torch.load("./checkpoints/stage4_best.pt",
                                map_location=DEVICE)
        self.model = Stage4AnswerGenerator(self.vocab).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        print("✅  ExplainabilityPredictor ready\n")

    @torch.no_grad()
    def predict_full(self, image_path: str,
                     question: str) -> dict:
        """
        Full prediction + explainability.
        Returns dict with all intermediate results.
        """
        # ── Load image ────────────────────────────────────────────────
        img_pil   = PILImage.open(image_path).convert("RGB")
        img_224   = img_pil.resize((224, 224))
        img_tensor = self.img_tfm(img_pil).unsqueeze(0).to(DEVICE)

        # ── GradCAM ───────────────────────────────────────────────────
        try:
            cam = self.gradcam_ext.get_cam(img_tensor.clone())
        except Exception:
            cam = np.zeros((224, 224))

        # ── Tokenise question ─────────────────────────────────────────
        q_enc = self.text_prep.preprocess(question)
        ids   = q_enc["input_ids"].unsqueeze(0).to(DEVICE)
        mask  = q_enc["attention_mask"].unsqueeze(0).to(DEVICE)

        # ── Stage 3 fusion ────────────────────────────────────────────
        fusion_out  = self.extractor.extract(img_tensor, ids, mask)
        fused       = fusion_out["fused_repr"]        # (1,512)
        disease_vec = fusion_out["disease_vec"]        # (1,23)
        route       = fusion_out["routing_label"].item()
        route_probs = fusion_out["routing_probs"][0].cpu().numpy()
        route_conf  = float(route_probs[route])

        # ── Stage 4 answer ────────────────────────────────────────────
        answers = self.model.predict(
            fused, disease_vec, route, self.vocab)
        answer  = answers[0]

        # ── Disease vector ────────────────────────────────────────────
        d_vec   = disease_vec[0].cpu().numpy()
        active  = [(DISEASE_NAMES[i], float(d_vec[i]))
                   for i in np.argsort(d_vec)[::-1] if d_vec[i] > 0.3]

        # ── Textual explanation ───────────────────────────────────────
        explanation = self._build_explanation(
            question, answer, route, route_conf,
            d_vec, route_probs)

        return dict(
            image_path  = image_path,
            image_pil   = img_pil,
            image_224   = img_224,
            question    = question,
            answer      = answer,
            route       = route,
            route_name  = QTYPE_NAMES[route],
            route_conf  = route_conf,
            route_probs = route_probs,
            disease_vec = d_vec,
            active_diseases = active,
            cam         = cam,
            explanation = explanation,
        )

    def _build_explanation(self, question, answer, route,
                           route_conf, d_vec, route_probs) -> str:
        """Generate human-readable explanation text."""

        top_disease_idx  = int(np.argmax(d_vec))
        top_disease_name = DISEASE_NAMES[top_disease_idx]
        top_disease_prob = float(d_vec[top_disease_idx])

        # Active diseases above threshold
        active = [DISEASE_NAMES[i] for i in range(23)
                  if d_vec[i] > 0.5]

        route_name   = QTYPE_NAMES[route]
        second_route = np.argsort(route_probs)[::-1][1]

        lines = []
        lines.append(f"QUESTION TYPE DETECTION:")
        lines.append(f"  The question was classified as a [{route_name}] "
                     f"question with {route_conf*100:.1f}% confidence.")
        if route_conf < 0.7:
            lines.append(f"  (Note: low confidence — next best was "
                         f"[{QTYPE_NAMES[second_route]}] at "
                         f"{route_probs[second_route]*100:.1f}%)")
        lines.append("")

        lines.append(f"DISEASE CONTEXT:")
        if active:
            lines.append(f"  Stage 1 detected {len(active)} active condition(s):")
            for i, name in enumerate(active[:4]):
                prob = d_vec[DISEASE_NAMES.index(name)]
                lines.append(f"    {i+1}. {name}: {prob*100:.1f}%")
        else:
            lines.append(f"  No diseases detected above 50% threshold.")
            lines.append(f"  Highest: {top_disease_name} ({top_disease_prob*100:.1f}%)")
        lines.append("")

        lines.append(f"ANSWER GENERATION:")
        lines.append(f"  Route [{route_name}] head selected answer: '{answer}'")

        # Route-specific explanation
        if route == 0:
            lines.append(f"  Binary yes/no decision driven by disease "
                         f"presence signals.")
            if answer == "yes":
                lines.append(f"  High disease probability scores supported "
                             f"a positive finding.")
            else:
                lines.append(f"  Low disease probability scores indicated "
                             f"absence of pathology.")
        elif route == 1:
            lines.append(f"  Single-choice classification selected from "
                         f"200-word vocabulary.")
        elif route == 2:
            lines.append(f"  Multi-label classification — multiple findings "
                         f"may co-occur.")
        elif route == 3:
            lines.append(f"  Color classification based on visual features "
                         f"extracted by ResNet50.")
        elif route == 4:
            lines.append(f"  Anatomical location identified from spatial "
                         f"visual features.")
        elif route == 5:
            lines.append(f"  Ordinal count estimated from visual density "
                         f"of findings.")

        lines.append("")
        lines.append(f"SPATIAL ATTENTION:")
        lines.append(f"  GradCAM highlights the image regions that most "
                     f"influenced the disease detection decision.")
        lines.append(f"  Red/warm regions = high attention, "
                     f"Blue/cool regions = low attention.")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────
def make_cam_overlay(img_224: PILImage.Image,
                     cam: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    """Blend GradCAM heatmap over original image."""
    img_np = np.array(img_224)

    # Custom medical colormap: blue → green → yellow → red
    colors_list = ["#0000FF","#00FF00","#FFFF00","#FF0000"]
    cmap = LinearSegmentedColormap.from_list("medical", colors_list)

    cam_uint8  = (cam * 255).astype(np.uint8)
    cam_color  = plt.get_cmap("jet")(cam)[:,:,:3]
    cam_color  = (cam_color * 255).astype(np.uint8)

    overlay = (alpha * cam_color + (1-alpha) * img_np).astype(np.uint8)
    return overlay


def plot_explainability_report(result: dict,
                                save_path: str = None) -> str:
    """
    Generate the complete single-figure explainability report.
    6-panel layout:
        [0,0] Original image
        [0,1] GradCAM overlay
        [0,2] Route confidence
        [1,0] Disease vector (top 10)
        [1,1] Disease vector (full 23)
        [1,2] Explanation text
    """
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#0D1117")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.35, wspace=0.30)

    route      = result["route"]
    route_color = QTYPE_COLORS[route]
    d_vec       = result["disease_vec"]

    fig.suptitle(
        f"Explainability Report — [{result['route_name'].upper()}] Question\n"
        f"Q: {result['question'][:90]}{'...' if len(result['question'])>90 else ''}\n"
        f"Predicted Answer: \"{result['answer']}\"   |   "
        f"Route Confidence: {result['route_conf']*100:.1f}%",
        fontsize=11, fontweight="bold",
        color="white", y=0.98)

    # ── Panel 0: Original image ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(result["image_224"])
    ax0.set_title("Original Image", color="white",
                  fontweight="bold", fontsize=10)
    ax0.axis("off")
    ax0.set_facecolor("#0D1117")

    # ── Panel 1: GradCAM overlay ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    overlay = make_cam_overlay(result["image_224"], result["cam"])
    ax1.imshow(overlay)
    ax1.set_title("GradCAM — Spatial Attention\n"
                  "(WHERE the model focused)",
                  color="white", fontweight="bold", fontsize=10)
    ax1.axis("off")
    ax1.set_facecolor("#0D1117")

    # Colorbar for CAM
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    sm  = plt.cm.ScalarMappable(cmap="jet",
          norm=plt.Normalize(vmin=0, vmax=1))
    cb  = plt.colorbar(sm, cax=cax)
    cb.set_label("Attention", color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # ── Panel 2: Route confidence ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#161B22")
    route_probs = result["route_probs"]
    colors_r    = [route_color if i == route
                   else "#404040" for i in range(6)]
    bars = ax2.barh(QTYPE_NAMES, route_probs*100,
                    color=colors_r, alpha=0.9,
                    edgecolor="none", height=0.65)
    ax2.set_title("Routing Decision\n(WHAT question type)",
                  color="white", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Confidence (%)", color="white", fontsize=8)
    ax2.set_xlim(0, 115)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines[:].set_color("#404040")
    for bar, val in zip(bars, route_probs*100):
        ax2.text(val+1, bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}%", va="center",
                 color="white", fontsize=8)

    # ── Panel 3: Top disease bar chart ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161B22")
    top10_idx  = np.argsort(d_vec)[::-1][:10]
    top10_vals = d_vec[top10_idx]
    top10_names= [DISEASE_SHORT[i] for i in top10_idx]
    top10_colors = ["#F44336" if v >= 0.5 else
                    "#FF9800" if v >= 0.3 else
                    "#4CAF50" for v in top10_vals]
    ax3.barh(range(10), top10_vals[::-1]*100,
             color=top10_colors[::-1], alpha=0.9,
             edgecolor="none", height=0.65)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top10_names[::-1], color="white", fontsize=8)
    ax3.axvline(50, color="red", linestyle="--",
                alpha=0.6, lw=1.5)
    ax3.set_title("Top 10 Disease Probabilities\n(WHAT was detected)",
                  color="white", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Probability (%)", color="white", fontsize=8)
    ax3.set_xlim(0, 115)
    ax3.tick_params(colors="white", labelsize=8)
    ax3.spines[:].set_color("#404040")
    for i, val in enumerate(top10_vals[::-1]*100):
        ax3.text(val+1, i, f"{val:.1f}%", va="center",
                 color="white", fontsize=7)

    # ── Panel 4: Full disease vector heatmap ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161B22")
    im = ax4.imshow(d_vec.reshape(1,-1), aspect="auto",
                    cmap="RdYlGn_r", vmin=0, vmax=1,
                    interpolation="nearest")
    ax4.set_xticks(range(23))
    ax4.set_xticklabels(DISEASE_SHORT, rotation=90,
                        fontsize=5.5, color="white")
    ax4.set_yticks([])
    ax4.set_title("Full Disease Vector d (23-D)\n"
                  "(WHY — pathology evidence)",
                  color="white", fontweight="bold", fontsize=10)
    ax4.spines[:].set_color("#404040")
    # Annotate each cell
    for j, val in enumerate(d_vec):
        color_t = "black" if val > 0.5 else "white"
        ax4.text(j, 0, f"{val:.2f}", ha="center", va="center",
                 fontsize=5, color=color_t, fontweight="bold")
    cb2 = plt.colorbar(im, ax=ax4, fraction=0.05, pad=0.02,
                       orientation="horizontal")
    cb2.set_label("Disease Probability", color="white", fontsize=7)
    cb2.ax.xaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb2.ax.xaxis.get_ticklabels(), color="white")

    # ── Panel 5: Textual explanation ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#161B22")
    ax5.axis("off")
    ax5.set_title("Textual Explanation", color="white",
                  fontweight="bold", fontsize=10)
    ax5.text(0.02, 0.97, result["explanation"],
             transform=ax5.transAxes,
             fontsize=7, va="top", color="#E0E0E0",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#1C2128",
                       edgecolor=route_color,
                       linewidth=1.5))

    # Footer
    fig.text(0.5, 0.01,
             "Pipeline: ResNet50 (Disease) → DistilBERT (Routing) → "
             "CrossAttn+DiseaseGate (Fusion) → Specialised Head (Answer)",
             ha="center", fontsize=8, color="#808080")

    if save_path is None:
        q_slug = result["question"][:30].replace(" ","_").replace("?","")
        save_path = f"{LOG_DIR}/explain_{q_slug}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────
# THESIS ANALYSIS FIGURES
# ─────────────────────────────────────────────────────────────────────────
def run_thesis_figures(predictor: ExplainabilityPredictor):
    """
    Generate 6 explainability reports (one per question type)
    + summary comparison figure for thesis.
    """
    from datasets import load_from_disk, Image as HFImage

    print("\n📚  Generating thesis explainability figures ...")
    raw = load_from_disk(S4_CFG["data_dir"])
    raw = raw.cast_column("image", HFImage())

    # Find one good example per question type
    from stage3_multimodal_fusion import infer_qtype_label
    selected = {}
    for ex in raw["test"]:
        q, a = ex["question"], ex["answer"]
        r    = infer_qtype_label(q, a)
        if r not in selected and ex["image"] is not None:
            selected[r] = ex
        if len(selected) == 6:
            break

    saved_paths = []
    all_results = []

    for r in sorted(selected.keys()):
        ex = selected[r]
        # Always save to a known temp path with .jpg extension
        img_path = f"/tmp/tmp_explain_route{r}.jpg"
        ex["image"].convert("RGB").save(img_path, format="JPEG")

        q = ex["question"]
        print(f"   Route {r} [{QTYPE_NAMES[r]}]: {q[:60]}")

        try:
            result = predictor.predict_full(img_path, q)
            path   = plot_explainability_report(
                result,
                save_path=f"{LOG_DIR}/thesis_explain_route{r}_{QTYPE_NAMES[r].replace('/','_').replace('-','_')}.png"
            )
            saved_paths.append(path)
            all_results.append(result)
            print(f"   ✅  {path}")
        except Exception as e:
            print(f"   ⚠️   Route {r} failed: {e}")

    # Summary comparison figure
    _plot_summary_comparison(all_results)
    return saved_paths


def _plot_summary_comparison(results: list):
    """Side-by-side mini reports for all routes on one page."""
    if not results:
        return

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4*n))
    fig.suptitle("Explainability Summary — One Example per Question Type\n"
                 "GradCAM | Disease Vector | Routing Confidence",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, result in enumerate(results):
        r = result["route"]

        # Original + GradCAM side-by-side
        overlay = make_cam_overlay(result["image_224"], result["cam"])
        img_arr = np.concatenate(
            [np.array(result["image_224"]), overlay], axis=1)
        axes[row,0].imshow(img_arr)
        axes[row,0].axis("off")
        axes[row,0].set_title(
            f"[{QTYPE_NAMES[r]}]\n"
            f"Q: {result['question'][:55]}\n"
            f"A: {result['answer']}",
            fontsize=8, fontweight="bold",
            color=QTYPE_COLORS[r])

        # Disease vector
        top8_idx   = np.argsort(result["disease_vec"])[::-1][:8]
        top8_vals  = result["disease_vec"][top8_idx]*100
        top8_colors= ["#F44336" if v>=50 else "#FF9800" if v>=30
                      else "#90CAF9" for v in top8_vals]
        axes[row,1].barh(range(8), top8_vals[::-1],
                         color=top8_colors[::-1], alpha=0.85,
                         edgecolor="white", height=0.7)
        axes[row,1].set_yticks(range(8))
        axes[row,1].set_yticklabels(
            [DISEASE_SHORT[i] for i in top8_idx[::-1]], fontsize=7)
        axes[row,1].axvline(50,color="red",linestyle="--",
                            alpha=0.6,lw=1)
        axes[row,1].set_xlabel("Disease Prob (%)",fontsize=7)
        axes[row,1].set_xlim(0,110)
        axes[row,1].grid(axis="x",alpha=0.3)

        # Route confidence
        axes[row,2].barh(QTYPE_NAMES,
                         result["route_probs"]*100,
                         color=[QTYPE_COLORS[i] if i==r
                                else "#CCCCCC" for i in range(6)],
                         alpha=0.85, edgecolor="white", height=0.65)
        axes[row,2].set_xlabel("Confidence (%)",fontsize=7)
        axes[row,2].set_xlim(0,115)
        axes[row,2].set_title(
            f"Route conf={result['route_conf']*100:.1f}%",
            fontsize=8)
        axes[row,2].grid(axis="x",alpha=0.3)
        for i, val in enumerate(result["route_probs"]*100):
            axes[row,2].text(val+0.5,i,f"{val:.1f}%",
                             va="center",fontsize=7)

    plt.tight_layout()
    path = f"{LOG_DIR}/thesis_explainability_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅  Summary figure: {path}")


# ─────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["single","batch","thesis"],
        default="single")
    parser.add_argument("--image",    default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--n_samples",type=int, default=20)
    args = parser.parse_args()

    predictor = ExplainabilityPredictor()

    if args.mode == "single":
        if not args.image or not args.question:
            print("❌  Provide --image and --question for single mode")
            print("    Example:")
            print("    python explainability.py \\")
            print("        --image ./data/kvasir_raw/images/xxx.jpg \\")
            print('        --question "Is there a polyp visible?"')
            sys.exit(1)

        print(f"\n🔍  Explaining: {args.question}")
        result = predictor.predict_full(args.image, args.question)
        path   = plot_explainability_report(result)

        print(f"\n{'='*60}")
        print(f"Question      : {result['question']}")
        print(f"Answer        : {result['answer']}")
        print(f"Question type : {result['route_name']}")
        print(f"Route conf    : {result['route_conf']*100:.2f}%")
        print(f"Active diseases:")
        for name, prob in result["active_diseases"][:5]:
            print(f"    {name}: {prob*100:.1f}%")
        print(f"\nExplanation:\n{result['explanation']}")
        print(f"\n✅  Report saved → {path}")

    elif args.mode == "batch":
        from datasets import load_from_disk, Image as HFImage
        from stage3_multimodal_fusion import infer_qtype_label

        raw = load_from_disk(S4_CFG["data_dir"])
        raw = raw.cast_column("image", HFImage())
        test_split = raw["test"]

        print(f"\n🔍  Batch explainability — {args.n_samples} samples")
        count = 0
        for ex in tqdm(test_split, total=args.n_samples):
            if count >= args.n_samples:
                break
            img = ex["image"]
            img_path = f"/tmp/explain_batch_{count:03d}.jpg"
            img.convert("RGB").save(img_path, format="JPEG")
            try:
                result = predictor.predict_full(img_path, ex["question"])
                path   = plot_explainability_report(
                    result,
                    save_path=f"{LOG_DIR}/batch_{count:03d}.png")
                count += 1
            except Exception as e:
                print(f"   ⚠️  Sample {count} failed: {e}")
        print(f"\n✅  {count} reports saved → {LOG_DIR}/")

    elif args.mode == "thesis":
        paths = run_thesis_figures(predictor)
        print(f"\n✅  {len(paths)} thesis figures generated → {LOG_DIR}/")
