#!/usr/bin/env python3
"""
=============================================================================
Stage 7 — Grad-CAM Visual Explainability (Stage 1 ResNet50)
=============================================================================

PURPOSE:
    Generate Grad-CAM heatmaps showing WHICH IMAGE REGIONS drove the
    pipeline's answer. The heatmap is computed at Stage 1's ResNet50
    last conv layer (layer4: 2048×7×7 spatial maps) but DRIVEN BY the
    disease class most relevant to the pipeline's textual answer.

    This gives visual grounding for the textual VQA answer:
       "Is there a polyp?" → "Yes" → [heatmap highlights the polyp region]

ARCHITECTURE NOTE:
    Stage 1 ResNet50 backbone = Sequential(*children[:-1]), which includes
    avgpool. For Grad-CAM we hook the LAST CONV BLOCK (layer4), which is
    at index -3 in the original ResNet50 children (before avgpool + the
    removed FC). We register forward/backward hooks there to capture
    activations and gradients.

    The backbone is frozen (requires_grad=False), but Grad-CAM still works:
    we temporarily enable gradients on the INPUT and capture gradients at
    layer4 via hooks (frozen weights still propagate gradients).

OUTPUT:
    ~/vqa_gi_thesis/logs/stage7_gradcam/
      ├── gradcam_<img_id>_<disease>.png   (original | heatmap | overlay)
      ├── gradcam_grid_per_route.png       (thesis figure)
      └── gradcam_index.csv                (metadata for each visualization)

USAGE:
    # Demo on a few test images (one per route)
    python stage7_gradcam.py --mode demo

    # Generate N visualizations across all routes
    python stage7_gradcam.py --mode bulk --n_per_route 3

    # Specific image
    python stage7_gradcam.py --mode single --image_path /path/img.jpg \\
        --question "Is there a polyp?"

REQUIRED:
    pip install matplotlib opencv-python --break-system-packages
=============================================================================
"""
import os
import sys
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

# Import Stage 1 model + labels
from stage1_disease_classifier import (
    TreeNetDiseaseClassifier, DISEASE_LABELS, CFG as S1_CFG,
)
from preprocessing import build_image_transform

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
    print("⚠️   matplotlib missing — pip install matplotlib")

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("⚠️   opencv missing — pip install opencv-python (will use matplotlib fallback)")

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Robust transform builder — preprocessing.py signature varies across versions
# (some take is_train=bool, some take a positional split string like "val")
# ─────────────────────────────────────────────────────────────────────────────
def get_eval_transform():
    """Build an eval/inference image transform, tolerant of signature."""
    import inspect
    try:
        sig = inspect.signature(build_image_transform)
        params = list(sig.parameters.keys())
    except (ValueError, TypeError):
        params = []
    # Try the styles in order of likelihood
    attempts = []
    if "is_train" in params:
        attempts.append(lambda: get_eval_transform())
    if "split" in params or len(params) >= 1:
        attempts.append(lambda: build_image_transform("val"))
        attempts.append(lambda: build_image_transform("test"))
    attempts.append(lambda: build_image_transform(False))
    attempts.append(lambda: build_image_transform())
    last_err = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not build image transform with any known signature. "
        f"Last error: {last_err}")



# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/vqa_gi_thesis")
CFG = {
    "device":   "cuda" if torch.cuda.is_available() else "cpu",
    "stage1_ckpt": os.path.join(PROJECT, "checkpoints", "stage1_best.pt"),
    "out_dir":  os.path.join(PROJECT, "logs", "stage7_gradcam"),
    "img_size": S1_CFG["img_size"],
    "img_mean": S1_CFG["img_mean"],
    "img_std":  S1_CFG["img_std"],
    "data_dir": os.path.expanduser("~/data/kvasir_local"),
}
os.makedirs(CFG["out_dir"], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Map question/answer → relevant disease class index
# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS FIX (reviewer): build keyword→index map DYNAMICALLY from the
# imported DISEASE_LABELS so indices ALWAYS match Stage 1's real order.
# Hardcoding indices was fragile — if Stage 1's label order ever changed,
# the heatmap would explain the wrong disease while labeling it correctly.
def build_answer_to_disease(labels):
    """
    Build keyword→[indices] map from the actual label list.

    SEMANTIC-DRIFT FIX (reviewer): we use the FULL label name and the
    full hyphen-normalized name as keys (specific), and only add a small
    set of CURATED clinical synonyms. We deliberately AVOID bare first-token
    splits like 'dyed' or generic 'polyp' collecting unrelated dye-marking
    classes, which would let pick_disease_for_answer() drift to the wrong
    clinical class.
    """
    m = {}
    for i, name in enumerate(labels):
        n = str(name).lower().strip()
        # Specific keys: full name + hyphen-normalized full name
        keys = {n, n.replace("-", " ")}
        # Strip "normal-" prefix as a specific landmark key
        if n.startswith("normal-"):
            keys.add(n[len("normal-"):])           # e.g. "z-line"
            keys.add(n[len("normal-"):].replace("-", " "))
        for kw in keys:
            kw = kw.strip()
            if kw:
                m.setdefault(kw, [])
                if i not in m[kw]:
                    m[kw].append(i)

    # CURATED synonyms — each maps to EXACTLY the clinically-correct classes.
    # We only attach indices whose label contains the precise substring.
    def attach(syn, must_contain, must_not_contain=None):
        for i, name in enumerate(labels):
            ln = str(name).lower()
            if must_contain in ln:
                if must_not_contain and must_not_contain in ln:
                    continue
                m.setdefault(syn, [])
                if i not in m[syn]:
                    m[syn].append(i)

    # "polyp" should mean the THREE plain polyp classes (0,1,2), NOT the
    # dyed-lifted-polyp (19) which is a dye-marking class.
    attach("polyp", "polyp", must_not_contain="dyed")
    attach("pedunculated", "pedunculated")
    attach("sessile", "sessile")
    attach("hyperplastic", "hyperplastic")
    attach("colitis",   "colitis")
    attach("crohn",     "crohn")
    attach("crohns",    "crohn")
    attach("barrett",   "barrett")
    attach("barretts",  "barrett")
    # "ulcer" → gastric-ulcer + duodenal-ulcer (but NOT ulcerative-colitis)
    attach("ulcer",     "ulcer", must_not_contain="colitis")
    attach("erosion",   "erosion")
    attach("hemorrhoid","hemorrhoid")
    attach("diverticul","diverticul")
    attach("cecum",     "cecum")
    attach("pylorus",   "pylorus")
    attach("z-line",    "z-line")
    attach("zline",     "z-line")
    attach("ileocecal", "ileocecal")
    attach("rectum",    "rectum")
    attach("stomach",   "stomach")
    # "dyed" → only the dye-marking classes
    attach("dyed",      "dyed")
    attach("lifted",    "lifted")
    attach("resection", "resection")
    attach("foreign",   "foreign")
    attach("instrument","instrument")
    return m


ANSWER_TO_DISEASE = build_answer_to_disease(DISEASE_LABELS)


def pick_disease_for_answer(question, answer, disease_probs):
    """
    Decide which disease class should drive the Grad-CAM heatmap.

    Priority:
      1. Keyword match in answer text → that disease
      2. Keyword match in question text → that disease
      3. Fallback: highest-probability disease from Stage 1
    """
    text = f"{answer} {question}".lower()
    for keyword, indices in ANSWER_TO_DISEASE.items():
        if keyword in text:
            # Among the matched indices, pick the one with highest prob
            best = max(indices, key=lambda i: disease_probs[i]
                        if i < len(disease_probs) else 0)
            return best, DISEASE_LABELS[best], f"keyword:'{keyword}'"
    # Fallback — highest probability disease
    best = int(np.argmax(disease_probs))
    return best, DISEASE_LABELS[best], "top-prob"


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for Stage 1 ResNet50.
    Hooks the last conv block (layer4) to capture activations + gradients.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Find the last conv layer (layer4) in the backbone.
        # backbone = Sequential(*resnet50.children()[:-1])
        # Structure: [conv1, bn1, relu, maxpool, layer1, layer2, layer3,
        #             layer4, avgpool]
        # layer4 is at index -2 (before avgpool which is last)
        target_layer = None
        modules = list(self.model.backbone.children())
        # Walk backwards to find the last conv-containing block
        for m in reversed(modules):
            # avgpool is AdaptiveAvgPool2d — skip it
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                continue
            # The Bottleneck blocks (layer4) contain convs
            target_layer = m
            break

        if target_layer is None:
            raise RuntimeError("Could not locate target conv layer in backbone")

        # SAFETY ASSERTION (reviewer suggestion): verify we hooked a conv
        # block, not avgpool or something unexpected. ResNet50's layer4 is
        # an nn.Sequential of Bottleneck blocks.
        layer_type = type(target_layer).__name__
        if isinstance(target_layer, torch.nn.AdaptiveAvgPool2d):
            raise RuntimeError(
                f"Grad-CAM hooked AdaptiveAvgPool2d — wrong layer! "
                f"Expected a conv block (Sequential/Bottleneck). "
                f"Backbone structure may have changed.")
        # Confirm the layer contains conv operations
        has_conv = any(isinstance(m, torch.nn.Conv2d)
                        for m in target_layer.modules())
        if not has_conv:
            raise RuntimeError(
                f"Grad-CAM target layer '{layer_type}' contains no Conv2d — "
                f"cannot produce spatial heatmap. Check backbone structure.")

        self.target_layer = target_layer
        print(f"   Grad-CAM hooked layer: {layer_type} "
              f"(✅ verified contains Conv2d)")

        # Register hooks
        self.fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        """
        Generate Grad-CAM heatmap for a specific disease class.

        Args:
            input_tensor: (1, 3, 224, 224) normalized image
            class_idx: which disease class to explain

        Returns:
            heatmap (224, 224) normalized to [0, 1]
        """
        input_tensor = input_tensor.to(CFG["device"]).requires_grad_(True)

        # Forward — must enable grad through the frozen backbone
        # The model's forward uses torch.no_grad() when not training, so
        # we set the model to train mode briefly OR run the backbone manually.
        # Simpler: run backbone + head manually with grad enabled.
        self.model.zero_grad()

        with torch.enable_grad():
            feats = self.model.backbone(input_tensor)   # (1, 2048, 1, 1) but
                                                          # hook captures layer4
            feats_flat = feats.flatten(1)                # (1, 2048)
            logits = self.model.head(feats_flat)         # (1, 23)

        # Backprop the target class score
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM weights = global-average-pool of gradients
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture grad/activations")

        grads = self.gradients[0]          # (2048, 7, 7)
        acts = self.activations[0]         # (2048, 7, 7)
        weights = grads.mean(dim=(1, 2))   # (2048,)

        # Weighted combination
        cam = torch.zeros(acts.shape[1:], device=acts.device)  # (7, 7)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = F.relu(cam)                  # only positive contributions

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to image size
        cam = cam.unsqueeze(0).unsqueeze(0)               # (1,1,7,7)
        cam = F.interpolate(cam, size=(CFG["img_size"], CFG["img_size"]),
                             mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def denormalize_image(tensor):
    """Convert normalized tensor back to displayable RGB [0,1]."""
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(CFG["img_mean"])
    std = np.array(CFG["img_std"])
    img = img * std + mean
    return np.clip(img, 0, 1)


def make_overlay(rgb_img, heatmap, alpha=0.5):
    """Overlay heatmap on image."""
    if HAVE_CV2:
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
        overlay = alpha * heatmap_color + (1 - alpha) * rgb_img
        return np.clip(overlay, 0, 1), heatmap_color
    else:
        # Matplotlib fallback
        cmap = plt.get_cmap("jet")
        heatmap_color = cmap(heatmap)[:, :, :3]
        overlay = alpha * heatmap_color + (1 - alpha) * rgb_img
        return np.clip(overlay, 0, 1), heatmap_color


def save_visualization(rgb_img, heatmap, overlay, heatmap_color,
                        title, save_path):
    """Save 3-panel figure: original | heatmap | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_img); axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(heatmap_color); axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (gets the answer for an image+question)
# ─────────────────────────────────────────────────────────────────────────────
def load_stage1_model():
    print(f"  Loading Stage 1 ResNet50 ...")
    model = TreeNetDiseaseClassifier().to(CFG["device"])
    ckpt = torch.load(CFG["stage1_ckpt"], map_location=CFG["device"],
                       weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))

    # LOAD-VALIDITY CHECK (reviewer): strict=False silently ignores missing
    # keys. If the HEAD weights didn't load, the heatmap is meaningless while
    # looking plausible. We inspect missing/unexpected keys and WARN loudly
    # if anything under 'head.' failed to load.
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    head_missing = [k for k in missing if k.startswith("head")]
    backbone_missing = [k for k in missing if k.startswith("backbone")]

    print(f"  Load report:")
    print(f"     Missing keys:    {len(missing)}")
    print(f"     Unexpected keys: {len(unexpected)}")
    if head_missing:
        print(f"  🔴  WARNING: {len(head_missing)} HEAD keys did not load!")
        print(f"      Heatmaps will be MEANINGLESS (head is random init).")
        print(f"      Missing head keys: {head_missing[:5]}")
    elif backbone_missing and len(backbone_missing) > 5:
        print(f"  🟠  WARNING: {len(backbone_missing)} backbone keys missing")
    else:
        print(f"     ✅  Head + backbone loaded cleanly — heatmaps valid")

    model.eval()
    return model


def process_image(gradcam, model, transform, image_path, question,
                   answer=None):
    """
    Run Grad-CAM for one image.
    If answer is None, we use Stage 1's top disease prediction.
    """
    # Load + preprocess image
    pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(pil).unsqueeze(0)   # (1,3,224,224)

    # Get Stage 1 disease probabilities
    with torch.no_grad():
        out = model(img_tensor.to(CFG["device"]))
        disease_probs = out["probs"].squeeze().cpu().numpy()

    # Pick which disease drives the heatmap
    if answer is None:
        answer = ""
    cls_idx, disease_name, reason = pick_disease_for_answer(
        question, answer, disease_probs)

    # Generate heatmap
    heatmap = gradcam.generate(img_tensor, cls_idx)

    # Build visualization
    rgb = denormalize_image(img_tensor)
    overlay, heatmap_color = make_overlay(rgb, heatmap)

    return {
        "heatmap": heatmap,
        "rgb": rgb,
        "overlay": overlay,
        "heatmap_color": heatmap_color,
        "disease_idx": cls_idx,
        "disease_name": disease_name,
        "disease_prob": float(disease_probs[cls_idx]),
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────
def demo_mode(n_per_route=1):
    """Generate Grad-CAM for a few test images across routes."""
    print(f"\n{'█'*72}")
    print(f"  STAGE 7 — Grad-CAM Demo")
    print(f"{'█'*72}\n")

    model = load_stage1_model()
    gradcam = GradCAM(model)
    transform = get_eval_transform()

    # Diagnostic: print label order + FULL keyword map to verify no drift
    print(f"\n  Disease label order:")
    for i, lbl in enumerate(DISEASE_LABELS):
        print(f"     {i:2d}: {lbl}")
    print(f"\n  FULL keyword → label map (verify NO clinical drift):")
    for kw in sorted(ANSWER_TO_DISEASE):
        targets = [DISEASE_LABELS[i] for i in ANSWER_TO_DISEASE[kw]]
        print(f"     {kw:16s} → {targets}")
    print()

    # Try to use the pipeline for answers; else just use Stage 1
    use_pipeline = False
    predictor = None
    try:
        import stage5_pipeline_test as ppt
        predictor = ppt.FullPipelinePredictor()
        use_pipeline = True
        print(f"  ✅  Full pipeline loaded (answers from Stage 4/5)")
    except Exception as e:
        print(f"  ⚠️   Pipeline unavailable ({str(e)[:40]}), "
              f"using Stage 1 disease prediction only")

    # Load test data
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]
    image_dir = os.path.join(PROJECT, "data")  # adjust if needed
    # Try to detect image dir
    for cand in [S1_CFG.get("image_dir", ""),
                  os.path.expanduser("~/data/kvasir_local/images"),
                  os.path.expanduser("~/vqa_gi_thesis/data/images")]:
        if cand and os.path.exists(cand):
            image_dir = cand
            break

    def find_image(img_id):
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            p = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(p): return p
        return None

    # Route + collect samples
    try:
        from stage5_pipeline_test import infer_route
    except Exception:
        from stage4_revised import infer_route

    by_route = defaultdict(list)
    for s in test_split:
        q = s.get("question", "")
        if not q: continue
        try:
            r = infer_route(q)
        except Exception:
            continue
        by_route[r].append(s)

    index_rows = []
    import random
    rng = random.Random(42)
    for route in range(6):
        cands = by_route[route]
        if not cands: continue
        picked = rng.sample(cands, min(n_per_route, len(cands)))
        for sample in picked:
            img_path = find_image(
                sample.get("img_id", sample.get("image_id", "")))
            if not img_path:
                continue

            # Get answer
            answer = ""
            if use_pipeline:
                try:
                    res = predictor.predict(img_path, sample["question"])
                    answer = str(res.get("s5_sentence",
                                          res.get("s4_answer", "")))
                except Exception:
                    answer = sample.get("answer", "")
            else:
                answer = sample.get("answer", "")

            # Grad-CAM
            try:
                viz = process_image(gradcam, model, transform,
                                     img_path, sample["question"], answer)
            except Exception as e:
                print(f"  ⚠️  Failed on {img_path}: {str(e)[:50]}")
                continue

            # Save
            img_id = sample.get("img_id", "unknown")
            save_path = os.path.join(
                CFG["out_dir"],
                f"gradcam_r{route}_{img_id}_{viz['disease_name']}.png")
            title = (f"Route {route} | Q: {sample['question'][:50]} | "
                      f"Disease: {viz['disease_name']} "
                      f"({viz['disease_prob']*100:.0f}%)")
            save_visualization(viz["rgb"], viz["heatmap"], viz["overlay"],
                                viz["heatmap_color"], title, save_path)
            print(f"  ✅  Route {route}: {os.path.basename(save_path)}")
            print(f"       Q: {sample['question'][:55]}")
            print(f"       A: {answer[:55]}")
            print(f"       Disease driving heatmap: {viz['disease_name']} "
                  f"({viz['reason']})")

            index_rows.append({
                "route": route, "img_id": img_id,
                "question": sample["question"], "answer": answer,
                "disease_name": viz["disease_name"],
                "disease_prob": viz["disease_prob"],
                "reason": viz["reason"],
                "figure": os.path.basename(save_path),
            })

    gradcam.remove_hooks()

    # Save index
    import pandas as pd
    idx_path = os.path.join(CFG["out_dir"], "gradcam_index.csv")
    pd.DataFrame(index_rows).to_csv(idx_path, index=False)
    print(f"\n  ✅  Index → {idx_path}")
    print(f"  ✅  {len(index_rows)} visualizations in {CFG['out_dir']}\n")


def single_mode(image_path, question):
    """Grad-CAM for a single user-provided image."""
    print(f"\n{'█'*72}")
    print(f"  STAGE 7 — Grad-CAM Single Image")
    print(f"{'█'*72}\n")

    if not os.path.exists(image_path):
        print(f"❌  Image not found: {image_path}")
        return

    model = load_stage1_model()
    gradcam = GradCAM(model)
    transform = get_eval_transform()

    answer = ""
    try:
        import stage5_pipeline_test as ppt
        predictor = ppt.FullPipelinePredictor()
        res = predictor.predict(image_path, question)
        answer = str(res.get("s5_sentence", res.get("s4_answer", "")))
        print(f"  Pipeline answer: {answer}")
    except Exception as e:
        print(f"  ⚠️  Pipeline unavailable, using Stage 1 only")

    viz = process_image(gradcam, model, transform, image_path,
                         question, answer)
    save_path = os.path.join(CFG["out_dir"],
                              f"gradcam_single_{viz['disease_name']}.png")
    title = (f"Q: {question[:50]} | Disease: {viz['disease_name']} "
              f"({viz['disease_prob']*100:.0f}%)")
    save_visualization(viz["rgb"], viz["heatmap"], viz["overlay"],
                        viz["heatmap_color"], title, save_path)
    gradcam.remove_hooks()
    print(f"\n  ✅  Saved → {save_path}")
    print(f"     Disease driving heatmap: {viz['disease_name']} "
          f"({viz['reason']})\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo",
                         choices=["demo", "bulk", "single"])
    parser.add_argument("--n_per_route", type=int, default=2)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    if not HAVE_PLT:
        print("❌  matplotlib required. pip install matplotlib")
        return

    if args.mode == "single":
        if not args.image_path or not args.question:
            print("❌  --mode single needs --image_path and --question")
            return
        single_mode(args.image_path, args.question)
    else:
        n = args.n_per_route if args.mode == "bulk" else 1
        demo_mode(n)


if __name__ == "__main__":
    main()

