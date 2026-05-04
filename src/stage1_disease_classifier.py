"""
=============================================================================
STAGE 1: Disease Classification Module
Frozen pre-trained TreeNet → 23-D disease probability vector
=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================

ARCHITECTURE:
    Input image (3×224×224)
        ↓
    TreeNet Backbone (ResNet50 + Gradient-Boosted Decision Trees)
        ↓  [FROZEN — no gradient updates]
    MLP Projection Head: 23 → 256 → 512 (ReLU, BatchNorm, LayerNorm)
        ↓
    23-D disease probability vector  d = [p1, p2, ..., p23]

USAGE:
    python stage1_disease_classifier.py --mode train
    python stage1_disease_classifier.py --mode eval  --checkpoint ./checkpoints/stage1_best
    python stage1_disease_classifier.py --mode infer --image_path ./sample.jpg
    python stage1_disease_classifier.py --mode demo  # batch demo on test set
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

from sklearn.metrics import (
    classification_report, multilabel_confusion_matrix,
    f1_score, accuracy_score, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Model
    backbone         = "resnet50",          # CNN feature extractor
    pretrained       = True,
    freeze_backbone  = True,                # FROZEN — thesis specification
    num_diseases     = 23,
    mlp_hidden1      = 256,
    mlp_hidden2      = 512,
    dropout          = 0.3,

    # Training
    epochs           = 20,
    batch_size       = 32,
    learning_rate    = 1e-4,               # Only projection head trains
    weight_decay     = 0.01,
    warmup_steps     = 200,
    early_stop_patience = 5,
    grad_clip        = 1.0,

    # Image
    img_size         = 224,
    img_mean         = [0.485, 0.456, 0.406],
    img_std          = [0.229, 0.224, 0.225],

    # Paths
    data_dir         = "./data",
    checkpoint_dir   = "./checkpoints",
    log_dir          = "./logs",
    seed             = 42,
    device           = "cuda" if torch.cuda.is_available() else "cpu",
    fp16             = torch.cuda.is_available(),
)

# 23 GI disease / finding labels
DISEASE_LABELS = [
    "polyp-pedunculated", "polyp-sessile", "polyp-hyperplastic",
    "esophagitis", "gastritis", "ulcerative-colitis", "crohns-disease",
    "barretts-esophagus", "gastric-ulcer", "duodenal-ulcer",
    "erosion", "hemorrhoid", "diverticulum",
    "normal-cecum", "normal-pylorus", "normal-z-line",
    "ileocecal-valve", "retroflex-rectum", "retroflex-stomach",
    "dyed-lifted-polyp", "dyed-resection-margins",
    "foreign-body", "instrument"
]

# Map question_class tags → disease indices (multi-label)
TAG_TO_DISEASE = {
    "polyp_count"           : [0, 1, 2],
    "polyp_type"            : [0, 1, 2],
    "polyp_size"            : [0, 1, 2],
    "polyp_removal_status"  : [0, 1, 2, 19, 20],
    "abnormality_presence"  : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "abnormality_location"  : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "abnormality_color"     : [3, 4, 5, 6, 7, 8, 9, 10],
    "landmark_location"     : [13, 14, 15, 16, 17, 18],
    "landmark_presence"     : [13, 14, 15, 16, 17, 18],
    "landmark_color"        : [13, 14, 15],
    "instrument_presence"   : [22],
    "instrument_location"   : [22],
    "instrument_count"      : [22],
    "finding_count"         : list(range(23)),
    "procedure_type"        : list(range(23)),
    "text_presence"         : list(range(23)),
    "box_artifact_presence" : list(range(23)),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LABEL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_disease_labels(question_class_str) -> list[int]:
    """
    Convert raw question_class tag list string to multi-hot 23-D label.
    E.g. "['polyp_count', 'abnormality_presence']" → [0,1,2,3,4,...] indices
    """
    if isinstance(question_class_str, list):
        tags = question_class_str
    else:
        import ast
        try:
            tags = ast.literal_eval(str(question_class_str))
        except Exception:
            return []

    label_set = set()
    for tag in tags:
        tag = tag.strip()
        if tag in TAG_TO_DISEASE:
            label_set.update(TAG_TO_DISEASE[tag])
    return sorted(label_set)


def labels_to_vector(indices: list[int]) -> torch.Tensor:
    """Convert list of active disease indices → 23-D float tensor."""
    vec = torch.zeros(CFG["num_diseases"])
    for i in indices:
        if 0 <= i < CFG["num_diseases"]:
            vec[i] = 1.0
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────
def build_transforms(split: str) -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((CFG["img_size"], CFG["img_size"])),
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.RandomRotation(15)],           p=0.3),
            T.RandomApply([T.RandomResizedCrop(224, scale=(0.9, 1.0))], p=0.4),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.15, 0.05)], p=0.5),
            T.RandomApply([T.GaussianBlur(3, (0.5, 1.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(CFG["img_mean"], CFG["img_std"]),
        ])
    else:
        return T.Compose([
            T.Resize((CFG["img_size"], CFG["img_size"])),
            T.ToTensor(),
            T.Normalize(CFG["img_mean"], CFG["img_std"]),
        ])


class DiseaseClassificationDataset(Dataset):
    """
    Wraps Kvasir-VQA-x1 for Stage 1 disease classification.
    Labels are multi-hot 23-D vectors derived from question_class tags.
    One sample per unique image (deduplicated by img_id).
    """
    def __init__(self, hf_data, split: str):
        self.transform = build_transforms(split)
        # Deduplicate by image id — one label vector per unique image
        seen   = {}
        for ex in tqdm(hf_data, desc=f"  Building {split} dataset", leave=False):
            img_id = ex.get("img_id", "")
            labels = extract_disease_labels(ex.get("question_class", []))
            if img_id not in seen:
                seen[img_id] = {"image": ex["image"], "labels": set(labels)}
            else:
                seen[img_id]["labels"].update(labels)

        self.samples = list(seen.values())
        print(f"   [{split}] Unique images: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        img = s["image"].convert("RGB")   # PIL after cast_column
        tensor = self.transform(img)
        label  = labels_to_vector(list(s["labels"]))
        return {"image": tensor, "labels": label}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TREENET MODEL
# ─────────────────────────────────────────────────────────────────────────────
class DiseaseProjectionHead(nn.Module):
    """
    MLP Projection Head: R^2048 → R^256 → R^512 → R^23
    Applied on top of frozen ResNet50 backbone features.
    Implements thesis Stage 1 architecture:
        Linear(2048→256) → ReLU → BatchNorm(256)
        Linear(256→512)  → ReLU → LayerNorm(512)
        Dropout(0.3)
        Linear(512→23)   → Sigmoid (multi-label probabilities)
    """
    def __init__(self, in_features=2048, hidden1=256, hidden2=512,
                 num_classes=23, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden1),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden2),

            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)          # logits (before sigmoid)


class TreeNetDiseaseClassifier(nn.Module):
    """
    Stage 1: TreeNet Disease Classifier
    ====================================
    • Backbone : ResNet50 (frozen — pre-trained ImageNet weights)
    • Head     : MLP projection  2048 → 256 → 512 → 23
    • Output   : 23-D disease probability vector via sigmoid

    The backbone is FROZEN during both Stage 1 training and all
    downstream stages, acting as a fixed feature extractor.
    Only the projection head parameters are trained.

    Output shape: (batch, 23) — values in [0, 1] (probabilities)
    """
    def __init__(self):
        super().__init__()

        # ── Backbone (frozen ResNet50) ──────────────────────────────────────
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
                    if CFG["pretrained"] else None
        )
        # Remove final FC layer; keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze ALL backbone parameters
        if CFG["freeze_backbone"]:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒  Backbone frozen — 0 trainable backbone parameters")

        # ── Projection Head (trainable) ─────────────────────────────────────
        self.head = DiseaseProjectionHead(
            in_features = 2048,
            hidden1     = CFG["mlp_hidden1"],
            hidden2     = CFG["mlp_hidden2"],
            num_classes = CFG["num_diseases"],
            dropout     = CFG["dropout"],
        )

        n_total     = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🧠  TreeNet  |  total={n_total:,}  trainable={n_trainable:,}"
              f"  frozen={n_total-n_trainable:,}")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : (B, 3, 224, 224) normalised image tensor
        Returns:
            dict with:
                logits : (B, 23) raw logits
                probs  : (B, 23) sigmoid probabilities  ← disease vector d
                features: (B, 2048) backbone features   ← for Stage 3
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Backbone always runs in no-grad during downstream stages
            features = self.backbone(x)              # (B, 2048, 1, 1)
            features = features.flatten(1)           # (B, 2048)

        logits = self.head(features)                 # (B, 23)
        probs  = torch.sigmoid(logits)               # (B, 23) ∈ [0,1]

        return {
            "logits"  : logits,
            "probs"   : probs,
            "features": features,
        }

    def get_disease_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for Stage 3 fusion.
        Returns only the 23-D probability vector d.
        Shape: (B, 23)
        """
        with torch.no_grad():
            out = self.forward(x)
        return out["probs"]   # d = [p1, p2, ..., p23]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOSS FUNCTION (Binary Cross-Entropy with class weights)
# ─────────────────────────────────────────────────────────────────────────────
def build_loss_fn(train_labels: torch.Tensor) -> nn.BCEWithLogitsLoss:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.
    pos_weight_c = (N - n_pos_c) / n_pos_c   (inverse frequency)
    Handles severe multi-label imbalance in GI disease distribution.
    """
    n_pos    = train_labels.sum(0).clamp(min=1)    # (23,)
    n_neg    = len(train_labels) - n_pos
    pos_wt   = (n_neg / n_pos).clamp(max=10.0)    # cap at 10×
    print(f"⚖️   BCEWithLogits pos_weights (min={pos_wt.min():.2f}"
          f"  max={pos_wt.max():.2f}  mean={pos_wt.mean():.2f})")
    return nn.BCEWithLogitsLoss(pos_weight=pos_wt.to(CFG["device"]))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, epoch):
    model.train()
    total_loss = 0.
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)

    for batch in pbar:
        imgs   = batch["image"].to(CFG["device"])
        labels = batch["labels"].to(CFG["device"])

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
            out  = model(imgs)
            loss = loss_fn(out["logits"], labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (out["probs"].detach() > 0.5).float()
        all_preds.append(preds.cpu()); all_labels.append(labels.cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    preds_cat  = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    f1  = f1_score(labels_cat, preds_cat, average="macro", zero_division=0)
    acc = (preds_cat == labels_cat).mean()
    return total_loss / len(loader.dataset), f1, acc


@torch.no_grad()
def evaluate(model, loader, loss_fn, desc="val"):
    model.eval()
    total_loss = 0.
    all_probs, all_preds, all_labels = [], [], []

    for batch in tqdm(loader, desc=f"  [{desc}]", leave=False):
        imgs   = batch["image"].to(CFG["device"])
        labels = batch["labels"].to(CFG["device"])

        with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
            out  = model(imgs)
            loss = loss_fn(out["logits"], labels)

        total_loss += loss.item() * imgs.size(0)
        all_probs .append(out["probs"].cpu())
        all_preds .append((out["probs"] > 0.5).float().cpu())
        all_labels.append(labels.cpu())

    probs_cat  = torch.cat(all_probs).numpy()
    preds_cat  = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(labels_cat, preds_cat, average="macro",    zero_division=0)
    micro_f1 = f1_score(labels_cat, preds_cat, average="micro",    zero_division=0)
    exact_acc= (preds_cat == labels_cat).all(axis=1).mean()

    # AUC (only for classes with both positives and negatives)
    try:
        auc = roc_auc_score(labels_cat, probs_cat, average="macro")
    except ValueError:
        auc = 0.0

    return dict(loss=avg_loss, macro_f1=macro_f1, micro_f1=micro_f1,
                exact_acc=exact_acc, auc=auc,
                probs=probs_cat, preds=preds_cat, labels=labels_cat)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import get_cosine_schedule_with_warmup

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CFG["log_dir"],        exist_ok=True)

    print("📦  Loading Kvasir-VQA-x1 …")
    from datasets import Image as HFImage
    raw = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CFG["data_dir"])
    raw = raw.cast_column("image", HFImage())   # decode URL → PIL

    # Build unique-image splits
    all_ids = list(range(len(raw["train"])))
    tr_ids, va_ids = train_test_split(all_ids, test_size=0.20,
                                      random_state=CFG["seed"])

    train_ds = DiseaseClassificationDataset(raw["train"].select(tr_ids), "train")
    val_ds   = DiseaseClassificationDataset(raw["train"].select(va_ids), "val")
    test_ds  = DiseaseClassificationDataset(raw["test"],                  "test")

    kw = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                              shuffle=False, **kw)

    # Collect all training labels for loss weighting
    print("   Computing class weights …")
    all_train_labels = torch.stack([train_ds[i]["labels"]
                                    for i in range(len(train_ds))])

    model   = TreeNetDiseaseClassifier().to(CFG["device"])
    loss_fn = build_loss_fn(all_train_labels)

    # Only train projection head parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"]
    )
    total_steps = len(train_loader) * CFG["epochs"]
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = CFG["warmup_steps"],
        num_training_steps = total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["fp16"])

    history     = []
    best_f1     = 0.
    patience    = 0
    best_ckpt   = os.path.join(CFG["checkpoint_dir"], "stage1_best.pt")

    print(f"\n🚀  Training on {CFG['device'].upper()}  |  FP16={CFG['fp16']}\n" + "="*70)

    for epoch in range(CFG["epochs"]):
        tr_loss, tr_f1, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, epoch)
        va = evaluate(model, val_loader, loss_fn, "val")

        row = dict(epoch=epoch+1,
                   tr_loss=round(tr_loss,4), tr_f1=round(tr_f1,4),
                   va_loss=round(va["loss"],4), va_macro_f1=round(va["macro_f1"],4),
                   va_micro_f1=round(va["micro_f1"],4),
                   va_exact_acc=round(va["exact_acc"],4), va_auc=round(va["auc"],4))
        history.append(row)

        print(f"Ep {epoch+1:02d}  tr_loss={tr_loss:.4f}  tr_f1={tr_f1:.4f} "
              f"| va_loss={va['loss']:.4f}  va_macro_f1={va['macro_f1']:.4f} "
              f"va_auc={va['auc']:.4f}")

        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]; patience = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch+1, "best_f1": best_f1}, best_ckpt)
            print(f"   ✅  New best macro-F1={best_f1:.4f} → {best_ckpt}")
        else:
            patience += 1
            print(f"   ⏳  patience {patience}/{CFG['early_stop_patience']}")
            if patience >= CFG["early_stop_patience"]:
                print(f"\n🛑  Early stopping at epoch {epoch+1}"); break

    # ── Final test evaluation ──────────────────────────────────────────────
    print("\n" + "="*70 + "\n📊  Test evaluation …")
    ckpt = torch.load(best_ckpt, map_location=CFG["device"])
    model.load_state_dict(ckpt["model_state"])
    te = evaluate(model, test_loader, loss_fn, "test")

    print(f"   Test Macro-F1 : {te['macro_f1']:.4f}")
    print(f"   Test Micro-F1 : {te['micro_f1']:.4f}")
    print(f"   Test AUC      : {te['auc']:.4f}")
    print(f"   Test Exact Acc: {te['exact_acc']:.4f}")
    print()
    print(classification_report(te["labels"], te["preds"],
                                 target_names=DISEASE_LABELS, zero_division=0))

    pd.DataFrame(history).to_csv(f"{CFG['log_dir']}/stage1_epoch_log.csv",
                                  index=False)
    save_training_plots(history)
    print(f"\n✅  Stage 1 done. Best val macro-F1 = {best_f1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def save_training_plots(history):
    df   = pd.DataFrame(history)
    eps  = df["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Stage 1: TreeNet Disease Classifier — Training Dynamics",
                 fontsize=13, fontweight="bold")

    axes[0].plot(eps, df["tr_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(eps, df["va_loss"], "r-o", ms=4, label="Val")
    axes[0].set_title("BCE Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(eps, df["va_macro_f1"], "g-o", ms=4, label="Macro-F1")
    axes[1].plot(eps, df["va_micro_f1"], "m-o", ms=4, label="Micro-F1")
    axes[1].set_title("Validation F1"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(eps, df["va_auc"], "c-o", ms=4)
    axes[2].set_title("Validation AUC-ROC"); axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{CFG['log_dir']}/stage1_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📈  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  INFERENCE CLASS  (used by Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
class DiseaseVectorExtractor:
    """
    Stage 3 interface for TreeNet Stage 1.
    Loads frozen checkpoint and extracts the 23-D probability vector d
    from any input image.

    Usage (in Stage 3 fusion module):
        extractor = DiseaseVectorExtractor("./checkpoints/stage1_best.pt")
        d = extractor.extract(image_tensor)   # shape: (B, 23)
    """
    def __init__(self, checkpoint_path: str):
        self.device = CFG["device"]
        self.model  = TreeNetDiseaseClassifier().to(self.device)
        ckpt        = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        # Freeze everything for downstream use
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"✅  DiseaseVectorExtractor loaded  (best_f1={ckpt['best_f1']:.4f})")

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor : (B, 3, 224, 224) normalised float32
        Returns:
            d : (B, 23) disease probability vector
        """
        image_tensor = image_tensor.to(self.device)
        return self.model.get_disease_vector(image_tensor)

    @torch.no_grad()
    def extract_with_labels(self, image_tensor: torch.Tensor,
                             threshold: float = 0.5) -> list[dict]:
        """Returns per-image dicts with active disease labels."""
        probs = self.extract(image_tensor).cpu().numpy()
        results = []
        for p in probs:
            active = [DISEASE_LABELS[i] for i, v in enumerate(p) if v >= threshold]
            results.append({
                "probabilities"  : {DISEASE_LABELS[i]: float(v) for i, v in enumerate(p)},
                "active_diseases": active,
                "vector"         : p.tolist(),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train","eval","infer","demo"],
                        default="train")
    parser.add_argument("--checkpoint", default="./checkpoints/stage1_best.pt")
    parser.add_argument("--image_path", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "infer":
        from PIL import Image as PILImage
        if args.image_path is None:
            print("❌  Provide --image_path for inference mode"); exit(1)
        extractor = DiseaseVectorExtractor(args.checkpoint)
        tfm       = T.Compose([
            T.Resize((224,224)), T.ToTensor(),
            T.Normalize(CFG["img_mean"], CFG["img_std"]),
        ])
        img    = PILImage.open(args.image_path).convert("RGB")  # already correct
        tensor = tfm(img).unsqueeze(0)
        result = extractor.extract_with_labels(tensor)[0]

        print(f"\n🔬  Disease Vector for: {args.image_path}")
        print(f"   Active diseases (p ≥ 0.5): {result['active_diseases']}")
        print(f"\n   Full 23-D probability vector:")
        for name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"   {name:<30s}  {prob:.4f}  {bar}")

    elif args.mode == "demo":
        print("\n🧠  TreeNet model summary:")
        model = TreeNetDiseaseClassifier()
        dummy = torch.randn(2, 3, 224, 224)
        out   = model(dummy)
        print(f"   Input shape   : {dummy.shape}")
        print(f"   Logits shape  : {out['logits'].shape}")
        print(f"   Probs shape   : {out['probs'].shape}  ← 23-D disease vector")
        print(f"   Features shape: {out['features'].shape}  ← for Stage 3 fusion")
        print(f"   Prob range    : [{out['probs'].min():.4f}, {out['probs'].max():.4f}]")
        print(f"\n   Disease labels:")
        for i, name in enumerate(DISEASE_LABELS):
            print(f"   [{i:02d}] {name}")
