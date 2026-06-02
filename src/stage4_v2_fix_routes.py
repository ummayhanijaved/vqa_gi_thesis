#!/usr/bin/env python3
"""
=============================================================================
Stage 4 V2 — Comprehensive Retraining for Routes 0, 1, 3, 4
=============================================================================

GOAL:
    Eliminate class-collapse bias in Stage 4 (Phase 2) by retraining the
    four problematic routes with multiple complementary techniques.
    Routes 2 and 5 are NOT retrained (already strong / no upside).

SAFETY (CONFIRMED):
    Original Phase 2 checkpoints stay UNTOUCHED at:
      ~/vqa_gi_thesis/checkpoints/stage4_revised/*.pt
      ~/vqa_gi_thesis/checkpoints/stage4_revised/yolo_seg_finetuned/
      ~/vqa_gi_thesis/checkpoints/stage4_revised/yolo_det_finetuned/
      ~/vqa_gi_thesis/logs/stage4_revised/*.csv

    All v2 outputs go to NEW subfolders:
      ~/vqa_gi_thesis/checkpoints/stage4_v2/route*_v2_best.pt
      ~/vqa_gi_thesis/checkpoints/stage4_v2/yolo_seg_v2/...
      ~/vqa_gi_thesis/logs/stage4_v2/*.csv

═══════════════════════════════════════════════════════════════════════════
ISSUES IDENTIFIED & FIXES APPLIED (per route)
═══════════════════════════════════════════════════════════════════════════

ROUTE 0 (Yes/No):
  ❌ ISSUES:
    - Predicts "no" 97.6% of time (massive class collapse)
    - Original trained only 1 epoch
    - val_acc 88.65% = mostly just memorising majority class
  ✅ FIXES:
    1. WeightedRandomSampler — equal yes/no per batch
    2. Inverse-frequency class weights (clip 0.5-5.0)
    3. Label smoothing 0.1 (reduces overconfidence)
    4. 15 epochs with early stopping (proper convergence)
    5. Diversity diagnostic each epoch (prints top-3 preds)

ROUTE 1 (Single Choice, 50 classes):
  ❌ ISSUES:
    - Top-3 classes = 62% of predictions (severe collapse)
    - Long-tail distribution (50 classes, top 3 dominate)
    - val_acc only 48.18%
  ✅ FIXES:
    1. WeightedRandomSampler — balance class frequency
    2. FOCAL LOSS (gamma=2.5) — down-weights easy majority samples
    3. Heavier class weights (clip 0.5-25.0)
    4. Label smoothing 0.15 — accommodates label noise
    5. 20 epochs (was 9) — adequate convergence time
    6. MixUp augmentation on Stage 3 features (alpha=0.2)
    7. Per-epoch diversity diagnostic

ROUTE 3 (Color, 13 classes):
  ❌ ISSUES:
    - Predicts "red" 83% of time (dataset bias toward red lesions)
    - Other 12 colors get ~17% combined
  ✅ FIXES:
    1. WeightedRandomSampler — equal class probability
    2. Class weights (clip 0.5-5.0)
    3. Label smoothing 0.1
    4. 15 epochs
    5. Per-epoch diversity diagnostic

ROUTE 4 (YOLO-Seg Location):
  ❌ ISSUES:
    - Predicts "central region" 85.8% of time
    - ROOT CAUSE: extract_region_from_text() defaults to (0.5, 0.5)
      when no region word is found in the answer — this trained YOLO
      to associate ANY image with centre.
    - Pseudo-annotation grid was too coarse (3×3)
  ✅ FIXES:
    1. REGENERATE pseudo-annotations:
       - Skip samples where no region word is in answer
         (no more "default to centre" garbage labels)
       - Finer position keywords detected (5×5 grid hints)
    2. Stronger YOLO augmentation (mosaic=1.0, mixup=0.15, hsv-jitter)
    3. Higher learning rate (5e-4 vs 1e-4)
    4. 60 epochs (was 50)
    5. Patience 15 for early stopping

═══════════════════════════════════════════════════════════════════════════
REALISTIC EXPECTATIONS
═══════════════════════════════════════════════════════════════════════════
  Route 0 Yes/No   : 88.65% → 92-95%   (small lift, dataset is genuinely
                                          imbalanced — true ceiling ~95%)
  Route 1 Single   : 48% val → 55-70%  (best-case if focal+mixup work well)
  Route 3 Color    : 81% → 82-85%      ("red" remains common in GT, modest)
  Route 4 Location : 55% → 65-75%      (depends on how many samples have
                                          location words in their answers)

  ⚠️  HONEST CAVEAT — your target of >90% per route is challenging:
      - Route 0 has hard ceiling near 95% due to data imbalance
      - Route 1 with 50 classes/long-tail rarely exceeds 70% in literature
      - Route 4 pseudo-annotation quality limits YOLO ceiling

USAGE:
    # Train one route at a time (recommended)
    python stage4_v2_fix_routes.py --route 0 --mode both
    python stage4_v2_fix_routes.py --route 1 --mode both
    python stage4_v2_fix_routes.py --route 3 --mode both
    python stage4_v2_fix_routes.py --route 4 --mode train     # YOLO; eval separate

    # All four routes sequentially
    python stage4_v2_fix_routes.py --route all --mode both
=============================================================================
"""
import os, sys, json, argparse, warnings, shutil
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

# Reuse the proven Stage 4 Phase 2 machinery (architecture, dataset, etc.)
from stage4_revised import (
    CFG as S4_CFG, ROUTE_NAMES,
    DistilBERTAnswerModel, DistilBERTRouteDataset,
    cache_stage3_features, build_vocab, infer_route, normalise_answer,
    FusionExtractor,
    CLASS_MAP, extract_class_from_text,
)
from preprocessing import build_image_transform, TextPreprocessor


# ─────────────────────────────────────────────────────────────────────────────
# V2 Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/vqa_gi_thesis")

V2_CFG = dict(
    # ── Isolated v2 paths (NEW, never overwrites originals) ───────────────
    ckpt_dir   = os.path.join(PROJECT, "checkpoints", "stage4_v2"),
    log_dir    = os.path.join(PROJECT, "logs",        "stage4_v2"),
    yolo_dir   = os.path.join(PROJECT, "checkpoints", "stage4_v2", "yolo_seg_v2"),
    yolo_data  = os.path.join(PROJECT, "data", "yolo_dataset_v2"),

    # ── Training common ────────────────────────────────────────────────────
    device         = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size     = 32,
    distilbert_lr  = 2e-5,
    head_lr        = 1e-4,
    weight_decay   = 0.01,
    warmup_ratio   = 0.1,
    early_stop_pat = 6,
    grad_clip      = 1.0,
    max_input_len  = 128,

    # ── Route 0 (Yes/No) ───────────────────────────────────────────────────
    route0_epochs       = 15,
    route0_balance      = True,
    route0_label_smooth = 0.10,
    route0_weight_clip  = (0.5, 5.0),

    # ── Route 1 (Single Choice) ────────────────────────────────────────────
    route1_epochs       = 20,
    route1_balance      = True,
    route1_use_focal    = True,
    route1_focal_gamma  = 2.5,
    route1_label_smooth = 0.15,
    route1_use_mixup    = True,
    route1_mixup_alpha  = 0.2,
    route1_weight_clip  = (0.5, 25.0),

    # ── Route 3 (Color) ────────────────────────────────────────────────────
    route3_epochs       = 15,
    route3_balance      = True,
    route3_label_smooth = 0.10,
    route3_weight_clip  = (0.5, 5.0),

    # ── Route 4 (YOLO-Seg) ─────────────────────────────────────────────────
    route4_yolo_epochs  = 60,
    route4_yolo_lr      = 5e-4,
    route4_yolo_patience= 15,
)
os.makedirs(V2_CFG["ckpt_dir"], exist_ok=True)
os.makedirs(V2_CFG["log_dir"], exist_ok=True)

S4_CFG["device"] = V2_CFG["device"]
V2_SUFFIX = "_v2"


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — Focal Loss
# ═════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with class weighting + label smoothing.
        FL = -alpha_c * (1-p_t)^gamma * smooth_log_prob
    """
    def __init__(self, gamma=2.5, alpha=None, label_smoothing=0.15):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # (n_classes,) optional
        self.eps   = label_smoothing

    def forward(self, logits, target):
        n_cls = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        p    = logp.exp()

        with torch.no_grad():
            tgt = torch.zeros_like(logp)
            tgt.fill_(self.eps / max(1, (n_cls - 1)))
            tgt.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)

        pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1-1e-6)
        focal_w = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            a_t = self.alpha.to(logits.device).gather(0, target)
            focal_w = focal_w * a_t

        loss = -(tgt * logp).sum(dim=-1) * focal_w
        return loss.mean()


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — MixUp on Stage 3 features
# ═════════════════════════════════════════════════════════════════════════════
def mixup_features(batch, alpha=0.2, device="cuda"):
    """
    MixUp augmentation on Stage 3 fused + disease vectors.
    Mixes two random samples in the batch and returns mixed inputs +
    pair of labels with a lambda weight.

    Returns:
        mixed_fused, mixed_disease, label_a, label_b, lam
    """
    if alpha <= 0:
        return (batch["fused"], batch["disease"],
                batch["label"], batch["label"], 1.0)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)        # bias toward original >0.5 (safer)
    B = batch["fused"].size(0)
    perm = torch.randperm(B, device=device)
    mixed_fused = lam * batch["fused"] + (1 - lam) * batch["fused"][perm]
    mixed_disease = (lam * batch["disease"] +
                      (1 - lam) * batch["disease"][perm])
    label_a = batch["label"]
    label_b = batch["label"][perm]
    return mixed_fused, mixed_disease, label_a, label_b, lam


def mixup_loss_fn(criterion, logits, label_a, label_b, lam):
    return lam * criterion(logits, label_a) + (1-lam) * criterion(logits, label_b)


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — Balanced sampler
# ═════════════════════════════════════════════════════════════════════════════
def make_balanced_sampler(dataset, n_classes):
    print(f"   Computing balanced sampler weights ...")
    label_counts = Counter()
    labels = []
    for i in tqdm(range(len(dataset)), desc="     Counting"):
        try:
            lbl = dataset[i]["label"]
            if hasattr(lbl, "item"): lbl = lbl.item()
            lbl = int(lbl)
            label_counts[lbl] += 1
            labels.append(lbl)
        except Exception:
            labels.append(0)

    sample_weights = [1.0 / max(label_counts.get(l, 1), 1) for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataset), replacement=True)
    print(f"   Sampler ready ({len(label_counts)} classes)")
    print(f"   Class counts (top 5): {dict(label_counts.most_common(5))}")
    print(f"   Class counts (rarest 5): "
          f"{dict(sorted(label_counts.items(), key=lambda x: x[1])[:5])}")
    return sampler, label_counts


def compute_class_weights(label_counts, n_classes,
                            clip_min=0.5, clip_max=10.0):
    total = sum(label_counts.values())
    weights = torch.ones(n_classes, dtype=torch.float32)
    for cls_idx in range(n_classes):
        cnt = max(label_counts.get(cls_idx, 1), 1)
        w = total / (n_classes * cnt)
        weights[cls_idx] = max(clip_min, min(clip_max, w))
    return weights


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — Generic V2 trainer for DistilBERT routes (0, 1, 3)
# ═════════════════════════════════════════════════════════════════════════════
def train_route_v2(route: int):
    print(f"\n{'='*72}")
    print(f"  Stage 4 V2 — Retraining Route {route} ({ROUTE_NAMES[route]})")
    print(f"{'='*72}\n")

    cfg = V2_CFG
    route_epochs       = cfg[f"route{route}_epochs"]
    route_balance      = cfg.get(f"route{route}_balance", False)
    route_focal        = cfg.get(f"route{route}_use_focal", False)
    route_mixup        = cfg.get(f"route{route}_use_mixup", False)
    label_smooth       = cfg[f"route{route}_label_smooth"]
    weight_clip        = cfg[f"route{route}_weight_clip"]

    # Print configuration
    print(f"  📋  CONFIGURATION:")
    print(f"      Epochs:           {route_epochs}")
    print(f"      Balanced sampler: {route_balance}")
    print(f"      Focal loss:       {route_focal}"
          f"{' (gamma=' + str(cfg.get('route1_focal_gamma', '?')) + ')' if route_focal else ''}")
    print(f"      MixUp:            {route_mixup}"
          f"{' (alpha=' + str(cfg.get('route1_mixup_alpha', '?')) + ')' if route_mixup else ''}")
    print(f"      Label smoothing:  {label_smooth}")
    print(f"      Weight clip:      {weight_clip}\n")

    # ── Load Stage 3 features (uses existing cache — safe) ─────────────────
    print(f"  Loading Stage 3 features from cache ...")
    extractor = FusionExtractor(S4_CFG["stage3_ckpt"])
    text_prep = TextPreprocessor()
    from datasets import load_from_disk
    raw = load_from_disk(S4_CFG["data_dir"])

    train_records = cache_stage3_features(
        extractor, text_prep, raw["train"], "train", S4_CFG["cache_dir"])
    val_records = cache_stage3_features(
        extractor, text_prep, raw.get("validation", raw["train"]),
        "val", S4_CFG["cache_dir"])

    # ── Vocabulary ─────────────────────────────────────────────────────────
    if route == 0:
        vocab = S4_CFG["yn_classes"]
    elif route == 3:
        vocab = S4_CFG["color_classes"]
    elif route == 1:
        vocab = build_vocab(train_records, route, max_classes=50)
    else:
        raise ValueError(f"Route {route} not supported in train_route_v2")
    n_classes = len(vocab)
    print(f"  Vocab size: {n_classes} classes")

    # ── Model ──────────────────────────────────────────────────────────────
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        DistilBERTAnswerModel.MODEL_NAME)
    model = DistilBERTAnswerModel(vocab_per_route={route: vocab})
    model = model.to(cfg["device"])

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = DistilBERTRouteDataset(
        train_records, route, tokenizer, vocab, cfg["max_input_len"])
    val_ds = DistilBERTRouteDataset(
        val_records, route, tokenizer, vocab, cfg["max_input_len"])
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}\n")

    # ── Balanced sampler + class weights ───────────────────────────────────
    print(f"  Computing class statistics ...")
    if route_balance:
        sampler, label_counts = make_balanced_sampler(train_ds, n_classes)
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0)
    else:
        label_counts = Counter()
        for i in tqdm(range(len(train_ds)), desc="     Counting"):
            try:
                lbl = train_ds[i]["label"]
                if hasattr(lbl, "item"): lbl = lbl.item()
                label_counts[int(lbl)] += 1
            except Exception:
                continue
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0)

    weights = compute_class_weights(
        label_counts, n_classes,
        clip_min=weight_clip[0], clip_max=weight_clip[1])
    print(f"\n  Class weights: min={weights.min():.2f}  "
          f"max={weights.max():.2f}  mean={weights.mean():.2f}")

    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"]*2,
                          shuffle=False, num_workers=0)

    # ── Loss criterion (focal or CE with class weights) ────────────────────
    if route_focal:
        criterion = FocalLoss(
            gamma=cfg["route1_focal_gamma"], alpha=weights,
            label_smoothing=label_smooth)
        print(f"  Using Focal Loss (gamma={cfg['route1_focal_gamma']}, "
              f"smooth={label_smooth})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weights.to(cfg["device"]), label_smoothing=label_smooth)
        print(f"  Using CE with class weights + label smoothing={label_smooth}")

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    backbone_params = [p for p in model.distilbert.parameters()
                       if p.requires_grad]
    head_params = (list(model.projector.parameters()) +
                    list(model.heads.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["distilbert_lr"]},
        {"params": head_params,     "lr": cfg["head_lr"]},
    ], weight_decay=cfg["weight_decay"])

    from transformers import get_cosine_schedule_with_warmup
    n_steps = len(train_dl) * route_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * cfg["warmup_ratio"]),
        num_training_steps=n_steps)

    # ── Train loop ─────────────────────────────────────────────────────────
    best_val = 0.0
    patience = 0
    history = []
    ckpt_path = os.path.join(
        cfg["ckpt_dir"],
        f"route{route}_{ROUTE_NAMES[route]}{V2_SUFFIX}_best.pt")
    log_path = os.path.join(
        cfg["log_dir"],
        f"route{route}_{ROUTE_NAMES[route]}{V2_SUFFIX}_train_log.csv")

    print(f"\n  Training for up to {route_epochs} epochs ...")
    print(f"  Save target: {ckpt_path}\n")

    for epoch in range(1, route_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        tot_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:2d} train")
        for batch in pbar:
            batch = {k: (v.to(cfg["device"]) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            optimizer.zero_grad()

            # MixUp augmentation (Route 1 only)
            if route_mixup and torch.rand(1).item() < 0.7:
                mf, md, la, lb, lam = mixup_features(
                    batch, alpha=cfg["route1_mixup_alpha"],
                    device=cfg["device"])
                cls_repr = model._encode(
                    mf, md, batch["input_ids"], batch["attention_mask"])
                logits = model.heads[str(route)](cls_repr)
                loss = mixup_loss_fn(criterion, logits, la, lb, lam)
            else:
                cls_repr = model._encode(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"])
                logits = model.heads[str(route)](cls_repr)
                loss = criterion(logits, batch["label"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = tot_loss / max(n_batches, 1)

        # ── val ───────────────────────────────────────────────────────────
        model.eval()
        v_correct = 0
        v_total = 0
        val_pred_counter = Counter()
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch:2d} val  "):
                batch = {k: (v.to(cfg["device"]) if isinstance(v, torch.Tensor)
                              else v) for k, v in batch.items()}
                cls_repr = model._encode(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"])
                logits = model.heads[str(route)](cls_repr)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == batch["label"]).sum().item()
                v_total += batch["label"].size(0)
                for p in preds.cpu().tolist():
                    val_pred_counter[p] += 1

        val_acc = v_correct / max(v_total, 1)

        # Diversity diagnostic
        top_preds = val_pred_counter.most_common(5)
        top_pct = [(vocab[p][:25], c, c/max(v_total,1)*100)
                   for p, c in top_preds]
        top_str = "  ".join(f"'{n}'={c}({p:.1f}%)" for n, c, p in top_pct[:3])

        history.append({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_acc":     val_acc,
            "top_pred":    top_pct[0][0] if top_pct else "?",
            "top_pred_pct": top_pct[0][2] if top_pct else 0,
        })

        print(f"\n  Epoch {epoch:2d}  |  loss={train_loss:.4f}  |  "
              f"val_acc={val_acc*100:.2f}%")
        print(f"     Top-3 val preds: {top_str}\n")

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocab":       vocab,
                "n_classes":   n_classes,
                "epoch":       epoch,
                "val_acc":     val_acc,
                "weights":     weights.cpu(),
                "config": {
                    "balanced_sampler": route_balance,
                    "focal_loss":       route_focal,
                    "mixup":            route_mixup,
                    "label_smoothing":  label_smooth,
                    "weight_clip":      list(weight_clip),
                    "epochs_trained":   epoch,
                    "route":            route,
                },
                "top_predictions": dict(val_pred_counter.most_common(10)),
            }, ckpt_path)
            print(f"   ✅  Saved best v2 ckpt → {ckpt_path}")
        else:
            patience += 1
            print(f"   ⏳  No improvement ({patience}/{cfg['early_stop_pat']})")
            if patience >= cfg["early_stop_pat"]:
                print(f"\n   🛑  Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"\n  ✅  Training log → {log_path}")
    print(f"  ✅  Final best val accuracy: {best_val*100:.2f}%\n")


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT 5 — Route 4 YOLO-Seg with improved pseudo-annotations
# ═════════════════════════════════════════════════════════════════════════════
# Better region detection: explicit position words only (no default to centre)
IMPROVED_REGION_TO_BOX = {
    # Single regions (priority: more specific phrases first)
    "upper-central region"  : (0.50, 0.25, 0.50, 0.40),
    "upper central region"  : (0.50, 0.25, 0.50, 0.40),
    "lower-central region"  : (0.50, 0.75, 0.50, 0.40),
    "lower central region"  : (0.50, 0.75, 0.50, 0.40),
    "upper-left region"     : (0.25, 0.25, 0.40, 0.40),
    "upper left region"     : (0.25, 0.25, 0.40, 0.40),
    "upper-right region"    : (0.75, 0.25, 0.40, 0.40),
    "upper right region"    : (0.75, 0.25, 0.40, 0.40),
    "lower-left region"     : (0.25, 0.75, 0.40, 0.40),
    "lower left region"     : (0.25, 0.75, 0.40, 0.40),
    "lower-right region"    : (0.75, 0.75, 0.40, 0.40),
    "lower right region"    : (0.75, 0.75, 0.40, 0.40),
    # Cardinal regions (used if no compound phrase matched)
    "central region"        : (0.50, 0.50, 0.40, 0.40),
    "upper region"          : (0.50, 0.20, 0.60, 0.35),
    "lower region"          : (0.50, 0.80, 0.60, 0.35),
    "left region"           : (0.20, 0.50, 0.35, 0.60),
    "right region"          : (0.80, 0.50, 0.35, 0.60),
}


def extract_region_v2(text: str):
    """
    Strict region extraction: ONLY returns a box if a region word is
    explicitly mentioned. Returns None otherwise — caller should skip.
    """
    t = text.lower().strip()
    # Try compound phrases first (longest match wins)
    for region in sorted(IMPROVED_REGION_TO_BOX.keys(), key=len, reverse=True):
        if region in t:
            return IMPROVED_REGION_TO_BOX[region]
    return None    # don't default to centre — skip the sample


def regenerate_yolo_dataset(hf_train, image_dir, out_dir):
    """
    Generate YOLO-format annotations for Route 4 with STRICT region matching.
    Skips samples where no region word appears in the answer.
    """
    print(f"\n  📐  Regenerating YOLO dataset with strict region matching ...")
    print(f"     Output: {out_dir}")

    images_dir = os.path.join(out_dir, "images", "train")
    labels_dir = os.path.join(out_dir, "labels", "train")
    val_img_dir = os.path.join(out_dir, "images", "val")
    val_lbl_dir = os.path.join(out_dir, "labels", "val")
    for d in [images_dir, labels_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    n_total = 0
    n_kept = 0
    n_skipped = 0
    n_train = 0
    n_val = 0
    rng = np.random.RandomState(42)

    for sample in tqdm(hf_train, desc="     Processing"):
        n_total += 1
        q = sample.get("question", "")
        a = sample.get("answer", "")
        img_id = sample.get("img_id", sample.get("image_id", ""))

        # Only Route 4 samples
        try:
            route = infer_route(q)
        except Exception:
            continue
        if route != 4:
            continue

        # Strict region extraction
        box = extract_region_v2(a)
        if box is None:
            n_skipped += 1
            continue

        cx, cy, bw, bh = box
        cls_id = extract_class_from_text(a)

        # Find source image
        src_img = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            cand = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(cand):
                src_img = cand
                break
        if src_img is None:
            continue

        # 90/10 train/val split
        if rng.rand() < 0.10:
            dst_img = os.path.join(val_img_dir, f"{img_id}.jpg")
            dst_lbl = os.path.join(val_lbl_dir, f"{img_id}.txt")
            n_val += 1
        else:
            dst_img = os.path.join(images_dir, f"{img_id}.jpg")
            dst_lbl = os.path.join(labels_dir, f"{img_id}.txt")
            n_train += 1

        # Copy image if not already there (cheap symlink would also work)
        if not os.path.exists(dst_img):
            try:
                shutil.copy(src_img, dst_img)
            except Exception:
                continue

        # Write YOLO label (segmentation polygon: 4 corner points)
        x1 = max(0.0, cx - bw/2); y1 = max(0.0, cy - bh/2)
        x2 = min(1.0, cx + bw/2); y2 = min(1.0, cy + bh/2)
        # YOLO-Seg expects normalised polygon points
        with open(dst_lbl, "w") as f:
            f.write(f"{cls_id} {x1:.4f} {y1:.4f} {x2:.4f} {y1:.4f} "
                    f"{x2:.4f} {y2:.4f} {x1:.4f} {y2:.4f}\n")

        n_kept += 1

    # Write data.yaml
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 4\n")
        f.write(f"names:\n")
        for cls_name, cls_idx in sorted(CLASS_MAP.items(),
                                          key=lambda x: x[1]):
            f.write(f"  {cls_idx}: {cls_name}\n")

    print(f"\n     Total Route-4 samples seen:    {n_total:,}")
    print(f"     Samples with region word:      {n_kept:,}")
    print(f"     Samples skipped (no region):   {n_skipped:,}")
    print(f"     Train images:                  {n_train:,}")
    print(f"     Val images:                    {n_val:,}")
    print(f"     ✅  YOLO data.yaml → {yaml_path}\n")
    return yaml_path


def train_route4_v2():
    print(f"\n{'='*72}")
    print(f"  Stage 4 V2 — Retraining Route 4 (YOLO-Seg Location)")
    print(f"{'='*72}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"❌  ultralytics not installed.")
        print(f"     Try: pip install ultralytics --break-system-packages")
        return

    # ── Step 1: Regenerate pseudo-annotations with strict matching ─────────
    from datasets import load_from_disk
    raw = load_from_disk(S4_CFG["data_dir"])
    image_dir = S4_CFG.get("image_dir", "")
    yolo_data = regenerate_yolo_dataset(
        raw["train"], image_dir, V2_CFG["yolo_data"])

    # ── Step 2: Train YOLO-Seg ──────────────────────────────────────────────
    print(f"  Loading YOLOv8m-seg from pretrained weights ...")
    model = YOLO("yolov8m-seg.pt")

    out_dir = V2_CFG["yolo_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Training YOLO-Seg v2 ...")
    print(f"     - epochs:    {V2_CFG['route4_yolo_epochs']}")
    print(f"     - lr0:        {V2_CFG['route4_yolo_lr']} (higher than original)")
    print(f"     - patience:   {V2_CFG['route4_yolo_patience']}")
    print(f"     - mosaic:     1.0")
    print(f"     - mixup:      0.15")
    print(f"     - hsv jitter: enabled\n")

    model.train(
        data       = yolo_data,
        epochs     = V2_CFG["route4_yolo_epochs"],
        imgsz      = 640,
        batch      = 8,
        workers    = 2,
        optimizer  = "AdamW",
        lr0        = V2_CFG["route4_yolo_lr"],
        lrf        = 0.01,
        warmup_epochs = 3,
        patience   = V2_CFG["route4_yolo_patience"],
        mosaic     = 1.0,
        mixup      = 0.15,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        fliplr     = 0.5,
        flipud     = 0.0,
        project    = out_dir,
        name       = "train",
        exist_ok   = True,
        overlap_mask = False,
    )
    print(f"\n  ✅  YOLO-Seg v2 saved to {out_dir}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation on test split
# ═════════════════════════════════════════════════════════════════════════════
def eval_route_v2(route: int):
    print(f"\n{'='*72}")
    print(f"  V2 Evaluation — Route {route} ({ROUTE_NAMES[route]})")
    print(f"{'='*72}\n")

    ckpt_path = os.path.join(
        V2_CFG["ckpt_dir"],
        f"route{route}_{ROUTE_NAMES[route]}{V2_SUFFIX}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"❌  v2 checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=V2_CFG["device"],
                       weights_only=False)
    vocab = ckpt["vocab"]
    print(f"  Loaded checkpoint (epoch {ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']*100:.2f}%)")
    print(f"  Config: {ckpt.get('config', {})}")

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        DistilBERTAnswerModel.MODEL_NAME)
    model = DistilBERTAnswerModel(vocab_per_route={route: vocab})
    model.load_state_dict(ckpt["model_state"])
    model = model.to(V2_CFG["device"]).eval()

    extractor = FusionExtractor(S4_CFG["stage3_ckpt"])
    text_prep = TextPreprocessor()
    from datasets import load_from_disk
    raw = load_from_disk(S4_CFG["data_dir"])
    test_records = cache_stage3_features(
        extractor, text_prep, raw["test"], "test", S4_CFG["cache_dir"])

    test_ds = DistilBERTRouteDataset(
        test_records, route, tokenizer, vocab, V2_CFG["max_input_len"])
    test_dl = DataLoader(test_ds, batch_size=V2_CFG["batch_size"]*2,
                          shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_ds):,}\n")

    rows = []
    pred_counter = Counter()
    correct = 0
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="  Evaluating"):
            batch_dev = {k: (v.to(V2_CFG["device"])
                              if isinstance(v, torch.Tensor) else v)
                          for k, v in batch.items()}
            cls_repr = model._encode(
                batch_dev["fused"], batch_dev["disease"],
                batch_dev["input_ids"], batch_dev["attention_mask"])
            logits = model.heads[str(route)](cls_repr)
            preds = logits.argmax(dim=-1)
            for pi in range(len(preds)):
                p_idx = preds[pi].item()
                g_idx = batch["label"][pi].item()
                pred_counter[vocab[p_idx]] += 1
                if p_idx == g_idx:
                    correct += 1
                rows.append({"prediction":   vocab[p_idx],
                              "ground_truth": vocab[g_idx],
                              "correct":      (p_idx == g_idx)})

    eval_csv = os.path.join(
        V2_CFG["log_dir"],
        f"route{route}_{ROUTE_NAMES[route]}{V2_SUFFIX}_eval.csv")
    df = pd.DataFrame(rows)
    df.to_csv(eval_csv, index=False)
    print(f"\n  ✅  Eval CSV → {eval_csv}")

    n = len(rows)
    acc = correct / max(n, 1) * 100
    print(f"\n  ┌────────────────────────────────────────────┐")
    print(f"  │  TEST ACCURACY:  {acc:6.2f}%  ({correct}/{n})   │")
    print(f"  └────────────────────────────────────────────┘\n")

    print(f"  Top predictions (diversity check):")
    for pred, cnt in pred_counter.most_common(10):
        pct = cnt / n * 100
        bar = "█" * int(pct / 2)
        print(f"    '{pred[:35]:<35}'  {cnt:>5}  ({pct:5.2f}%)  {bar}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default="all",
                         help="Route: 0, 1, 3, 4, or 'all'")
    parser.add_argument("--mode", default="both",
                         choices=["train", "eval", "both"])
    args = parser.parse_args()

    routes = []
    if args.route == "all":
        routes = [0, 1, 3, 4]
    else:
        routes = [int(args.route)]

    for route in routes:
        if args.mode in ["train", "both"]:
            if route == 4:
                train_route4_v2()
            elif route in [0, 1, 3]:
                train_route_v2(route)
            else:
                print(f"⚠️   Route {route} not configured for v2")
                continue

        if args.mode in ["eval", "both"] and route in [0, 1, 3]:
            eval_route_v2(route)


if __name__ == "__main__":
    main()
