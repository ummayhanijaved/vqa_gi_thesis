#!/usr/bin/env python3
"""
=============================================================================
Stage 4 — Combined Single + Multi Choice Multi-Task Training
=============================================================================

Supervisor recommendation: train Route 1 (Single Choice) and Route 2
(Multi Choice) JOINTLY on a SHARED DistilBERT backbone with two
task-specific heads.

This is multi-task learning. The shared backbone learns richer text
representations from BOTH datasets combined; at inference time the
two heads produce separate predictions for Single and Multi Choice.

Why this should help:
  - Route 1 has ~2,643 test samples but Route 2 only has ~309
  - Combined training exposes the model to 14,594 + 1,556 train samples
  - Shared backbone learns better medical text representations
  - Each head specialises in its own output structure (softmax vs sigmoid)

Architecture:
  Question ──→ DistilBERT (shared, frozen+2 unfrozen)
                    │
                    ▼
              [CLS] token (768-D, with Stage 3 soft-prompt)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    Single Head            Multi Head
    Linear(768 → 50)       Linear(768 → 200)
    Softmax + CE           Sigmoid + BCE
        │                       │
        ▼                       ▼
    Single pred             Multi preds

Loss = α * L_single + β * L_multi   (α=β=1 by default)

Usage:
    python stage4_combined_single_multi.py --mode train
    python stage4_combined_single_multi.py --mode eval
    python stage4_combined_single_multi.py --mode both       # train + eval

Outputs:
    ~/vqa_gi_thesis/checkpoints/stage4_revised/stage4_combined_sm_best.pt
    ~/vqa_gi_thesis/logs/stage4_revised/route1_combined_eval.csv
    ~/vqa_gi_thesis/logs/stage4_revised/route2_combined_eval.csv
    ~/vqa_gi_thesis/logs/stage4_revised/combined_sm_log.csv
=============================================================================
"""
import os
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from transformers import (
    DistilBertModel, DistilBertTokenizerFast,
    get_cosine_schedule_with_warmup,
)

# Reuse machinery from the main pipeline
SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

from stage4_revised import (
    CFG, ROUTE_NAMES, build_vocab, normalise_answer,
    FusionExtractor, cache_stage3_features,
)
from preprocessing import build_image_transform, TextPreprocessor

# Override checkpoint location for the combined model
CKPT_NAME = "stage4_combined_sm_best.pt"
LOG_NAME  = "combined_sm_log.csv"

CKPT_PATH = os.path.join(CFG["ckpt_dir"], CKPT_NAME)
LOG_PATH  = os.path.join(CFG["log_dir"],  LOG_NAME)
os.makedirs(CFG["ckpt_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"],  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Dataset — yields both Single and Multi samples in the same batch
# ─────────────────────────────────────────────────────────────────────────────
class CombinedSingleMultiDataset(Dataset):
    """
    Yields samples from BOTH Route 1 (Single) and Route 2 (Multi).
    Each item carries its route id so the model knows which head to use.
    """
    def __init__(self, records, vocab_single, vocab_multi, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.vocab_s   = vocab_single
        self.vocab_m   = vocab_multi
        self.s2i       = {v: i for i, v in enumerate(vocab_single)}
        self.m2i       = {v: i for i, v in enumerate(vocab_multi)}

        # Only keep records for routes 1 and 2
        self.samples = [r for r in records if r["route"] in [1, 2]]
        n1 = sum(1 for r in self.samples if r["route"] == 1)
        n2 = sum(1 for r in self.samples if r["route"] == 2)
        print(f"   Combined Dataset: {len(self.samples):,} samples  "
              f"(Single={n1:,}  Multi={n2:,})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        # Tokenise question
        enc = self.tokenizer(
            r["question"], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt")

        # Stage 3 features
        fused   = r.get("fused", r.get("fused_repr"))
        disease = r.get("disease", r.get("disease_vec"))

        item = {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "fused"          : fused,
            "disease"        : disease,
            "route"          : r["route"],
            "answer_raw"     : r["answer"],
        }

        # Build BOTH labels — only the active one is used per sample
        # Single label (route 1)
        ans_s = normalise_answer(r["answer"], 1)
        item["label_single"] = torch.tensor(
            self.s2i.get(ans_s, 0), dtype=torch.long)

        # Multi label (route 2)
        label_m = torch.zeros(len(self.vocab_m))
        for tok in r["answer"].split(","):
            tok = normalise_answer(tok.strip(), 2)
            if tok in self.m2i:
                label_m[self.m2i[tok]] = 1.0
        item["label_multi"] = label_m

        return item


# ─────────────────────────────────────────────────────────────────────────────
# Stage3Projector — same as in stage4_revised
# ─────────────────────────────────────────────────────────────────────────────
class Stage3Projector(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CFG["head_input_dim"], hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, fused, disease):
        x = torch.cat([fused, disease], dim=-1)
        return self.proj(x).unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Model: Shared DistilBERT + 2 heads
# ─────────────────────────────────────────────────────────────────────────────
class CombinedSingleMultiModel(nn.Module):
    """
    Shared DistilBERT backbone with two task-specific heads.

    - Single head: Linear(768 → 50)  + CrossEntropyLoss (with class weights)
    - Multi  head: Linear(768 → 200) + BCEWithLogitsLoss
    """
    HIDDEN     = 768
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, vocab_single, vocab_multi):
        super().__init__()
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self.distilbert = DistilBertModel.from_pretrained(self.MODEL_NAME)
        self.projector  = Stage3Projector(self.HIDDEN)

        # Freeze all DistilBERT layers — unfreeze last 2
        for p in self.distilbert.parameters():
            p.requires_grad = False
        for layer in self.distilbert.transformer.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Two task-specific heads
        self.head_single = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.HIDDEN // 2, len(vocab_single)),
        )
        self.head_multi = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.HIDDEN // 2, len(vocab_multi)),
        )

        self.vocab_single = vocab_single
        self.vocab_multi  = vocab_multi
        self.ce_loss      = None        # set after class weights computed
        self.bce_loss     = nn.BCEWithLogitsLoss()

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_bert = sum(p.numel() for p in self.distilbert.parameters()
                     if p.requires_grad)
        n_hs   = sum(p.numel() for p in self.head_single.parameters())
        n_hm   = sum(p.numel() for p in self.head_multi.parameters())
        print(f"\n🧠  CombinedSingleMultiModel")
        print(f"    Projector       : {n_proj:>10,} params")
        print(f"    Trainable BERT  : {n_bert:>10,} params  (last 2 layers)")
        print(f"    Single head     : {n_hs:>10,} params  → {len(vocab_single)} classes")
        print(f"    Multi  head     : {n_hm:>10,} params  → {len(vocab_multi)} classes")
        print(f"    TOTAL TRAINABLE : {n_proj + n_bert + n_hs + n_hm:>10,} params\n")

    def set_class_weights_single(self, weights: torch.Tensor):
        """Set inverse-frequency class weights for the Single-Choice CE loss."""
        self.register_buffer("_w_single", weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    def _encode(self, fused, disease, input_ids, attention_mask):
        """Run DistilBERT with Stage 3 soft-prompt prefix.  Same as stage4_revised."""
        emb       = self.distilbert.embeddings
        word_emb  = emb.word_embeddings(input_ids)
        prefix    = self.projector(fused, disease).to(word_emb.dtype)
        combined  = torch.cat([prefix, word_emb], dim=1)

        seq_len   = combined.size(1)
        pos_ids   = torch.arange(seq_len, dtype=torch.long,
                                  device=combined.device).unsqueeze(0)
        combined  = combined + emb.position_embeddings(pos_ids)
        combined  = emb.LayerNorm(combined)
        combined  = emb.dropout(combined)

        prefix_mask = torch.ones(fused.size(0), 1,
                                  dtype=attention_mask.dtype,
                                  device=attention_mask.device)
        ext_mask    = torch.cat([prefix_mask, attention_mask], dim=1)

        # Run transformer — newer transformers versions renamed the first
        # positional argument to "hidden_states"; older versions accept
        # positional or "x". Try in order of compatibility.
        try:
            out = self.distilbert.transformer(
                hidden_states=combined, attn_mask=ext_mask)
        except TypeError:
            try:
                out = self.distilbert.transformer(combined, attn_mask=ext_mask)
            except TypeError:
                out = self.distilbert.transformer(x=combined, attn_mask=ext_mask)

        # Output may be tuple, ModelOutput dataclass, or tensor
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        return hidden[:, 1, :]          # position-1 = original [CLS]

    def forward(self, batch):
        cls = self._encode(
            batch["fused"], batch["disease"],
            batch["input_ids"], batch["attention_mask"])
        return self.head_single(cls), self.head_multi(cls)

    def compute_loss(self, logits_s, logits_m, batch):
        """
        Mixed loss: for each sample apply ONLY its route's loss,
        weighted equally.

        For samples in Route 1 (single) — use CE on logits_s
        For samples in Route 2 (multi)  — use BCE on logits_m
        """
        routes = batch["route"]                          # (B,)
        mask_s = (routes == 1)
        mask_m = (routes == 2)

        loss_s = torch.tensor(0.0, device=logits_s.device)
        loss_m = torch.tensor(0.0, device=logits_s.device)

        if mask_s.any():
            loss_s = self.ce_loss(
                logits_s[mask_s], batch["label_single"][mask_s])
        if mask_m.any():
            loss_m = self.bce_loss(
                logits_m[mask_m], batch["label_multi"][mask_m])

        # Equal weighting; sum (not mean) so each task contributes
        return loss_s + loss_m, loss_s.item(), loss_m.item()


# ─────────────────────────────────────────────────────────────────────────────
# Compute class weights for Single Choice (inverse frequency)
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(records, vocab):
    counts = Counter()
    for r in records:
        if r["route"] == 1:
            ans = normalise_answer(r["answer"], 1)
            counts[ans] += 1
    n_total = sum(counts.values())
    n_classes = len(vocab)
    weights = []
    for cls in vocab:
        nc = max(counts.get(cls, 1), 1)
        w = n_total / (n_classes * nc)
        w = max(0.5, min(10.0, w))     # clip
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'='*72}")
    print(f"  Combined Single + Multi Choice Multi-Task Training")
    print(f"{'='*72}\n")

    print(f"  Stage 3 cache: {CFG['cache_dir']}")
    extractor = FusionExtractor(CFG["stage3_ckpt"])
    text_prep = TextPreprocessor()

    # Load HF dataset and build cache (uses existing cache if present)
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])

    train_records = cache_stage3_features(
        extractor, text_prep, raw["train"], "train", CFG["cache_dir"])
    val_records = cache_stage3_features(
        extractor, text_prep, raw.get("validation", raw["train"]),
        "val", CFG["cache_dir"])

    # Build vocabularies (top-50 single + top-200 multi, same as before)
    vocab_single = build_vocab(train_records, route=1, max_classes=50)
    vocab_multi  = build_vocab(train_records, route=2, max_classes=200)
    print(f"\n  Vocab sizes:  Single={len(vocab_single):>3}  "
          f"Multi={len(vocab_multi):>3}")

    # Class weights for Route 1
    weights = compute_class_weights(train_records, vocab_single)
    weights = weights.to(CFG["device"])
    print(f"  Single-Choice class weights: min={weights.min().item():.2f}  "
          f"max={weights.max().item():.2f}  mean={weights.mean().item():.2f}")

    # Model
    model = CombinedSingleMultiModel(vocab_single, vocab_multi)
    model.set_class_weights_single(weights)
    model = model.to(CFG["device"])

    # Datasets / loaders
    tokenizer = model.tokenizer
    train_ds = CombinedSingleMultiDataset(
        train_records, vocab_single, vocab_multi,
        tokenizer, CFG["max_input_len"])
    val_ds   = CombinedSingleMultiDataset(
        val_records, vocab_single, vocab_multi,
        tokenizer, CFG["max_input_len"])

    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                           shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=0)

    # Optimiser with per-group LR
    bert_params = [p for p in model.distilbert.parameters() if p.requires_grad]
    head_params = (list(model.projector.parameters()) +
                    list(model.head_single.parameters()) +
                    list(model.head_multi.parameters()))
    optimiser = torch.optim.AdamW([
        {"params": bert_params,  "lr": CFG["distilbert_lr"]},
        {"params": head_params,  "lr": CFG["head_lr"]},
    ], weight_decay=CFG["weight_decay"])

    n_steps = len(train_dl) * CFG["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(n_steps * CFG["warmup_ratio"]),
        num_training_steps=n_steps)

    # Training loop
    best_val_score = 0.0
    patience       = 0
    history        = []
    print(f"\n  Training for up to {CFG['epochs']} epochs ...\n")

    for epoch in range(1, CFG["epochs"] + 1):
        # ─── TRAIN ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_loss_s = 0.0
        total_loss_m = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:2d} train")
        for batch in pbar:
            batch = {k: (v.to(CFG["device"]) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            logits_s, logits_m = model(batch)
            loss, ls, lm = model.compute_loss(logits_s, logits_m, batch)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimiser.step()
            scheduler.step()

            total_loss   += loss.item()
            total_loss_s += ls
            total_loss_m += lm
            n_batches    += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                              ls=f"{ls:.3f}", lm=f"{lm:.3f}")

        avg_loss   = total_loss   / max(n_batches, 1)
        avg_loss_s = total_loss_s / max(n_batches, 1)
        avg_loss_m = total_loss_m / max(n_batches, 1)

        # ─── VAL ───────────────────────────────────────────────────────────
        model.eval()
        s_correct = s_total = 0
        m_tp = m_fp = m_fn = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch:2d} val  "):
                batch = {k: (v.to(CFG["device"]) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                logits_s, logits_m = model(batch)
                routes = batch["route"]

                # Single accuracy
                mask_s = (routes == 1)
                if mask_s.any():
                    preds = logits_s[mask_s].argmax(dim=-1)
                    s_correct += (preds == batch["label_single"][mask_s]).sum().item()
                    s_total   += mask_s.sum().item()

                # Multi F1 (micro)
                mask_m = (routes == 2)
                if mask_m.any():
                    probs = torch.sigmoid(logits_m[mask_m])
                    preds = (probs >= CFG["threshold"]).float()
                    gt    = batch["label_multi"][mask_m]
                    m_tp += ((preds == 1) & (gt == 1)).sum().item()
                    m_fp += ((preds == 1) & (gt == 0)).sum().item()
                    m_fn += ((preds == 0) & (gt == 1)).sum().item()

        s_acc = (s_correct / max(s_total, 1)) * 100
        precision = m_tp / max(m_tp + m_fp, 1)
        recall    = m_tp / max(m_tp + m_fn, 1)
        m_f1      = 2 * precision * recall / max(precision + recall, 1e-9) * 100

        # Validation score = balanced combination
        val_score = 0.5 * s_acc + 0.5 * m_f1

        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "train_loss_single": avg_loss_s, "train_loss_multi": avg_loss_m,
            "val_single_acc": s_acc, "val_multi_f1": m_f1,
            "val_combined": val_score,
        })
        print(f"\n  Epoch {epoch:2d}  |  "
              f"loss={avg_loss:.4f} (s={avg_loss_s:.3f} m={avg_loss_m:.3f})  |  "
              f"val_single={s_acc:5.2f}%  val_multi_F1={m_f1:5.2f}%  "
              f"combined={val_score:5.2f}\n")

        # Save best
        if val_score > best_val_score:
            best_val_score = val_score
            patience = 0
            torch.save({
                "model_state":  model.state_dict(),
                "vocab_single": vocab_single,
                "vocab_multi":  vocab_multi,
                "weights_single": weights.cpu(),
                "epoch":        epoch,
                "val_single":   s_acc,
                "val_multi":    m_f1,
                "val_combined": val_score,
            }, CKPT_PATH)
            print(f"   ✅  Saved best checkpoint  →  {CKPT_PATH}")
        else:
            patience += 1
            print(f"   ⏳  No improvement ({patience}/{CFG['early_stop_pat']})")
            if patience >= CFG["early_stop_pat"]:
                print(f"\n   🛑  Early stopping at epoch {epoch}")
                break

    # Save history
    pd.DataFrame(history).to_csv(LOG_PATH, index=False)
    print(f"\n   ✅  Training log saved → {LOG_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on test split — produces two separate eval CSVs
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    print(f"\n{'='*72}")
    print(f"  Combined Model Evaluation on Test Split")
    print(f"{'='*72}\n")

    if not os.path.exists(CKPT_PATH):
        print(f"❌  No checkpoint found at {CKPT_PATH}")
        return

    ckpt = torch.load(CKPT_PATH, map_location=CFG["device"], weights_only=False)
    vocab_single = ckpt["vocab_single"]
    vocab_multi  = ckpt["vocab_multi"]
    print(f"  Loaded checkpoint (best epoch {ckpt['epoch']}, "
          f"val_single={ckpt['val_single']:.2f}%, "
          f"val_multi_F1={ckpt['val_multi']:.2f}%)")

    model = CombinedSingleMultiModel(vocab_single, vocab_multi)
    model.set_class_weights_single(ckpt["weights_single"].to(CFG["device"]))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(CFG["device"]).eval()

    extractor = FusionExtractor(CFG["stage3_ckpt"])
    text_prep = TextPreprocessor()
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])
    test_records = cache_stage3_features(
        extractor, text_prep, raw["test"], "test", CFG["cache_dir"])

    tokenizer = model.tokenizer
    test_ds = CombinedSingleMultiDataset(
        test_records, vocab_single, vocab_multi,
        tokenizer, CFG["max_input_len"])
    test_dl = DataLoader(test_ds, batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0)

    rows_single = []
    rows_multi  = []
    s_correct = s_total = 0
    m_tp = m_fp = m_fn = 0

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating combined"):
            batch_dev = {k: (v.to(CFG["device"]) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch.items()}
            logits_s, logits_m = model(batch_dev)
            routes = batch_dev["route"]

            # Single
            mask_s = (routes == 1).nonzero(as_tuple=True)[0].tolist()
            for idx in mask_s:
                pred_idx = logits_s[idx].argmax().item()
                pred = vocab_single[pred_idx]
                gt   = batch["answer_raw"][idx]
                correct = (normalise_answer(gt, 1) == pred) or (pred in gt.lower())
                if correct:
                    s_correct += 1
                s_total += 1
                rows_single.append({
                    "prediction": pred,
                    "ground_truth": gt,
                    "correct": correct,
                })

            # Multi
            mask_m = (routes == 2).nonzero(as_tuple=True)[0].tolist()
            for idx in mask_m:
                probs = torch.sigmoid(logits_m[idx]).cpu().numpy()
                preds_set = [vocab_multi[i] for i, p in enumerate(probs)
                              if p >= CFG["threshold"]]
                gt_set = []
                for tok in batch["answer_raw"][idx].split(","):
                    nrm = normalise_answer(tok.strip(), 2)
                    if nrm: gt_set.append(nrm)
                tp = len(set(preds_set) & set(gt_set))
                fp = len(set(preds_set) - set(gt_set))
                fn = len(set(gt_set)    - set(preds_set))
                m_tp += tp; m_fp += fp; m_fn += fn
                rows_multi.append({
                    "prediction":   ", ".join(preds_set) if preds_set else "(none)",
                    "ground_truth": batch["answer_raw"][idx],
                    "n_pred": len(preds_set), "n_gt": len(gt_set),
                    "tp": tp, "fp": fp, "fn": fn,
                })

    s_acc     = (s_correct / max(s_total, 1)) * 100
    precision = m_tp / max(m_tp + m_fp, 1)
    recall    = m_tp / max(m_tp + m_fn, 1)
    m_f1      = 2 * precision * recall / max(precision + recall, 1e-9) * 100

    print(f"\n{'─'*60}")
    print(f"  COMBINED MODEL FINAL TEST RESULTS")
    print(f"{'─'*60}")
    print(f"  Route 1 (Single Choice) accuracy : {s_acc:.2f}%  "
          f"({s_correct}/{s_total})")
    print(f"  Route 2 (Multi  Choice) F1       : {m_f1:.2f}%  "
          f"(P={precision*100:.2f}, R={recall*100:.2f})")
    print(f"{'─'*60}\n")

    # Save eval CSVs
    out_dir = CFG["log_dir"]
    s_path = os.path.join(out_dir, "route1_combined_eval.csv")
    m_path = os.path.join(out_dir, "route2_combined_eval.csv")
    pd.DataFrame(rows_single).to_csv(s_path, index=False)
    pd.DataFrame(rows_multi).to_csv(m_path,  index=False)
    print(f"  ✅  Single eval CSV → {s_path}")
    print(f"  ✅  Multi  eval CSV → {m_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both",
                        choices=["train", "eval", "both"])
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        train()
    if args.mode in ["eval", "both"]:
        evaluate()


if __name__ == "__main__":
    main()
