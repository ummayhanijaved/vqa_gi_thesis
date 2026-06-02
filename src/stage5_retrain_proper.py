#!/usr/bin/env python3
"""
=============================================================================
PHASE 2 — Proper Stage 5 T5 Retraining (using Stage 4 cache)
=============================================================================

PURPOSE:
    Retrain T5-small verbalizer on FULL 143,594 training samples using the
    Stage 4 cache built in Phase 1. This is MUCH more data than the original
    Stage 5 V1 (which used only ~8K test eval CSVs).

INPUT:
    ~/vqa_gi_thesis/cache/stage4_predictions/stage4_cache_train.pt
    ~/vqa_gi_thesis/cache/stage4_predictions/stage4_cache_val.pt

OUTPUT:
    ~/vqa_gi_thesis/checkpoints/stage5_verbalizer/
        ├── stage5_verbalizer_best.pt       (V1 — UNTOUCHED)
        └── stage5_verbalizer_v2_best.pt    (V2 — NEW, this script)

SAFETY:
    - Original Stage 5 V1 checkpoint is NEVER touched
    - V2 saved to a NEW filename
    - Training pairs are filtered for quality

USAGE:
    python stage5_retrain_proper.py
=============================================================================
"""
import os
import sys
import argparse
import warnings
import random
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

try:
    from transformers import (
        T5Tokenizer, T5ForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )
except ImportError as e:
    print(f"❌  Failed to import T5: {e}")
    print(f"    Try: pip install sentencepiece protobuf --break-system-packages")
    sys.exit(1)

from stage4_revised import ROUTE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/vqa_gi_thesis")
CFG = {
    "model_name":      "t5-small",
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "max_input_len":   128,
    "max_output_len":  96,
    "batch_size":      16,
    "epochs":          10,
    "lr":              2e-4,
    "weight_decay":    0.01,
    "warmup_ratio":    0.1,
    "grad_clip":       1.0,
    "early_stop_pat":  3,
    "stage4_cache_dir": os.path.join(PROJECT, "cache", "stage4_predictions"),
    "ckpt_dir":         os.path.join(PROJECT, "checkpoints", "stage5_verbalizer"),
    "log_dir":          os.path.join(PROJECT, "logs", "stage5_verbalizer"),
}
os.makedirs(CFG["ckpt_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"], exist_ok=True)

V2_CKPT_PATH = os.path.join(CFG["ckpt_dir"], "stage5_verbalizer_v2_best.pt")
V2_LOG_PATH  = os.path.join(CFG["log_dir"],  "stage5_v2_log.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Build training pairs from Stage 4 cache
# ─────────────────────────────────────────────────────────────────────────────
def load_and_filter(cache_path, split_name):
    print(f"\n  Loading {split_name} from {cache_path}")
    if not os.path.exists(cache_path):
        print(f"  ❌  Cache not found!")
        return []

    records = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"  ✅  Loaded {len(records):,} records")

    pairs = []
    skipped = Counter()

    for r in records:
        pred = str(r.get("stage4_pred", "")).strip()
        gt   = str(r.get("gt_answer", "")).strip()
        question = str(r.get("question", "")).strip()
        route = r.get("route", -1)
        route_name = r.get("route_name", "unknown")

        # Filter bad entries
        if not pred or "error" in pred.lower() or pred.lower() in ["nan", "no image"]:
            skipped["bad_pred"] += 1; continue
        if not gt or len(gt) < 5 or len(gt) > 250:
            skipped["bad_gt"] += 1; continue
        if not question:
            skipped["no_q"] += 1; continue

        input_text = (
            f"verbalize | route: {route_name} "
            f"| question: {question[:100]} "
            f"| answer: {pred[:120]}"
        )
        pairs.append({
            "route":      route,
            "input":      input_text,
            "target":     gt,
            "question":   question,
            "stage4_pred": pred,
        })

    print(f"  ✅  Kept: {len(pairs):,}  Skipped: {sum(skipped.values()):,}")
    for k, v in skipped.items():
        print(f"     {k}: {v:,}")

    # Per-route count
    rc = Counter(p["route"] for p in pairs)
    print(f"  Pairs per route:")
    for r in range(6):
        print(f"     Route {r} ({ROUTE_NAMES[r]:<15}): {rc.get(r, 0):,}")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class VerbalizerDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        inp = self.tokenizer(str(p["input"]),
                              max_length=CFG["max_input_len"],
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        tgt = self.tokenizer(str(p["target"]),
                              max_length=CFG["max_output_len"],
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        labels = tgt["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      inp["input_ids"].squeeze(0),
            "attention_mask": inp["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'█'*72}")
    print(f"  PHASE 2 — Stage 5 V2 Retraining (Full 143K Dataset)")
    print(f"{'█'*72}\n")

    # ── Load Stage 4 cache ────────────────────────────────────────────────
    train_path = os.path.join(CFG["stage4_cache_dir"], "stage4_cache_train.pt")
    val_path   = os.path.join(CFG["stage4_cache_dir"], "stage4_cache_val.pt")

    print(f"  Building training pairs from Stage 4 cache ...")
    train_pairs = load_and_filter(train_path, "TRAIN")
    val_pairs   = load_and_filter(val_path,   "VAL")

    if not train_pairs:
        print(f"\n❌  No training pairs. Run Phase 1 (stage4_build_cache.py) first.")
        return

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  FINAL DATASET:                              │")
    print(f"  │   Train pairs: {len(train_pairs):>7,}                    │")
    print(f"  │   Val pairs:   {len(val_pairs):>7,}                    │")
    print(f"  └─────────────────────────────────────────────┘\n")

    # ── Load T5 fresh ─────────────────────────────────────────────────────
    print(f"  Loading {CFG['model_name']} from scratch (NOT continuing V1) ...")
    tokenizer = T5Tokenizer.from_pretrained(CFG["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(CFG["model_name"])
    model = model.to(CFG["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅  Loaded — {n_params:,} parameters\n")

    train_ds = VerbalizerDataset(train_pairs, tokenizer)
    val_ds   = VerbalizerDataset(val_pairs, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                           shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n_steps = len(train_dl) * CFG["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * CFG["warmup_ratio"]),
        num_training_steps=n_steps)

    best_val = float("inf"); patience = 0; history = []
    print(f"  Training for up to {CFG['epochs']} epochs ...")
    print(f"  ETA: ~{(len(train_dl) * CFG['epochs']) * 1.5 / 3600:.1f} hours\n")

    for epoch in range(1, CFG["epochs"] + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        tot_loss = 0.0; n_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:2d} train")
        for batch in pbar:
            batch = {k: v.to(CFG["device"]) for k, v in batch.items()}
            out = model(**batch); loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            CFG["grad_clip"])
            optimizer.step(); scheduler.step()
            tot_loss += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = tot_loss / max(n_batches, 1)

        # ── val ───────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0; v_n = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch:2d} val  "):
                batch = {k: v.to(CFG["device"]) for k, v in batch.items()}
                v_loss += model(**batch).loss.item(); v_n += 1
        val_loss = v_loss / max(v_n, 1)

        # Sample generations
        if len(val_pairs) >= 3:
            print(f"\n  Sample generations (epoch {epoch}):")
            for idx in random.sample(range(len(val_pairs)), 3):
                p = val_pairs[idx]
                inp = tokenizer(str(p["input"]),
                                max_length=CFG["max_input_len"],
                                truncation=True, return_tensors="pt"
                                ).to(CFG["device"])
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inp, max_length=CFG["max_output_len"],
                        num_beams=2, early_stopping=True)
                gen = tokenizer.decode(gen_ids[0],
                                        skip_special_tokens=True)
                print(f"    [R{p['route']}] S4: '{p['stage4_pred'][:50]}'")
                print(f"           Gen: '{gen[:100]}'")
                print(f"           GT : '{p['target'][:100]}'\n")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss
        })
        print(f"  Epoch {epoch:2d}  |  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}\n")

        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save({
                "model_state":    model.state_dict(),
                "tokenizer_name": CFG["model_name"],
                "epoch":          epoch,
                "val_loss":       val_loss,
                "train_samples":  len(train_pairs),
                "val_samples":    len(val_pairs),
            }, V2_CKPT_PATH)
            print(f"   ✅  Saved V2 ckpt → {V2_CKPT_PATH}")
        else:
            patience += 1
            print(f"   ⏳  No improvement ({patience}/{CFG['early_stop_pat']})")
            if patience >= CFG["early_stop_pat"]:
                print(f"\n   🛑  Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(V2_LOG_PATH, index=False)
    print(f"\n  ✅  Training log → {V2_LOG_PATH}")
    print(f"  ✅  Best val_loss: {best_val:.4f}")
    print(f"\n{'█'*72}\n  PHASE 2 COMPLETE\n{'█'*72}\n")


if __name__ == "__main__":
    train()
