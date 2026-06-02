#!/usr/bin/env python3
"""
=============================================================================
Stage 5 — T5-small Answer Verbalizer (Production Version)
=============================================================================

PURPOSE:
    Convert Stage 4's structured outputs (e.g. "yes", "upper-central",
    "polyp") into natural-language full sentences matching the
    Kvasir-VQA-x1 ground-truth answer style.

KEY DESIGN CHOICES (verified against your actual codebase):
    1. Eval CSVs only contain (prediction, ground_truth) — NO question column.
       So we fetch the question from the original Kvasir-VQA-x1 dataset
       using infer_route() to align rows.
    2. T5-small fine-tuning from Hugging Face — 60M params.
    3. Train/val split is stratified by route AND by image_id to prevent
       data leakage (same image's QA pairs stay in the same split).
    4. Pairs are filtered: target length 5-250 chars, predictions non-empty.
    5. Beam search (num_beams=4) for inference quality, greedy for speed.

USAGE:
    pip install sentencepiece protobuf --break-system-packages

    python stage5_verbalizer.py --mode build_data   # build training pairs
    python stage5_verbalizer.py --mode train         # fine-tune T5-small
    python stage5_verbalizer.py --mode test          # inference test
    python stage5_verbalizer.py --mode all           # build + train + test
=============================================================================
"""
import os
import sys
import json
import argparse
import warnings
import random

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Try import T5 — fail early with a helpful message if sentencepiece missing
try:
    from transformers import (
        T5Tokenizer, T5ForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )
    _ = T5Tokenizer
except Exception as e:
    print(f"❌  Failed to import T5: {e}")
    print(f"    Try: pip install sentencepiece protobuf --break-system-packages")
    sys.exit(1)

# Bring in CFG and infer_route from your main pipeline
SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

from stage4_revised import CFG as S4_CFG, infer_route

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "model_name"     : "t5-small",
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
    "max_input_len"  : 128,
    "max_output_len" : 96,
    "batch_size"     : 16,
    "epochs"         : 8,
    "lr"             : 3e-4,
    "weight_decay"   : 0.01,
    "warmup_ratio"   : 0.1,
    "grad_clip"      : 1.0,
    "early_stop_pat" : 3,
    "data_dir"       : S4_CFG["data_dir"],
    "eval_dir"       : S4_CFG["log_dir"],
    "ckpt_dir"       : os.path.expanduser(
        "~/vqa_gi_thesis/checkpoints/stage5_verbalizer"),
    "log_dir"        : os.path.expanduser(
        "~/vqa_gi_thesis/logs/stage5_verbalizer"),
}
os.makedirs(CFG["ckpt_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"], exist_ok=True)

CKPT_PATH      = os.path.join(CFG["ckpt_dir"], "stage5_verbalizer_best.pt")
TRAIN_PAIRS    = os.path.join(CFG["log_dir"], "training_pairs.csv")
TRAIN_LOG_PATH = os.path.join(CFG["log_dir"], "stage5_log.csv")

ROUTE_NAMES = {
    0: "yes_no", 1: "single_choice", 2: "multi_choice",
    3: "color",  4: "location",      5: "count",
}

# Eval CSV filenames (verified from stage4_revised.py)
EVAL_CSV_FILES = {
    0: "route0_yes_no_eval.csv",
    1: "route1_single_choice_eval.csv",
    2: "route2_multi_choice_eval.csv",
    3: "route3_color_eval.csv",
    4: "route4_location_yolo_eval.csv",
    5: "route5_count_yolo_eval.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Build training pairs
# ─────────────────────────────────────────────────────────────────────────────
def build_training_pairs():
    print(f"\n{'='*72}")
    print(f"  Stage 5 — Building Training Pairs")
    print(f"{'='*72}\n")

    from datasets import load_from_disk
    print(f"  Loading Kvasir-VQA-x1 ...")
    raw = load_from_disk(CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]
    print(f"  ✅  Test split: {len(test_split):,} samples\n")

    print(f"  Routing HF test samples by question category ...")
    samples_by_route = {r: [] for r in range(6)}
    for s in tqdm(test_split, desc="  Routing"):
        q = s.get("question", "")
        if not q: continue
        try:
            r = infer_route(q)
        except Exception:
            continue
        if r in samples_by_route:
            samples_by_route[r].append({
                "question": q,
                "answer_full": s.get("answer", ""),
                "img_id": s.get("img_id", s.get("image_id", "")),
            })
    print(f"\n  Samples per route in HF test split:")
    for r in range(6):
        print(f"    Route {r} ({ROUTE_NAMES[r]:<15}): "
              f"{len(samples_by_route[r]):,}")

    print(f"\n  Building training pairs ...")
    all_pairs = []
    for route, fname in EVAL_CSV_FILES.items():
        path = os.path.join(CFG["eval_dir"], fname)
        if not os.path.exists(path):
            print(f"  ⚠️   Missing {fname} — skipping route {route}")
            continue

        df = pd.read_csv(path)
        hf_samples = samples_by_route.get(route, [])
        n_csv, n_hf = len(df), len(hf_samples)
        if n_csv != n_hf:
            print(f"  ⚠️   Route {route}: CSV {n_csv} rows, HF {n_hf} samples — using min")
        n = min(n_csv, n_hf)
        if n == 0:
            print(f"  ⚠️   Route {route}: no usable rows")
            continue

        kept = 0
        for i in range(n):
            pred_raw = str(df.iloc[i]["prediction"]).strip()
            gt_csv   = str(df.iloc[i]["ground_truth"]).strip()
            hf       = hf_samples[i]
            question = hf["question"].strip()
            target_text = hf["answer_full"].strip() or gt_csv

            if not pred_raw or pred_raw.lower() in ["nan", "(none)", ""]:
                continue
            if not target_text or len(target_text) < 5 or len(target_text) > 250:
                continue

            input_text = (
                f"verbalize | route: {ROUTE_NAMES[route]} "
                f"| question: {question[:100]} "
                f"| answer: {pred_raw[:80]}"
            )
            all_pairs.append({
                "route": route, "img_id": hf["img_id"],
                "input": input_text, "target": target_text,
                "question": question, "pred": pred_raw,
            })
            kept += 1
        print(f"  ✅  Route {route} ({ROUTE_NAMES[route]:<15}): kept {kept:,}/{n:,}")

    df_out = pd.DataFrame(all_pairs)
    df_out.to_csv(TRAIN_PAIRS, index=False)
    print(f"\n  ✅  Total pairs:   {len(df_out):,}")
    print(f"  ✅  Saved to:     {TRAIN_PAIRS}")

    print(f"\n  Sample training pairs (3 random):")
    rng = random.Random(42)
    if len(df_out) > 0:
        for i in rng.sample(range(len(df_out)), min(3, len(df_out))):
            row = df_out.iloc[i]
            print(f"\n    [Route {row['route']}] {ROUTE_NAMES[row['route']]}")
            print(f"      INPUT  : {row['input'][:120]}")
            print(f"      TARGET : {row['target'][:120]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class VerbalizerDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_in  = CFG["max_input_len"]
        self.max_out = CFG["max_output_len"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inp = self.tokenizer(str(row["input"]), max_length=self.max_in,
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        tgt = self.tokenizer(str(row["target"]), max_length=self.max_out,
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        labels = tgt["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      inp["input_ids"].squeeze(0),
            "attention_mask": inp["attention_mask"].squeeze(0),
            "labels":         labels,
        }


def stratified_split(df, val_frac=0.1, seed=42):
    """Split by image_id within each route to avoid image leakage."""
    rng = random.Random(seed)
    train_dfs, val_dfs = [], []
    for route in range(6):
        sub = df[df["route"] == route]
        if len(sub) == 0: continue
        if "img_id" in sub.columns and sub["img_id"].notna().any():
            img_ids = sub["img_id"].dropna().astype(str).unique().tolist()
            rng.shuffle(img_ids)
            cut = int(len(img_ids) * (1 - val_frac))
            train_ids = set(img_ids[:cut])
            t = sub[sub["img_id"].astype(str).isin(train_ids)]
            v = sub[~sub["img_id"].astype(str).isin(train_ids)]
        else:
            idx = list(range(len(sub))); rng.shuffle(idx)
            cut = int(len(idx) * (1 - val_frac))
            t = sub.iloc[idx[:cut]]; v = sub.iloc[idx[cut:]]
        train_dfs.append(t); val_dfs.append(v)
    return (pd.concat(train_dfs).reset_index(drop=True),
            pd.concat(val_dfs).reset_index(drop=True))


def compute_quality(pred: str, gt: str) -> dict:
    pred_tokens = pred.lower().split()
    gt_tokens   = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return {"token_f1": 0.0, "exact": 0.0}
    pset = set(pred_tokens); gset = set(gt_tokens)
    common = pset & gset
    p = len(common) / len(pset) if pset else 0
    r = len(common) / len(gset) if gset else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    exact = 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0
    return {"token_f1": f1, "exact": exact}


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'='*72}")
    print(f"  Stage 5 — T5-small Verbalizer Training")
    print(f"{'='*72}\n")

    if not os.path.exists(TRAIN_PAIRS):
        print(f"❌  Training pairs CSV not found at {TRAIN_PAIRS}")
        print(f"    Run with --mode build_data first")
        return

    df = pd.read_csv(TRAIN_PAIRS)
    df = df.dropna(subset=["input", "target"])
    print(f"  Total pairs after dropna: {len(df):,}")

    train_df, val_df = stratified_split(df, val_frac=0.1)
    print(f"  Stratified split — Train: {len(train_df):,}  Val: {len(val_df):,}\n")

    print(f"  Loading {CFG['model_name']} ...")
    tokenizer = T5Tokenizer.from_pretrained(CFG["model_name"])
    model     = T5ForConditionalGeneration.from_pretrained(CFG["model_name"])
    model = model.to(CFG["device"])
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✅  Loaded — total {n_params:,} params, trainable {n_train:,}\n")

    train_ds = VerbalizerDataset(train_df, tokenizer)
    val_ds   = VerbalizerDataset(val_df,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                           shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=0)

    optim = torch.optim.AdamW(model.parameters(),
                                lr=CFG["lr"],
                                weight_decay=CFG["weight_decay"])
    n_steps = len(train_dl) * CFG["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(n_steps * CFG["warmup_ratio"]),
        num_training_steps=n_steps)

    best_val_loss = float("inf"); patience = 0; history = []
    print(f"  Training for up to {CFG['epochs']} epochs ...\n")

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        tot_loss = 0.0; n_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:2d} train")
        for batch in pbar:
            batch = {k: v.to(CFG["device"]) for k, v in batch.items()}
            out = model(**batch); loss = out.loss
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optim.step(); scheduler.step()
            tot_loss += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = tot_loss / max(n_batches, 1)

        model.eval()
        v_loss = 0.0; v_n = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch:2d} val  "):
                batch = {k: v.to(CFG["device"]) for k, v in batch.items()}
                v_loss += model(**batch).loss.item(); v_n += 1
        val_loss = v_loss / max(v_n, 1)

        print(f"\n  Sample generations (epoch {epoch}):")
        if len(val_df) >= 3:
            sample_idx = random.sample(range(len(val_df)), 3)
            for idx in sample_idx:
                row = val_df.iloc[idx]
                inp = tokenizer(str(row["input"]),
                                max_length=CFG["max_input_len"],
                                truncation=True, return_tensors="pt"
                                ).to(CFG["device"])
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inp, max_length=CFG["max_output_len"],
                        num_beams=2, early_stopping=True)
                gen = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                print(f"    [R{row['route']}] S4: '{row['pred'][:50]}'")
                print(f"           Gen: '{gen[:100]}'")
                print(f"           GT : '{row['target'][:100]}'\n")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss})
        print(f"  Epoch {epoch:2d}  |  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience = 0
            torch.save({
                "model_state":   model.state_dict(),
                "tokenizer_name": CFG["model_name"],
                "epoch":          epoch,
                "val_loss":       val_loss,
            }, CKPT_PATH)
            print(f"   ✅  Saved best checkpoint → {CKPT_PATH}")
        else:
            patience += 1
            print(f"   ⏳  No improvement ({patience}/{CFG['early_stop_pat']})")
            if patience >= CFG["early_stop_pat"]:
                print(f"\n   🛑  Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(TRAIN_LOG_PATH, index=False)
    print(f"\n  ✅  Training log → {TRAIN_LOG_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Inference test
# ─────────────────────────────────────────────────────────────────────────────
def test():
    print(f"\n{'='*72}")
    print(f"  Stage 5 — Inference Quality Test")
    print(f"{'='*72}\n")

    if not os.path.exists(CKPT_PATH):
        print(f"❌  No checkpoint at {CKPT_PATH}")
        return
    if not os.path.exists(TRAIN_PAIRS):
        print(f"❌  No training pairs CSV at {TRAIN_PAIRS}")
        return

    print(f"  Loading model ...")
    tokenizer = T5Tokenizer.from_pretrained(CFG["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(CFG["model_name"])
    ckpt = torch.load(CKPT_PATH, map_location=CFG["device"],
                       weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(CFG["device"]).eval()
    print(f"  ✅  Loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})\n")

    df = pd.read_csv(TRAIN_PAIRS)
    _, val_df = stratified_split(df, val_frac=0.1)

    print(f"  Computing quality metrics on validation set ...")
    metrics = {r: {"token_f1": [], "exact": []} for r in range(6)}
    rng = random.Random(0); sample_outputs = []
    with torch.no_grad():
        for r in range(6):
            sub = val_df[val_df["route"] == r]
            if len(sub) == 0: continue
            idx_list = rng.sample(range(len(sub)), min(100, len(sub)))
            for k, idx in enumerate(tqdm(idx_list, desc=f"  Route {r}")):
                row = sub.iloc[idx]
                inp = tokenizer(str(row["input"]),
                                max_length=CFG["max_input_len"],
                                truncation=True, return_tensors="pt"
                                ).to(CFG["device"])
                gen_ids = model.generate(
                    **inp, max_length=CFG["max_output_len"],
                    num_beams=4, early_stopping=True)
                gen = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                q = compute_quality(gen, str(row["target"]))
                metrics[r]["token_f1"].append(q["token_f1"])
                metrics[r]["exact"].append(q["exact"])
                if k < 2:
                    sample_outputs.append({
                        "route": r, "pred": row["pred"],
                        "gen": gen, "target": row["target"],
                        "question": row["question"],
                    })

    print(f"\n{'='*72}")
    print(f"  Per-Route Quality Metrics")
    print(f"{'='*72}\n")
    print(f"  {'Route':<24} {'Token F1':>10} {'Exact':>10} {'Samples':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    overall_f1 = []; overall_ex = []
    for r in range(6):
        if not metrics[r]["token_f1"]: continue
        f1 = float(np.mean(metrics[r]["token_f1"])) * 100
        ex = float(np.mean(metrics[r]["exact"])) * 100
        n  = len(metrics[r]["token_f1"])
        overall_f1.extend(metrics[r]["token_f1"])
        overall_ex.extend(metrics[r]["exact"])
        print(f"  R{r}: {ROUTE_NAMES[r]:<20} {f1:>9.2f}% {ex:>9.2f}% {n:>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    if overall_f1:
        print(f"  {'OVERALL':<24} "
              f"{np.mean(overall_f1)*100:>9.2f}% "
              f"{np.mean(overall_ex)*100:>9.2f}% "
              f"{len(overall_f1):>10}")
    print()

    print(f"{'='*72}")
    print(f"  Sample Generated Sentences")
    print(f"{'='*72}\n")
    for s in sample_outputs:
        print(f"  Route {s['route']} ({ROUTE_NAMES[s['route']]}):")
        print(f"    Q   : {s['question'][:90]}")
        print(f"    S4  : {s['pred'][:90]}   ← structured Stage 4 output")
        print(f"    S5  : {s['gen'][:140]}   ← Stage 5 sentence")
        print(f"    GT  : {s['target'][:140]}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                         choices=["build_data", "train", "test", "all"])
    args = parser.parse_args()
    if args.mode in ["build_data", "all"]: build_training_pairs()
    if args.mode in ["train", "all"]:      train()
    if args.mode in ["test", "all"]:       test()


if __name__ == "__main__":
    main()
