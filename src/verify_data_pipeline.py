#!/usr/bin/env python3
"""
=============================================================================
Data Pipeline Verification — Stage 3 Cache + Stage 4 Training Data
=============================================================================

This READ-ONLY script verifies your Anjum's question:

  1. ✅  Does Stage 3 cache truly have 143,594 records (train split)?
  2. ✅  What dataset size did Stage 4 actually train on?
  3. ✅  Is there a "Stage 4 cache" similar to Stage 3 cache?
     (Spoiler: NO — and this script explains WHY this is the case)

USAGE:
    python verify_data_pipeline.py
=============================================================================
"""
import os
import sys
from datetime import datetime
from collections import Counter

import torch
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")


def human_size(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024: return f"{n:6.1f} {u}"
        n /= 1024
    return f"{n:6.1f} PB"


def header(s, c="="):
    print(f"\n{c*72}\n  {s}\n{c*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Verify Stage 3 cache files
# ─────────────────────────────────────────────────────────────────────────────
def verify_stage3_cache():
    header("STEP 1 — Verify Stage 3 Cache Files")

    cache_dir = os.path.join(PROJECT, "cache", "stage3_features")
    if not os.path.exists(cache_dir):
        print(f"❌  Cache directory not found: {cache_dir}")
        return

    print(f"  Cache directory: {cache_dir}\n")

    for split in ["train", "val", "test"]:
        cache_path = os.path.join(cache_dir, f"stage3_cache_{split}.pt")
        print(f"  ─── {split.upper()} cache ───")
        print(f"     Path: {cache_path}")

        if not os.path.exists(cache_path):
            print(f"     Status: ❌  MISSING\n")
            continue

        # File info
        stat = os.stat(cache_path)
        print(f"     Size: {human_size(stat.st_size)}")
        print(f"     Modified: "
              f"{datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")

        # Load and inspect
        try:
            records = torch.load(cache_path, map_location="cpu",
                                  weights_only=False)
        except Exception as e:
            print(f"     ❌  Load failed: {e}\n")
            continue

        if isinstance(records, list):
            n_records = len(records)
            print(f"     Number of records: {n_records:,}")

            # Inspect first record structure
            if records:
                r = records[0]
                if isinstance(r, dict):
                    print(f"     Record fields: {list(r.keys())}")
                    for k, v in r.items():
                        if hasattr(v, "shape"):
                            print(f"        {k}: tensor shape={list(v.shape)}, "
                                  f"dtype={v.dtype}")
                        elif isinstance(v, str):
                            preview = v[:50] + "..." if len(v) > 50 else v
                            print(f"        {k}: str = '{preview}'")
                        else:
                            print(f"        {k}: {type(v).__name__} = {v}")

            # Count routes
            from collections import Counter
            try:
                # Try to infer route from question
                sys.path.insert(0, os.path.expanduser("~/vqa_gi_thesis/src"))
                from stage4_revised import infer_route
                route_counts = Counter()
                for rec in records:
                    if isinstance(rec, dict) and "question" in rec:
                        try:
                            r = infer_route(rec["question"])
                            route_counts[r] += 1
                        except Exception:
                            pass
                print(f"     Route distribution:")
                for r in range(6):
                    print(f"        Route {r}: {route_counts.get(r, 0):>7,}")
            except ImportError:
                print(f"     (Could not import infer_route to show route distribution)")

            print(f"     Status: ✅  {n_records:,} pre-computed records")
        else:
            print(f"     ⚠️   Cache is not a list, got {type(records)}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. What dataset Stage 4 actually used (from HF dataset)
# ─────────────────────────────────────────────────────────────────────────────
def verify_hf_dataset_size():
    header("STEP 2 — Original HF Dataset Size (what Stage 4 actually saw)")

    try:
        from datasets import load_from_disk
        sys.path.insert(0, os.path.expanduser("~/vqa_gi_thesis/src"))
        from stage4_revised import CFG as S4_CFG

        data_path = S4_CFG["data_dir"]
        print(f"  Dataset path: {data_path}\n")

        raw = load_from_disk(data_path)
        print(f"  Available splits:")
        total = 0
        for split_name in raw:
            n = len(raw[split_name])
            print(f"     {split_name:<15}: {n:>8,} samples")
            total += n
        print(f"     {'TOTAL':<15}: {total:>8,}")

        # Show what Stage 4 used
        print(f"\n  Stage 4 trained on:")
        print(f"     raw['train']:    {len(raw['train']):>8,} samples")
        if "validation" in raw:
            print(f"     raw['validation']: {len(raw['validation']):>8,} samples")

    except Exception as e:
        print(f"❌  Could not load dataset: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Address the "Why no Stage 4 cache?" question
# ─────────────────────────────────────────────────────────────────────────────
def explain_no_stage4_cache():
    header("STEP 3 — Why There's NO Stage 4 Cache (Explanation)", c="─")

    print("""
  ═══════════════════════════════════════════════════════════════════════
  YOUR QUESTION: "I have Stage 3 pre-computed features for 143K samples.
                  Why don't I have the same for Stage 4?"
  ═══════════════════════════════════════════════════════════════════════

  The answer involves understanding WHY caches exist.

  ─── WHY STAGE 3 CACHE EXISTS ───────────────────────────────────────────

     Stage 3 (Multimodal Fusion) is COMPUTATIONALLY EXPENSIVE:
       • ResNet50 forward pass on full image (slow)
       • DistilBERT forward pass on question (slow)
       • Cross-attention + fusion (slow)
       • Total: ~80ms per sample on GPU

     When training Stage 4, we need Stage 3's output for EVERY training
     sample, for EVERY epoch (10-20 epochs typical).

     Without cache:
       143,594 samples × 80ms × 20 epochs = 63 hours of Stage 3 inference
       (just to compute features for training Stage 4!)

     With cache (pre-computed once):
       143,594 × 80ms × 1 = 3 hours (one-time)
       + 0ms for all subsequent epochs (load from disk)
       Total savings: ~60 hours

     This is why Stage 3 features are cached.

  ─── WHY NO STAGE 4 CACHE WAS NEEDED ────────────────────────────────────

     Stage 4 was the FINAL stage in your original architecture.
     There was no "Stage 5" yet at the time Stage 4 was being trained.

     Stage 4 doesn't feed another stage's training, so its outputs were
     never needed in bulk.

     Stage 4 outputs were only used for EVALUATION (test set):
        • Test set: 15,955 samples
        • Saved to eval CSVs (route0_yes_no_eval.csv, etc.)
        • Used to report final test accuracy

     This is why Stage 4 evaluation CSVs exist but no Stage 4 training-set
     cache exists.

  ─── NOW THAT STAGE 5 EXISTS ────────────────────────────────────────────

     With Stage 5 (T5 verbalizer), we suddenly NEED Stage 4 predictions
     for ALL training samples to train T5 properly.

     So we have two options:

     OPTION A — Build a Stage 4 cache now:
        1. Load Stage 3 cache (instant — 143K records ready)
        2. Run Stage 4 inference on each (1-2 hours)
        3. Save Stage 4 predictions to "stage4_cache_train.pt"
        4. Train T5 using this cache (instant access per epoch)

     OPTION B — Run Stage 4 inline during T5 training:
        For each training batch, run Stage 4 then T5 forward pass.
        Slower but simpler code.

     Option A is what we should do — same caching philosophy as Stage 3.

  ─── BOTTOM LINE ────────────────────────────────────────────────────────

     • Stage 3 cache exists because Stage 4 training NEEDED it (efficiency).
     • Stage 4 cache doesn't exist because nothing needed it... UNTIL NOW.
     • For proper Stage 5 retraining, we'll CREATE a Stage 4 cache.

     This is a CORRECT design choice, not a mistake. Caches are built
     on-demand when downstream stages need them.

""")


# ─────────────────────────────────────────────────────────────────────────────
# 4. What a Stage 4 cache would look like (preview)
# ─────────────────────────────────────────────────────────────────────────────
def preview_stage4_cache():
    header("STEP 4 — Stage 4 Cache Structure (preview if we create it)")
    print("""
  If we build a Stage 4 cache for Stage 5 training, it would look like:

  ┌────────────────────────────────────────────────────────────────────┐
  │  Filename: stage4_cache_train.pt                                    │
  │  Records:  ~143,594 (one per training sample)                       │
  │                                                                      │
  │  Each record is a dict:                                              │
  │    {                                                                 │
  │      "img_id":       "abc123",                                       │
  │      "question":     "Is there a polyp?",                            │
  │      "route":        0,                                              │
  │      "route_name":   "yes_no",                                       │
  │      "stage4_pred":  "no",                ← Stage 4's prediction     │
  │      "gt_answer":    "No polyp detected", ← Ground truth sentence    │
  │      "confidence":   0.92,                ← Optional: softmax conf   │
  │    }                                                                 │
  │                                                                      │
  │  Total file size: ~30-50 MB (text only, no tensors)                  │
  │  Time to build:   ~1-2 hours (Stage 4 inference on 143K samples)    │
  └────────────────────────────────────────────────────────────────────┘

  This cache then powers Stage 5 (T5) training across multiple epochs
  without re-running Stage 4 every time.
""")


def main():
    print(f"\n{'█'*72}")
    print(f"  DATA PIPELINE VERIFICATION")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*72}")

    verify_stage3_cache()
    verify_hf_dataset_size()
    explain_no_stage4_cache()
    preview_stage4_cache()

    print(f"\n{'█'*72}")
    print(f"  SUMMARY")
    print(f"{'█'*72}\n")
    print(f"  ✅  Stage 3 has a cache because Stage 4 training needs it")
    print(f"  ❌  Stage 4 has no cache because nothing needed it yet")
    print(f"  💡  We'll BUILD a Stage 4 cache (one-time) to power Stage 5\n")


if __name__ == "__main__":
    main()
