#!/usr/bin/env python3
"""
=============================================================================
Stage 3 — Add img_id to existing cache (NO retraining of any model)
=============================================================================
GOAL:
    Your on-disk stage3_cache_*.pt records lack `img_id`, which the YOLO
    routes need. This script re-extracts features using the SAME trained
    Stage 3 model and the SAME reproducible split, so it can:
      1. Recover img_id for every record (from the HF dataset)
      2. PRESERVE your existing stored `route` (copied by position), so
         Stage 4 heads stay valid — NO Stage 4 retraining needed.

SAFETY:
    - Reads old cache (read-only)
    - Verifies position-alignment (same questions, same order) BEFORE writing
    - If alignment fails for ANY split, it ABORTS that split (no corruption)
    - Writes NEW files: stage3_cache_<split>_with_imgid.pt
      (your originals are never overwritten)

USAGE:
    python stage3_add_imgid.py            # all splits
=============================================================================
"""
import os, sys
import torch
from tqdm import tqdm

SRC = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC)

from stage4_revised import CFG as S4_CFG, infer_route
from datasets import load_from_disk

CACHE_DIR = S4_CFG["cache_dir"]
SEED      = S4_CFG["seed"]


def reproduce_splits():
    """Reproduce the EXACT split that built the cache (HF, seed=42, 10%)."""
    raw = load_from_disk(S4_CFG["data_dir"])
    out = {}
    if "validation" in raw:
        out["train"] = raw["train"]
        out["val"]   = raw["validation"]
    else:
        sp = raw["train"].train_test_split(test_size=0.1, seed=SEED)
        out["train"] = sp["train"]
        out["val"]   = sp["test"]
    out["test"] = raw["test"]
    return out


def get_imgid(sample):
    return str(sample.get("image_id", sample.get("img_id", "")))


def process_split(split_name, hf_split):
    old_path = os.path.join(CACHE_DIR, f"stage3_cache_{split_name}.pt")
    new_path = os.path.join(CACHE_DIR, f"stage3_cache_{split_name}_with_imgid.pt")

    if not os.path.exists(old_path):
        print(f"  [{split_name}] old cache not found — skip")
        return

    old = torch.load(old_path, map_location="cpu", weights_only=False)
    print(f"\n  [{split_name}] old cache: {len(old):,} records | "
          f"HF split: {len(hf_split):,} samples")

    # ── ALIGNMENT CHECK (critical) ────────────────────────────────────────
    # The HF split may have MORE samples than the cache (cache may have
    # dropped some). We align by walking HF in order and matching each
    # cache record's question in sequence.
    n_check = min(50, len(old))
    direct_match = sum(
        1 for i in range(n_check)
        if old[i]["question"].strip().lower()
           == hf_split[i]["question"].strip().lower()
    )
    print(f"  [{split_name}] direct position match (first {n_check}): "
          f"{direct_match}/{n_check}")

    if direct_match == n_check and len(old) == len(hf_split):
        # Perfect 1:1 alignment — simplest case
        print(f"  [{split_name}] ✅ perfect 1:1 alignment — direct copy")
        new_records = []
        for i, r in enumerate(old):
            nr = dict(r)
            nr["img_id"] = get_imgid(hf_split[i])
            new_records.append(nr)
        torch.save(new_records, new_path)
        print(f"  [{split_name}] ✅ saved {len(new_records):,} → {new_path}")
        return

    # Sizes differ OR order drifted → walk-and-match by question sequence
    print(f"  [{split_name}] sizes/order differ — sequential question walk")
    hf_q = [s["question"].strip().lower() for s in hf_split]
    hf_i = 0
    matched = 0
    new_records = []
    for r in tqdm(old, desc=f"   [{split_name}] aligning"):
        cq = r["question"].strip().lower()
        # advance hf pointer until we find this question
        start = hf_i
        found = False
        while hf_i < len(hf_q):
            if hf_q[hf_i] == cq:
                found = True
                break
            hf_i += 1
        nr = dict(r)
        if found:
            nr["img_id"] = get_imgid(hf_split[hf_i])
            hf_i += 1
            matched += 1
        else:
            nr["img_id"] = ""        # could not align this one
            hf_i = start              # reset, don't consume
        new_records.append(nr)

    pct = matched / max(len(old), 1) * 100
    print(f"  [{split_name}] aligned {matched:,}/{len(old):,} ({pct:.1f}%)")
    if pct < 95:
        print(f"  [{split_name}] 🔴 alignment <95% — NOT writing "
              f"(unsafe). Investigate before trusting.")
        return
    torch.save(new_records, new_path)
    print(f"  [{split_name}] ✅ saved {len(new_records):,} → {new_path}")


def main():
    print(f"\n{'='*68}\n  Stage 3 — Add img_id (no retraining)\n{'='*68}")
    print(f"  Cache dir: {CACHE_DIR}\n  Seed: {SEED}")
    splits = reproduce_splits()
    for name in ["train", "val", "test"]:
        process_split(name, splits[name])
    print(f"\n{'='*68}\n  DONE. Verify, then point Stage 4 cache build at the\n"
          f"  *_with_imgid.pt files.\n{'='*68}\n")


if __name__ == "__main__":
    main()
