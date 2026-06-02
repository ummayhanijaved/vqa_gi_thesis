#!/usr/bin/env python3
"""
=============================================================================
Stage 3 Recompute — Diagnostic & Decision Tool
=============================================================================

Before recomputing anything, this READ-ONLY script answers the questions
that determine which path is safe:

  Q1. How much does current infer_route() disagree with the stored
      (learned-router) route in the existing cache?
      → If small: recompute is low-risk for Stage 4
      → If large: Stage 4 retraining is mandatory after recompute

  Q2. Can we recover img_id for the EXISTING cache via the reproducible
      train_test_split(seed=42)?
      → Checks if cache order matches the split order

  Q3. What's the actual collision rate for (question,answer)→img_id?

USAGE:
    python stage3_recompute_plan.py
=============================================================================
"""
import os
import sys
from collections import Counter

import torch

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
CACHE_DIR = os.path.join(PROJECT, "cache", "stage3_features")


def header(s):
    print(f"\n{'='*72}\n  {s}\n{'='*72}\n")


def main():
    from stage4_revised import infer_route, CFG as S4_CFG
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    print(f"\n{'█'*72}")
    print(f"  STAGE 3 RECOMPUTE — DIAGNOSTIC (READ-ONLY)")
    print(f"{'█'*72}")

    # ── Q1: Routing disagreement ──────────────────────────────────────────
    header("Q1 — Does current infer_route disagree with stored route?")

    train_cache = os.path.join(CACHE_DIR, "stage3_cache_train.pt")
    recs = torch.load(train_cache, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(recs):,} train cache records")

    stored = Counter()
    recomputed = Counter()
    mismatches = 0
    for r in recs:
        s_route = r.get("route", -1)
        i_route = infer_route(r["question"])
        stored[s_route] += 1
        recomputed[i_route] += 1
        if s_route != i_route:
            mismatches += 1

    print(f"\n  Stored route (learned router) distribution:")
    for rt in range(6):
        print(f"     Route {rt}: {stored.get(rt, 0):>7,}")
    print(f"\n  Current infer_route() distribution:")
    for rt in range(6):
        print(f"     Route {rt}: {recomputed.get(rt, 0):>7,}")

    pct = mismatches / max(len(recs), 1) * 100
    print(f"\n  ➤  Mismatches: {mismatches:,} / {len(recs):,}  ({pct:.1f}%)")
    if pct < 5:
        print(f"  ✅  LOW disagreement → recompute is LOW-RISK for Stage 4")
        print(f"      (Stage 4 heads would see nearly identical routing)")
    elif pct < 20:
        print(f"  🟠  MODERATE disagreement → Stage 4 may need light retraining")
    else:
        print(f"  🔴  HIGH disagreement → Stage 4 RETRAIN required after recompute")
        print(f"      (routing changed substantially)")

    # ── Q2: Can we recover img_id via reproducible split? ─────────────────
    header("Q2 — Can reproducible split(seed=42) recover img_id?")

    raw = load_from_disk(S4_CFG["data_dir"])
    n_hf_train = len(raw["train"])
    print(f"  HF train size: {n_hf_train:,}")

    indices = list(range(n_hf_train))
    tr_idx, va_idx = train_test_split(indices, test_size=0.20,
                                       random_state=42)
    print(f"  Reproduced split: train={len(tr_idx):,}  val={len(va_idx):,}")
    print(f"  Cache sizes:      train={len(recs):,}  "
          f"val={len(torch.load(os.path.join(CACHE_DIR,'stage3_cache_val.pt'), map_location='cpu', weights_only=False)):,}")

    # Does reproduced train size match cache?
    if len(tr_idx) == len(recs):
        print(f"  ✅  Split train size MATCHES cache train size!")
        print(f"      BUT cache was built with shuffle — order may differ.")
        # Verify by checking if questions match in split order
        print(f"\n  Checking if cache[i] matches HF[tr_idx[i]] (first 5):")
        matches = 0
        for i in range(min(5, len(recs))):
            cache_q = recs[i]["question"].strip().lower()[:40]
            hf_q = raw["train"][tr_idx[i]]["question"].strip().lower()[:40]
            m = "✅" if cache_q == hf_q else "❌"
            if cache_q == hf_q: matches += 1
            print(f"     [{i}] cache: '{cache_q}'")
            print(f"         hf   : '{hf_q}'  {m}")
        if matches == 5:
            print(f"\n  ✅✅  Cache IS in split order! img_id recoverable directly!")
        else:
            print(f"\n  ⚠️   Cache NOT in split order (was shuffled).")
            print(f"      Need (question,answer) join instead.")
    else:
        print(f"  ⚠️   Split size ({len(tr_idx):,}) != cache ({len(recs):,})")
        print(f"      Some samples were dropped during caching (missing images?)")

    # ── Q3: Collision rate for (question, answer) → img_id ────────────────
    header("Q3 — (question, answer) → img_id collision rate")

    key_to_imgs = {}
    for s in raw["train"]:
        k = (s["question"].strip().lower(),
             s.get("answer", "").strip().lower())
        key_to_imgs.setdefault(k, []).append(
            str(s.get("img_id", s.get("image_id", ""))))

    matched = uniq = 0
    for r in recs:
        k = (r["question"].strip().lower(), r["answer"].strip().lower())
        imgs = key_to_imgs.get(k, [])
        if imgs: matched += 1
        if len(imgs) == 1: uniq += 1

    print(f"  Cache records matched at all:   {matched:,} / {len(recs):,}")
    print(f"  Cache records UNIQUELY matched: {uniq:,} / {len(recs):,}  "
          f"({uniq/max(len(recs),1)*100:.1f}%)")

    # ── Final recommendation ──────────────────────────────────────────────
    header("RECOMMENDATION")
    print(f"  Based on the three checks above:")
    print(f"")
    print(f"  • Routing mismatch: {pct:.1f}%")
    print(f"  • Split-order recovery: see Q2")
    print(f"  • Unique (q,a) match: {uniq/max(len(recs),1)*100:.1f}%")
    print(f"")
    print(f"  Decision logic:")
    print(f"    - If routing mismatch <5% AND split-order matches:")
    print(f"        → Recompute fresh is SAFE, minimal Stage 4 impact")
    print(f"    - If routing mismatch high:")
    print(f"        → Either keep old cache + recover img_id by split order,")
    print(f"          OR recompute + retrain Stage 4 (clean but slow)")
    print(f"")
    print(f"  Paste this output and we'll pick the exact path.\n")


if __name__ == "__main__":
    main()
