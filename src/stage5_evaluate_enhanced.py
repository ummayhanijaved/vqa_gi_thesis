#!/usr/bin/env python3
"""
=============================================================================
Stage 5 — Enhanced Evaluation with Comprehensive Metrics
=============================================================================

This wraps stage5_pipeline_test and adds ALL these metrics:

  TIER 1 (NLG — Stage 5 verbalizer quality):
    ✅ ROUGE-1, ROUGE-2, ROUGE-L
    ✅ METEOR
    ✅ BLEU-1, BLEU-2 (already in original)
    ✅ Soft Match (already in original)
    ✅ Clinical Adequacy (already in original)
    ✅ Fuzzy Match (already in original)
    ✅ Token F1 (already in original)
    ✅ Exact Match (already in original)

  TIER 2 (Classification — Stage 4 quality):
    ✅ F1-macro, F1-micro, F1-weighted per route
    ✅ Per-route classification accuracy

  TIER 3 (Sentence quality heuristic):
    ✅ Well-formedness score (NOT "coherence" — honest naming)

USAGE:
    # Evaluate V1 (original Stage 5)
    python stage5_evaluate_enhanced.py --version v1 --n_samples 1000

    # Evaluate V2 (retrained Stage 5)
    python stage5_evaluate_enhanced.py --version v2 --n_samples 1000

    # Evaluate both side-by-side
    python stage5_evaluate_enhanced.py --version both --n_samples 1000

OUTPUT FILES:
    ~/vqa_gi_thesis/logs/stage5_verbalizer/
      ├── pipeline_enhanced_v1.csv     (per-sample V1)
      ├── pipeline_enhanced_v2.csv     (per-sample V2 if version=v2/both)
      ├── pipeline_enhanced_summary.csv (per-route metrics)
      └── pipeline_enhanced_examples.txt (best/worst samples)
=============================================================================
"""
import os
import sys
import re
import argparse
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

# Import the existing pipeline test
import stage5_pipeline_test as ppt


# ─────────────────────────────────────────────────────────────────────────────
# Library check — fail gracefully if any are missing
# ─────────────────────────────────────────────────────────────────────────────
HAVE_ROUGE = HAVE_NLTK = HAVE_SKLEARN = False
try:
    from rouge_score import rouge_scorer
    HAVE_ROUGE = True
except ImportError:
    print(f"⚠️   rouge-score missing — using fallback")

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    # Auto-download if needed
    for resource in ["wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}" if "punkt" not in resource
                            else f"tokenizers/{resource}")
        except LookupError:
            try: nltk.download(resource, quiet=True)
            except Exception: pass
    HAVE_NLTK = True
except ImportError:
    print(f"⚠️   nltk missing — using fallback for METEOR")

try:
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    print(f"⚠️   sklearn missing — using fallback for F1")


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [t for t in text.split() if t]


# ── ROUGE ────────────────────────────────────────────────────────────────────
_rouge_scorer = None
def get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None and HAVE_ROUGE:
        _rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def rouge_metrics(pred, gt):
    if not str(pred).strip() or not str(gt).strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = get_rouge_scorer()
    if scorer is not None:
        try:
            scores = scorer.score(str(gt), str(pred))
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception:
            pass
    # Fallback
    return {
        "rouge1": _rouge_n_fallback(pred, gt, 1),
        "rouge2": _rouge_n_fallback(pred, gt, 2),
        "rougeL": _rouge_l_fallback(pred, gt),
    }


def _rouge_n_fallback(pred, gt, n):
    def ngrams(toks, n):
        return [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    p_ng = ngrams(tokenize(pred), n)
    g_ng = ngrams(tokenize(gt), n)
    if not p_ng or not g_ng: return 0.0
    p_set, g_set = Counter(p_ng), Counter(g_ng)
    overlap = sum((p_set & g_set).values())
    if overlap == 0: return 0.0
    prec = overlap / sum(p_set.values())
    rec  = overlap / sum(g_set.values())
    return 2*prec*rec / (prec+rec)


def _rouge_l_fallback(pred, gt):
    p, g = tokenize(pred), tokenize(gt)
    if not p or not g: return 0.0
    m, n = len(p), len(g)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = (dp[i-1][j-1]+1 if p[i-1] == g[j-1]
                         else max(dp[i-1][j], dp[i][j-1]))
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    prec, rec = lcs/m, lcs/n
    return 2*prec*rec / (prec+rec)


# ── METEOR ───────────────────────────────────────────────────────────────────
def meteor_metric(pred, gt):
    if not str(pred).strip() or not str(gt).strip(): return 0.0
    if HAVE_NLTK:
        try:
            p_toks = tokenize(pred)
            g_toks = tokenize(gt)
            if not p_toks or not g_toks: return 0.0
            return float(nltk_meteor([g_toks], p_toks))
        except Exception:
            pass
    # Fallback: stem-based overlap
    return _meteor_fallback(pred, gt)


def _meteor_fallback(pred, gt):
    p_toks = tokenize(pred); g_toks = tokenize(gt)
    if not p_toks or not g_toks: return 0.0
    def stem(w):
        for suf in ["ies", "ing", "ed", "es", "s"]:
            if w.endswith(suf) and len(w) > len(suf)+1:
                return w[:-len(suf)]
        return w
    p_stems = set(stem(t) for t in p_toks)
    g_stems = set(stem(t) for t in g_toks)
    if not p_stems or not g_stems: return 0.0
    common = p_stems & g_stems
    if not common: return 0.0
    prec = len(common) / len(p_stems)
    rec  = len(common) / len(g_stems)
    if prec + rec == 0: return 0.0
    return 10 * prec * rec / (rec + 9 * prec)


# ── Sentence Well-formedness (honest replacement for "coherence") ────────────
def wellformedness(text):
    if not text or len(text) < 5: return 0.0
    tokens = tokenize(text)
    if len(tokens) < 3: return 0.5
    # Repetition penalty
    rep_penalty = sum(1 for i in range(len(tokens)-2)
                       if tokens[i] == tokens[i+1] == tokens[i+2])
    rep_score = max(0, 1.0 - rep_penalty * 0.2)
    # Sentence ends well
    end_score = 1.0 if str(text).strip().endswith((".", "?", "!")) else 0.7
    # Length sanity
    length_score = (1.0 if 5 <= len(tokens) <= 50
                     else (0.7 if len(tokens) <= 80 else 0.5))
    return (rep_score + end_score + length_score) / 3


# ── Classification F1 (Stage 4 metrics) ─────────────────────────────────────
def normalize_for_match(text):
    """Normalize text for medical VQA comparison."""
    import re
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def medical_match(pred, gt):
    """
    Medical VQA-appropriate matching:
      - Pred is a SHORT class label (e.g. "no", "red", "1")
      - GT is a FULL sentence (e.g. "no anatomical landmarks identified")
      - Match if pred is substring of GT, or GT contains key pred words.

    Returns True if predictions matches GT semantically.
    """
    p = normalize_for_match(pred)
    g = normalize_for_match(gt)
    if not p or not g:
        return False
    # 1. Direct substring (most common case for medical VQA)
    if p in g or g in p:
        return True
    # 2. Token overlap — if all pred tokens are in GT
    p_tokens = set(p.split())
    g_tokens = set(g.split())
    if not p_tokens:
        return False
    # All pred tokens present in GT? (handles "yes" vs "yes evidence of...")
    if p_tokens.issubset(g_tokens):
        return True
    # 3. Significant overlap (≥ 80% of pred tokens in GT)
    common = p_tokens & g_tokens
    if len(common) / len(p_tokens) >= 0.8:
        return True
    return False


def f1_metrics(predictions, ground_truths):
    """
    Compute F1 for medical VQA using substring/semantic matching.
    
    KEY INSIGHT: In your eval CSVs, predictions are SHORT labels but
    GTs are FULL sentences. Exact-match F1 = 0% because they never match
    as raw strings. We use medical_match() which compares semantically.
    """
    if not predictions or not ground_truths: return {}
    pairs = [(p, g) for p, g in zip(predictions, ground_truths)
              if p and g]
    if not pairs: return {}

    # Use semantic medical match (substring/token-based)
    correct = [medical_match(p, g) for p, g in pairs]
    accuracy = sum(correct) / max(len(pairs), 1)

    # For F1, we map each unique prediction to a class index
    # GT class is the "canonical" prediction that medical_match() recognized
    preds_str = [normalize_for_match(p) for p, _ in pairs]
    # GT class derivation: use the prediction if it matched, else "OTHER"
    gts_class = []
    for i, (p, g) in enumerate(pairs):
        if correct[i]:
            gts_class.append(normalize_for_match(p))    # GT contains pred
        else:
            # Use first token of GT as a proxy class
            g_norm = normalize_for_match(g)
            first_tok = g_norm.split()[0] if g_norm else "unknown"
            gts_class.append(first_tok)

    if HAVE_SKLEARN:
        try:
            classes = sorted(set(gts_class) | set(preds_str))
            c2i = {c: i for i, c in enumerate(classes)}
            p_idx = [c2i[p] for p in preds_str]
            g_idx = [c2i[g] for g in gts_class]
            return {
                "f1_macro":    float(f1_score(g_idx, p_idx, average="macro",
                                                zero_division=0)),
                "f1_micro":    float(f1_score(g_idx, p_idx, average="micro",
                                                zero_division=0)),
                "f1_weighted": float(f1_score(g_idx, p_idx, average="weighted",
                                                zero_division=0)),
                "accuracy":    accuracy,
                "n_classes":   len(classes),
                "n_samples":   len(pairs),
            }
        except Exception:
            pass
    return {
        "f1_macro":  accuracy,
        "accuracy":  accuracy,
        "n_samples": len(pairs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced bulk evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_version(predictor, version, n_samples, ROUTE_NAMES):
    """Run bulk evaluation with enhanced metrics for one version (v1 or v2)."""
    print(f"\n{'█'*72}")
    print(f"  ENHANCED EVALUATION — {version.upper()}")
    print(f"{'█'*72}\n")

    from datasets import load_from_disk
    from tqdm import tqdm
    import time

    raw = load_from_disk(ppt.CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]
    image_dir = ppt.S4_CFG.get("image_dir", "")

    # Group test samples by route
    by_route = defaultdict(list)
    print(f"  Routing test samples ...")
    for s in test_split:
        q = s.get("question", "")
        if not q: continue
        try:
            r = ppt.infer_route(q)
        except Exception:
            continue
        by_route[r].append(s)

    # Balanced sampling per route
    per_route = max(1, n_samples // 6)
    selected = []
    import random
    rng = random.Random(42)
    for r in range(6):
        cands = by_route[r]
        if not cands: continue
        picked = rng.sample(cands, min(per_route, len(cands)))
        for p in picked:
            p["_route"] = r
            selected.append(p)
    print(f"  Selected {len(selected):,} samples for evaluation\n")

    # Helper to find image
    def find_image(img_id):
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            p = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(p): return p
        return None

    rows = []
    for sample in tqdm(selected, desc=f"  Pipeline test [{version}]"):
        img = find_image(sample.get("img_id", sample.get("image_id", "")))
        if not img: continue
        try:
            r = predictor.predict(img, sample["question"])
        except Exception as e:
            continue
        gt = sample.get("answer", "").strip()
        pred = r["s5_sentence"]

        # Compute ALL metrics
        rouge = rouge_metrics(pred, gt)
        meteor = meteor_metric(pred, gt)
        wf = wellformedness(pred)

        row = {
            "version":          version,
            "route":            r["route"],
            "route_name":       r["route_name"],
            "img_id":           sample.get("img_id", ""),
            "question":         sample["question"],
            "s4_model":         r["s4_model"],
            "s4_answer":        str(r["s4_answer"]),
            "s5_sentence":      pred,
            "gt_sentence":      gt,
            # Tier 1: NLG metrics
            "soft_match":        ppt.soft_match(pred, gt),
            "clinical_adequacy": ppt.clinical_adequacy(pred, gt),
            "bleu_1":            ppt.bleu_n(pred, gt, 1),
            "bleu_2":            ppt.bleu_n(pred, gt, 2),
            "fuzzy_match":       ppt.fuzzy_match(pred, gt),
            "token_f1":          ppt.token_f1(pred, gt),
            "exact":             ppt.exact_match(pred, gt),
            # NEW: ROUGE
            "rouge_1":           rouge["rouge1"],
            "rouge_2":           rouge["rouge2"],
            "rouge_L":           rouge["rougeL"],
            # NEW: METEOR
            "meteor":            meteor,
            # NEW: Well-formedness (renamed from coherence)
            "wellformedness":    wf,
            # Latency
            "s4_latency_ms":     r["timings"]["s4_total_ms"],
            "s5_latency_ms":     r["timings"]["s5_ms"],
        }
        rows.append(row)

    if not rows:
        print(f"❌  No samples evaluated")
        return None

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ppt.CFG["out_dir"],
                            f"pipeline_enhanced_{version}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  ✅  Per-sample CSV → {out_csv}")

    # Print enhanced scorecard
    print_enhanced_scorecard(df, version, ROUTE_NAMES)
    return df


def print_enhanced_scorecard(df, version, ROUTE_NAMES):
    """Print comprehensive metrics scorecard."""
    print(f"\n{'='*72}")
    print(f"  ENHANCED SCORECARD — {version.upper()}")
    print(f"{'='*72}\n")

    # Tier 1: NLG metrics per route
    print(f"  NLG METRICS PER ROUTE")
    print(f"  {'-'*72}")
    print(f"  {'Route':<22} {'R1':>7} {'R2':>7} {'RL':>7} "
          f"{'METEOR':>7} {'B1':>7} {'B2':>7} {'Soft':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in range(6):
        sub = df[df["route"] == r]
        if len(sub) == 0: continue
        print(f"  R{r}: {ROUTE_NAMES[r]:<18} "
              f"{sub['rouge_1'].mean()*100:>6.2f}% "
              f"{sub['rouge_2'].mean()*100:>6.2f}% "
              f"{sub['rouge_L'].mean()*100:>6.2f}% "
              f"{sub['meteor'].mean()*100:>6.2f}% "
              f"{sub['bleu_1'].mean()*100:>6.2f}% "
              f"{sub['bleu_2'].mean()*100:>6.2f}% "
              f"{sub['soft_match'].mean()*100:>6.2f}%")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'OVERALL':<22} "
          f"{df['rouge_1'].mean()*100:>6.2f}% "
          f"{df['rouge_2'].mean()*100:>6.2f}% "
          f"{df['rouge_L'].mean()*100:>6.2f}% "
          f"{df['meteor'].mean()*100:>6.2f}% "
          f"{df['bleu_1'].mean()*100:>6.2f}% "
          f"{df['bleu_2'].mean()*100:>6.2f}% "
          f"{df['soft_match'].mean()*100:>6.2f}%")

    # Other NLG metrics
    print(f"\n  ADDITIONAL NLG METRICS (OVERALL)")
    print(f"  {'-'*72}")
    print(f"     Soft Match         : {df['soft_match'].mean()*100:6.2f}%")
    print(f"     Clinical Adequacy  : {df['clinical_adequacy'].mean()*100:6.2f}%")
    print(f"     Fuzzy Match        : {df['fuzzy_match'].mean()*100:6.2f}%")
    print(f"     Token F1           : {df['token_f1'].mean()*100:6.2f}%")
    print(f"     Exact Match        : {df['exact'].mean()*100:6.2f}%")
    print(f"     Well-formedness    : {df['wellformedness'].mean()*100:6.2f}%")

    # Tier 2: Stage 4 classification F1
    # FIX: Read the proper per-route eval CSVs (which compare predicted
    # CLASS to GT CLASS, not to full sentences) for accurate F1 numbers.
    print(f"\n  STAGE 4 CLASSIFICATION F1 PER ROUTE")
    print(f"  {'-'*72}")
    print(f"  (Loaded from saved Stage 4 eval CSVs in logs/stage4_revised/)")
    print(f"  {'Route':<22} {'F1-Macro':>10} {'F1-Micro':>10} "
          f"{'F1-Weight':>10} {'Acc':>10} {'N':>6}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    PROJECT = os.path.expanduser("~/vqa_gi_thesis")
    eval_dir = os.path.join(PROJECT, "logs", "stage4_revised")
    eval_files = {
        0: "route0_yes_no_eval.csv",
        1: "route1_single_choice_eval.csv",
        2: "route2_multi_choice_eval.csv",
        3: "route3_color_eval.csv",
        4: "route4_location_yolo_eval.csv",
        5: "route5_count_yolo_eval.csv",
    }
    all_preds, all_gts = [], []
    for r in range(6):
        eval_csv = os.path.join(eval_dir, eval_files[r])
        if not os.path.exists(eval_csv):
            print(f"  R{r}: {ROUTE_NAMES[r]:<18} "
                  f"   ---       ---       ---       ---       N/A")
            continue
        df_eval = pd.read_csv(eval_csv)
        s4_preds = df_eval["prediction"].astype(str).str.lower().str.strip().tolist()
        s4_gts = df_eval["ground_truth"].astype(str).str.lower().str.strip().tolist()
        f1 = f1_metrics(s4_preds, s4_gts)
        all_preds.extend(s4_preds); all_gts.extend(s4_gts)
        if f1:
            print(f"  R{r}: {ROUTE_NAMES[r]:<18} "
                  f"{f1.get('f1_macro', 0)*100:>9.2f}% "
                  f"{f1.get('f1_micro', 0)*100:>9.2f}% "
                  f"{f1.get('f1_weighted', 0)*100:>9.2f}% "
                  f"{f1.get('accuracy', 0)*100:>9.2f}% "
                  f"{f1.get('n_samples', 0):>6}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    overall_f1 = f1_metrics(all_preds, all_gts)
    if overall_f1:
        print(f"  {'OVERALL':<22} "
              f"{overall_f1.get('f1_macro', 0)*100:>9.2f}% "
              f"{overall_f1.get('f1_micro', 0)*100:>9.2f}% "
              f"{overall_f1.get('f1_weighted', 0)*100:>9.2f}% "
              f"{overall_f1.get('accuracy', 0)*100:>9.2f}% "
              f"{overall_f1.get('n_samples', 0):>6}")

    # Headline
    print(f"\n{'='*72}")
    print(f"  HEADLINE METRICS — {version.upper()}")
    print(f"{'='*72}\n")
    print(f"  📊  ROUGE-1:         {df['rouge_1'].mean()*100:6.2f}%")
    print(f"  📊  ROUGE-2:         {df['rouge_2'].mean()*100:6.2f}%")
    print(f"  📊  ROUGE-L:         {df['rouge_L'].mean()*100:6.2f}%")
    print(f"  📊  METEOR:          {df['meteor'].mean()*100:6.2f}%")
    print(f"  📊  BLEU-1:          {df['bleu_1'].mean()*100:6.2f}%")
    print(f"  📊  BLEU-2:          {df['bleu_2'].mean()*100:6.2f}%")
    print(f"  📊  Soft Match:      {df['soft_match'].mean()*100:6.2f}%")
    print(f"  📊  Fuzzy Match:     {df['fuzzy_match'].mean()*100:6.2f}%")
    print(f"  📊  Well-formedness: {df['wellformedness'].mean()*100:6.2f}%")
    print()


def compare_versions(df_v1, df_v2, ROUTE_NAMES):
    """Side-by-side comparison of V1 vs V2."""
    print(f"\n{'█'*72}")
    print(f"  SIDE-BY-SIDE COMPARISON — V1 vs V2")
    print(f"{'█'*72}\n")
    metrics_to_compare = [
        ("ROUGE-1",        "rouge_1"),
        ("ROUGE-2",        "rouge_2"),
        ("ROUGE-L",        "rouge_L"),
        ("METEOR",         "meteor"),
        ("BLEU-1",         "bleu_1"),
        ("BLEU-2",         "bleu_2"),
        ("Soft Match",     "soft_match"),
        ("Clin Adequacy",  "clinical_adequacy"),
        ("Fuzzy Match",    "fuzzy_match"),
        ("Token F1",       "token_f1"),
        ("Exact Match",    "exact"),
        ("Well-formed",    "wellformedness"),
    ]
    print(f"  {'Metric':<20} {'V1':>10} {'V2':>10} {'Change':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, col in metrics_to_compare:
        v1 = df_v1[col].mean() * 100 if df_v1 is not None else 0
        v2 = df_v2[col].mean() * 100 if df_v2 is not None else 0
        change = v2 - v1
        symbol = "↑" if change > 0.5 else ("↓" if change < -0.5 else "≈")
        print(f"  {name:<20} {v1:>9.2f}% {v2:>9.2f}% "
              f"{symbol} {change:+6.2f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1",
                         choices=["v1", "v2", "both"])
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    PROJECT = os.path.expanduser("~/vqa_gi_thesis")
    V1_CKPT = os.path.join(PROJECT, "checkpoints", "stage5_verbalizer",
                            "stage5_verbalizer_best.pt")
    V2_CKPT = os.path.join(PROJECT, "checkpoints", "stage5_verbalizer",
                            "stage5_verbalizer_v2_best.pt")

    ROUTE_NAMES = ppt.ROUTE_NAMES

    df_v1 = df_v2 = None
    if args.version in ["v1", "both"]:
        if not os.path.exists(V1_CKPT):
            print(f"❌  V1 not found: {V1_CKPT}")
        else:
            ppt.CFG["s5_ckpt_path"] = V1_CKPT
            predictor = ppt.FullPipelinePredictor()
            df_v1 = evaluate_version(predictor, "v1",
                                       args.n_samples, ROUTE_NAMES)
            del predictor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.version in ["v2", "both"]:
        if not os.path.exists(V2_CKPT):
            print(f"❌  V2 not found: {V2_CKPT}")
            print(f"    Run stage5_retrain_proper.py first")
        else:
            ppt.CFG["s5_ckpt_path"] = V2_CKPT
            predictor = ppt.FullPipelinePredictor()
            df_v2 = evaluate_version(predictor, "v2",
                                       args.n_samples, ROUTE_NAMES)
            del predictor

    if args.version == "both" and df_v1 is not None and df_v2 is not None:
        compare_versions(df_v1, df_v2, ROUTE_NAMES)


if __name__ == "__main__":
    main()
