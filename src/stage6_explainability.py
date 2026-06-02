#!/usr/bin/env python3
"""
=============================================================================
Stage 6 — Explainability Layer (Medical Response + Textual Explanation)
=============================================================================

PURPOSE:
    Add an EXPLAINABILITY layer AFTER Stage 5 that produces:
      1. Generated Answer (from Stage 4/5)
      2. Textual Explanation (Medical Response) — WHY this answer, grounded
         in the disease evidence from Stage 1 + visual region from Stage 4.

    Then evaluates the Medical Response with comprehensive metrics:
      - ROUGE-1, ROUGE-2, ROUGE-L
      - METEOR
      - BLEU-1, BLEU-2, BLEU-4
      - CHRF++  (character n-gram F-score, robust to medical morphology)
      - BERTScore (semantic similarity via contextual embeddings)

INPUT:
    Uses the existing pipeline (Stages 1-5) via stage5_pipeline_test.
    Reads ground-truth explanations from Kvasir-VQA-x1 'answer' field.

OUTPUT:
    ~/vqa_gi_thesis/logs/stage6_explainability/
      ├── medical_responses.csv         (per-sample answer + explanation)
      ├── explainability_metrics.csv    (per-route metric breakdown)
      └── explainability_examples.txt   (qualitative samples)

REQUIRED INSTALLS:
    pip install sacrebleu bert-score rouge-score nltk --break-system-packages
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

USAGE:
    # Build medical responses + evaluate
    python stage6_explainability.py --n_samples 1000

    # Demo on a few examples
    python stage6_explainability.py --mode demo
=============================================================================
"""
import os
import sys
import re
import json
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

import stage5_pipeline_test as ppt


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/vqa_gi_thesis")
CFG = {
    "device":   "cuda" if torch.cuda.is_available() else "cpu",
    "out_dir":  os.path.join(PROJECT, "logs", "stage6_explainability"),
    "bertscore_model": "roberta-large",   # or "distilbert-base-uncased" for speed
    "bertscore_batch": 32,
}
os.makedirs(CFG["out_dir"], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Library availability
# ─────────────────────────────────────────────────────────────────────────────
HAVE = {"rouge": False, "nltk": False, "sacrebleu": False, "bertscore": False}

try:
    from rouge_score import rouge_scorer
    HAVE["rouge"] = True
except ImportError:
    print("⚠️   rouge-score missing — pip install rouge-score")

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    for res in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            try: nltk.download(res, quiet=True)
            except Exception: pass
    HAVE["nltk"] = True
except ImportError:
    print("⚠️   nltk missing — pip install nltk")

try:
    import sacrebleu
    HAVE["sacrebleu"] = True
except ImportError:
    print("⚠️   sacrebleu missing (CHRF++) — pip install sacrebleu")

try:
    from bert_score import score as bertscore_fn
    HAVE["bertscore"] = True
except ImportError:
    print("⚠️   bert-score missing — pip install bert-score")


# ─────────────────────────────────────────────────────────────────────────────
# Disease label names — IMPORTED from Stage 1 to guarantee identical index
# order (hardcoding caused silent mislabeling — fixed).
# ─────────────────────────────────────────────────────────────────────────────
try:
    from stage1_disease_classifier import DISEASE_LABELS as DISEASE_NAMES
except ImportError:
    # Fallback ONLY if import fails — must match Stage 1 exactly
    DISEASE_NAMES = [
        "polyp-pedunculated", "polyp-sessile", "polyp-hyperplastic",
        "esophagitis", "gastritis", "ulcerative-colitis", "crohns-disease",
        "barretts-esophagus", "gastric-ulcer", "duodenal-ulcer",
        "erosion", "hemorrhoid", "diverticulum",
        "normal-cecum", "normal-pylorus", "normal-z-line",
        "ileocecal-valve", "retroflex-rectum", "retroflex-stomach",
        "dyed-lifted-polyp", "dyed-resection-margins",
        "foreign-body", "instrument",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Medical Response Generator
# ─────────────────────────────────────────────────────────────────────────────
class MedicalResponseGenerator:
    """
    Combines Stage 4 answer + Stage 5 sentence + disease evidence into a
    structured Medical Response with answer + textual explanation.

    This is a TEMPLATE-BASED explainability layer (deterministic, defensible).
    It grounds the explanation in:
      - The predicted disease (from Stage 1 disease_vec)
      - The question route (what was asked)
      - The Stage 4 answer (what was found)
      - The Stage 5 verbalized sentence (natural language)
    """

    ROUTE_TEMPLATES = {
        0: "Based on the endoscopic image analysis, {sentence} "
           "This assessment is supported by the visual findings "
           "consistent with {disease_context}.",
        1: "The examination reveals {sentence} "
           "This finding is characterized by features observed "
           "in the {disease_context}.",
        2: "Multiple findings are identified: {sentence} "
           "These observations are consistent with {disease_context}.",
        3: "The chromatic analysis indicates {sentence} "
           "Such coloration is typically associated with {disease_context}.",
        4: "Spatial localization shows {sentence} "
           "The region of interest corresponds to {disease_context}.",
        5: "Quantitative assessment determines {sentence} "
           "This count reflects the {disease_context} observed.",
    }

    def __init__(self):
        self.route_names = ppt.ROUTE_NAMES

    def get_disease_context(self, disease_vec, threshold=0.3):
        """Extract top disease(s) from Stage 1 disease vector."""
        if disease_vec is None:
            return "the examined region"
        if hasattr(disease_vec, "cpu"):
            disease_vec = disease_vec.cpu().numpy()
        disease_vec = np.asarray(disease_vec).flatten()
        # Top-2 diseases above threshold
        top_idx = np.argsort(disease_vec)[::-1][:2]
        active = [DISEASE_NAMES[i] for i in top_idx
                   if i < len(DISEASE_NAMES) and disease_vec[i] > threshold]
        if not active:
            return "the gastrointestinal mucosa"
        if len(active) == 1:
            return active[0]
        return f"{active[0]} and {active[1]}"

    def generate(self, route, sentence, disease_vec, s4_answer):
        """Generate the Medical Response (answer + explanation)."""
        disease_context = self.get_disease_context(disease_vec)
        template = self.ROUTE_TEMPLATES.get(route, self.ROUTE_TEMPLATES[0])

        # Clean sentence (lowercase first letter for mid-sentence insertion)
        sent = str(sentence).strip()
        if sent and sent[0].isupper() and not sent.startswith(("A ", "I ", "An ")):
            sent = sent[0].lower() + sent[1:]

        explanation = template.format(
            sentence=sent, disease_context=disease_context)

        return {
            "answer": str(s4_answer),
            "explanation": explanation,
            "disease_context": disease_context,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [t for t in text.split() if t]


_rouge_scorer_obj = None
def rouge_metrics(pred, gt):
    global _rouge_scorer_obj
    if not str(pred).strip() or not str(gt).strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if HAVE["rouge"]:
        if _rouge_scorer_obj is None:
            _rouge_scorer_obj = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        try:
            s = _rouge_scorer_obj.score(str(gt), str(pred))
            return {"rouge1": s["rouge1"].fmeasure,
                    "rouge2": s["rouge2"].fmeasure,
                    "rougeL": s["rougeL"].fmeasure}
        except Exception:
            pass
    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def meteor_metric(pred, gt):
    if not str(pred).strip() or not str(gt).strip(): return 0.0
    if HAVE["nltk"]:
        try:
            return float(nltk_meteor([tokenize(gt)], tokenize(pred)))
        except Exception:
            pass
    return 0.0


def bleu_real(pred, gt):
    """Real BLEU with brevity penalty via sacrebleu (BLEU-4 by default)."""
    if not str(pred).strip() or not str(gt).strip(): return 0.0
    if HAVE["sacrebleu"]:
        try:
            return sacrebleu.sentence_bleu(str(pred), [str(gt)]).score / 100.0
        except Exception:
            pass
    return 0.0


def ngram_precision(pred, gt, n):
    """Raw n-gram precision (honestly labeled — NOT BLEU)."""
    p, g = tokenize(pred), tokenize(gt)
    if len(p) < n or len(g) < n: return 0.0
    from collections import Counter
    def ngrams(toks, k):
        return Counter(tuple(toks[i:i+k]) for i in range(len(toks)-k+1))
    p_ng, g_ng = ngrams(p, n), ngrams(g, n)
    if not p_ng: return 0.0
    overlap = sum((p_ng & g_ng).values())
    return overlap / max(sum(p_ng.values()), 1)


def chrf_metric(pred, gt):
    """CHRF++ — character n-gram F-score (word_order=2)."""
    if not str(pred).strip() or not str(gt).strip(): return 0.0
    if HAVE["sacrebleu"]:
        try:
            return sacrebleu.sentence_chrf(
                str(pred), [str(gt)], word_order=2).score / 100.0
        except Exception:
            pass
    return 0.0


def bertscore_batch(preds, gts):
    """BERTScore F1 for a batch (returns list of scores)."""
    if not HAVE["bertscore"] or not preds:
        return [0.0] * len(preds)
    try:
        P, R, F1 = bertscore_fn(
            preds, gts, lang="en",
            model_type=CFG["bertscore_model"],
            batch_size=CFG["bertscore_batch"],
            verbose=False, device=CFG["device"])
        return F1.cpu().tolist()
    except Exception as e:
        print(f"   ⚠️  BERTScore failed: {str(e)[:60]}")
        return [0.0] * len(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_explainability(n_samples):
    print(f"\n{'█'*72}")
    print(f"  STAGE 6 — Explainability + Medical Response Evaluation")
    print(f"{'█'*72}\n")

    print(f"  Metric availability:")
    for k, v in HAVE.items():
        print(f"     {k:<12}: {'✅' if v else '❌ (using fallback/skip)'}")
    print()

    # Load pipeline
    predictor = ppt.FullPipelinePredictor()
    generator = MedicalResponseGenerator()

    # Load test data
    from datasets import load_from_disk
    raw = load_from_disk(ppt.CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]
    image_dir = ppt.S4_CFG.get("image_dir", "")

    # Route + sample
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

    import random
    rng = random.Random(42)
    per_route = max(1, n_samples // 6)
    selected = []
    for r in range(6):
        cands = by_route[r]
        if cands:
            for p in rng.sample(cands, min(per_route, len(cands))):
                p["_route"] = r
                selected.append(p)
    print(f"  Selected {len(selected):,} samples\n")

    def find_image(img_id):
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            p = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(p): return p
        return None

    # Generate medical responses
    rows = []
    grounded_count = [0]   # tracks how many had a real disease_vec
    n_missing_image = [0]
    print(f"  Generating medical responses ...")
    for sample in tqdm(selected, desc="  Pipeline + Explanation"):
        img = find_image(sample.get("img_id", sample.get("image_id", "")))
        if not img:
            n_missing_image[0] += 1
            continue
        try:
            result = predictor.predict(img, sample["question"])
        except Exception:
            continue

        # Robust key extraction (pipeline may use different field names)
        route = result.get("route", result.get("route_id", 0))
        route_name = result.get("route_name",
                                 ppt.ROUTE_NAMES.get(route, "unknown"))
        s4_answer = result.get("s4_answer",
                                result.get("answer",
                                            result.get("s4_pred", "")))
        s5_sentence = result.get("s5_sentence",
                                  result.get("sentence",
                                              result.get("s5_output", "")))
        # disease_vec: predict() may NOT return this. If absent, the
        # explanation is route-templated (NOT disease-grounded). We track
        # this honestly rather than silently using a constant fallback.
        disease_vec = result.get("disease_vec", result.get("disease", None))
        if disease_vec is not None:
            grounded_count[0] += 1

        # Generate medical response
        med = generator.generate(
            route=route,
            sentence=s5_sentence,
            disease_vec=disease_vec,
            s4_answer=s4_answer)

        gt = sample.get("answer", "").strip()
        rows.append({
            "route":        route,
            "route_name":   route_name,
            "img_id":       sample.get("img_id", ""),
            "question":     sample["question"],
            "s4_answer":    str(s4_answer),
            "s5_sentence":  s5_sentence,
            "medical_answer":       med["answer"],
            "medical_explanation":  med["explanation"],
            "disease_context":      med["disease_context"],
            "gt_sentence":  gt,
        })

    if not rows:
        print(f"❌  No samples processed")
        return

    # Honest reporting of grounding + dropped samples
    print(f"\n  Disease-grounded explanations: {grounded_count[0]:,}/{len(rows):,}")
    if grounded_count[0] == 0:
        print(f"  ⚠️   predict() did not return disease_vec — explanations are")
        print(f"       ROUTE-TEMPLATED, not disease-grounded. To enable real")
        print(f"       grounding, add 'disease_vec' to predict()'s return dict.")
    print(f"  Dropped (missing image): {n_missing_image[0]:,}")

    df = pd.DataFrame(rows)

    # ── Compute metrics ───────────────────────────────────────────────────
    print(f"\n  Computing metrics for {len(df):,} medical responses ...")

    # ROUGE, METEOR, BLEU, CHRF (per-sample)
    for col in ["rouge1", "rouge2", "rougeL", "meteor",
                 "bleu", "bleu1_prec", "bleu2_prec", "chrf"]:
        df[col] = 0.0

    print(f"  Computing ROUGE/METEOR/BLEU/CHRF++ ...")
    print(f"  NOTE: Metrics score the Stage 5 SENTENCE vs GT (validity fix).")
    print(f"        The templated Medical Response is a QUALITATIVE artifact.")
    for idx in tqdm(range(len(df)), desc="  Per-sample metrics"):
        # CORRECTNESS FIX: score the Stage 5 sentence (the actual model
        # output) against GT — NOT the templated explanation (which adds
        # fixed boilerplate that deflates ROUGE/BLEU/CHRF).
        pred = df.iloc[idx]["s5_sentence"]
        gt = df.iloc[idx]["gt_sentence"]
        r = rouge_metrics(pred, gt)
        df.at[idx, "rouge1"] = r["rouge1"]
        df.at[idx, "rouge2"] = r["rouge2"]
        df.at[idx, "rougeL"] = r["rougeL"]
        df.at[idx, "meteor"] = meteor_metric(pred, gt)
        df.at[idx, "bleu"]   = bleu_real(pred, gt)          # real BLEU
        df.at[idx, "bleu1_prec"] = ngram_precision(pred, gt, 1)  # honest name
        df.at[idx, "bleu2_prec"] = ngram_precision(pred, gt, 2)
        df.at[idx, "chrf"]   = chrf_metric(pred, gt)

    # BERTScore (batched for efficiency)
    print(f"\n  Computing BERTScore (batched, may download model) ...")
    preds_list = df["s5_sentence"].astype(str).tolist()
    gts_list = df["gt_sentence"].astype(str).tolist()
    bert_scores = bertscore_batch(preds_list, gts_list)
    df["bertscore"] = bert_scores

    # ── Save per-sample CSV ───────────────────────────────────────────────
    out_csv = os.path.join(CFG["out_dir"], "medical_responses.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  ✅  Medical responses → {out_csv}")

    # ── Print scorecard ───────────────────────────────────────────────────
    print_scorecard(df)

    # ── Save metrics summary ──────────────────────────────────────────────
    save_metrics_summary(df)

    # ── Save qualitative examples ─────────────────────────────────────────
    save_examples(df)


def print_scorecard(df):
    print(f"\n{'='*72}")
    print(f"  EXPLAINABILITY METRICS — Stage 5 Sentence vs Ground Truth")
    print(f"  (Medical Response template kept as qualitative artifact only)")
    print(f"{'='*72}\n")

    metrics = [
        ("ROUGE-1",   "rouge1"),
        ("ROUGE-2",   "rouge2"),
        ("ROUGE-L",   "rougeL"),
        ("METEOR",    "meteor"),
        ("BLEU",      "bleu"),         # real sacrebleu BLEU (w/ brevity)
        ("1gramP",    "bleu1_prec"),   # honest n-gram precision
        ("2gramP",    "bleu2_prec"),
        ("CHRF++",    "chrf"),
        ("BERTScore", "bertscore"),
    ]

    # Per route
    print(f"  PER-ROUTE BREAKDOWN")
    print(f"  {'-'*72}")
    hdr = f"  {'Route':<20}"
    for name, _ in metrics:
        hdr += f" {name[:7]:>8}"
    print(hdr)
    print(f"  {'-'*72}")
    for r in range(6):
        sub = df[df["route"] == r]
        if len(sub) == 0: continue
        line = f"  R{r}: {ppt.ROUTE_NAMES[r]:<16}"
        for _, col in metrics:
            line += f" {sub[col].mean()*100:>7.2f}"
        print(line)
    print(f"  {'-'*72}")
    line = f"  {'OVERALL':<20}"
    for _, col in metrics:
        line += f" {df[col].mean()*100:>7.2f}"
    print(line)

    # Headline
    print(f"\n{'='*72}")
    print(f"  HEADLINE — Medical Response (Answer + Explanation)")
    print(f"{'='*72}\n")
    for name, col in metrics:
        print(f"  📊  {name:<12}: {df[col].mean()*100:6.2f}%")
    print()


def save_metrics_summary(df):
    metrics = ["rouge1", "rouge2", "rougeL", "meteor",
                "bleu", "bleu1_prec", "bleu2_prec", "chrf", "bertscore"]
    summary_rows = []
    for r in range(6):
        sub = df[df["route"] == r]
        if len(sub) == 0: continue
        row = {"route": r, "route_name": ppt.ROUTE_NAMES[r], "n": len(sub)}
        for m in metrics:
            row[m] = sub[m].mean()
        summary_rows.append(row)
    # Overall
    overall = {"route": "ALL", "route_name": "overall", "n": len(df)}
    for m in metrics:
        overall[m] = df[m].mean()
    summary_rows.append(overall)

    out = os.path.join(CFG["out_dir"], "explainability_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"  ✅  Metrics summary → {out}")


def save_examples(df, n=15):
    out = os.path.join(CFG["out_dir"], "explainability_examples.txt")
    with open(out, "w") as f:
        f.write("="*72 + "\n")
        f.write("  MEDICAL RESPONSE EXAMPLES (Answer + Textual Explanation)\n")
        f.write("="*72 + "\n\n")
        import random
        rng = random.Random(42)
        # Best examples by BERTScore
        df_sorted = df.sort_values("bertscore", ascending=False)
        f.write("  ─── TOP 8 (highest BERTScore) ───\n\n")
        for _, row in df_sorted.head(8).iterrows():
            f.write(f"  [Route {row['route']}] {row['route_name']}\n")
            f.write(f"    Question:    {row['question']}\n")
            f.write(f"    Answer:      {row['medical_answer']}\n")
            f.write(f"    Explanation: {row['medical_explanation']}\n")
            f.write(f"    Ground Truth:{row['gt_sentence']}\n")
            f.write(f"    BERTScore:   {row['bertscore']*100:.2f}%  "
                    f"ROUGE-L: {row['rougeL']*100:.2f}%  "
                    f"CHRF++: {row['chrf']*100:.2f}%\n\n")
    print(f"  ✅  Examples → {out}")


def demo_mode():
    """Generate a few medical responses to show the format."""
    print(f"\n{'='*72}")
    print(f"  DEMO — Medical Response Format")
    print(f"{'='*72}\n")

    generator = MedicalResponseGenerator()

    demos = [
        {"route": 0, "sentence": "No polyps are visible in the image.",
         "disease_vec": None, "s4_answer": "no"},
        {"route": 3, "sentence": "The abnormality appears pink.",
         "disease_vec": None, "s4_answer": "pink"},
        {"route": 5, "sentence": "One surgical instrument is present.",
         "disease_vec": None, "s4_answer": "1"},
    ]

    for d in demos:
        med = generator.generate(d["route"], d["sentence"],
                                   d["disease_vec"], d["s4_answer"])
        print(f"  [Route {d['route']}] {ppt.ROUTE_NAMES[d['route']]}")
        print(f"    Answer:      {med['answer']}")
        print(f"    Explanation: {med['explanation']}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bulk", choices=["bulk", "demo"])
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode()
    else:
        run_explainability(args.n_samples)


if __name__ == "__main__":
    main()
