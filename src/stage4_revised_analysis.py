"""
=============================================================================
STAGE 4 REVISED — Full Analysis & Evaluation Report
=============================================================================
Generates comprehensive analysis for the revised Stage 4 architecture
(DistilBERT Routes 0–3 + YOLO Routes 4–5).

Produces:
  Per-route:
    - Confusion matrix (raw counts + normalised recall)
    - Per-class Precision / Recall / F1 bar charts
    - Sample predictions table
    - Results CSV

  Summary (all routes together):
    - Accuracy comparison bar chart (all 6 routes)
    - Macro F1 comparison
    - Route-by-route metrics summary table
    - Training loss curves (if logs available)
    - Baseline comparison (MLP vs DistilBERT+YOLO)
    - Question type distribution pie chart

USAGE:
    # Analyse a single route
    python stage4_revised_analysis.py --route 0

    # Analyse all routes + produce summary dashboard
    python stage4_revised_analysis.py --route all

    # Only produce the summary dashboard (no per-route details)
    python stage4_revised_analysis.py --summary_only

OUTPUT DIRECTORY:
    ~/vqa_gi_thesis/logs/stage4_revised/analysis/

=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# ── Project paths ─────────────────────────────────────────────────────────────
HOME    = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "vqa_gi_thesis")
sys.path.insert(0, os.path.join(PROJECT, "src"))

# ── Colour palette ────────────────────────────────────────────────────────────
DARK   = "#0D1117"
PANEL  = "#161B22"
WHITE  = "#E6EDF3"
GREY   = "#8B949E"
TEAL   = "#00B4D8"
GREEN  = "#2ECC71"
RED    = "#E74C3C"
AMBER  = "#F39C12"
PURPLE = "#9B59B6"
BLUE   = "#3498DB"
PINK   = "#E91E8C"

ROUTE_NAMES = {
    0: "yes_no",        1: "single_choice",
    2: "multi_choice",  3: "color",
    4: "location",      5: "count",
}
ROUTE_LABELS = {
    0: "Yes / No",      1: "Single Choice",
    2: "Multi Choice",  3: "Colour",
    4: "Location",      5: "Count",
}
ROUTE_MODELS = {
    0: "DistilBERT",            1: "DistilBERT",
    2: "DistilBERT",            3: "DistilBERT",
    4: "YOLO-Seg (FT)",         5: "YOLO-Det (FT)",
}
ROUTE_COLORS = {
    0: TEAL, 1: GREEN, 2: PURPLE,
    3: AMBER, 4: BLUE, 5: PINK,
}

# MLP baseline results (from Stage 4 MLP — 73.97% overall)
BASELINE_MLP = {
    # MLP baseline accuracies per route (from Stage 4 MLP, 73.97% overall)
    0: 0.8200, 1: 0.7100, 2: 0.6800,
    3: 0.7500, 4: 0.6500, 5: 0.7200,
}

OUT_DIR = os.path.join(PROJECT, "logs", "stage4_revised", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — dark axis styling
# ─────────────────────────────────────────────────────────────────────────────
def dark_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363D")
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PREDICTIONS FROM SAVED EVAL CSV  or  RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def load_or_run_predictions(route: int):
    """
    Try to load from saved eval CSV first (DistilBERT or YOLO).
    If not found, run model inference to generate predictions.
    Returns (preds, gts) as lists of strings.
    """
    route_name = ROUTE_NAMES[route]

    # Try multiple possible CSV locations
    candidates = [
        os.path.join(PROJECT, "logs", "stage4_revised",
                     f"route{route}_{route_name}_eval.csv"),
        os.path.join(PROJECT, "logs", "stage4_revised",
                     f"route{route}_{route_name}_yolo_eval.csv"),
    ]
    for csv_path in candidates:
        if os.path.exists(csv_path):
            print(f"   📂  Loading saved eval results: {csv_path}")
            df = pd.read_csv(csv_path)
            return df["prediction"].astype(str).tolist(), \
                   df["ground_truth"].astype(str).tolist()

    csv_path = candidates[0]   # default for error message below

    # YOLO routes — no auto-inference; user must run training first
    if route in (4, 5):
        print(f"   ⚠️   No YOLO eval CSV found for route {route}.")
        print(f"   Run: python stage4_revised.py --mode train --route 4")
        return [], []

    # DistilBERT routes — run inference
    print(f"   🔍  No saved eval found — running inference for route {route}")
    try:
        from stage4_revised import (
            DistilBERTAnswerModel, DistilBERTRouteDataset,
            CFG, cache_stage3_features,
            FusionExtractor, TextPreprocessor, infer_route,
        )
        from preprocessing import build_image_transform
        from datasets import load_from_disk
        from transformers import DistilBertTokenizerFast

        ckpt_path = os.path.join(
            PROJECT, "checkpoints", "stage4_revised",
            f"stage4_revised_{route_name}_best.pt")
        assert os.path.exists(ckpt_path), \
            f"No checkpoint at {ckpt_path}. Train first: " \
            f"python stage4_revised.py --mode train --route {route}"

        ckpt      = torch.load(ckpt_path, map_location=CFG["device"],
                               weights_only=False)
        vocab     = ckpt["vocab"]
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            DistilBERTAnswerModel.MODEL_NAME)
        model     = DistilBERTAnswerModel(vocab_per_route={route: vocab})
        model.load_state_dict(ckpt["model_state"])
        model     = model.to(CFG["device"])
        model.eval()

        extractor  = FusionExtractor(CFG["stage3_ckpt"])
        text_prep  = TextPreprocessor()
        raw        = load_from_disk(CFG["data_dir"])
        test_cache = cache_stage3_features(
            extractor, text_prep, raw["test"], "test", CFG["cache_dir"])

        test_ds = DistilBERTRouteDataset(
            test_cache, route, tokenizer, vocab, CFG["max_input_len"])
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0)

        all_preds, all_gts = [], []
        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"  Predicting route {route}"):
                fused   = batch["fused"].to(CFG["device"])
                disease = batch["disease"].to(CFG["device"])
                inp_ids = batch["input_ids"].to(CFG["device"])
                att_msk = batch["attention_mask"].to(CFG["device"])
                preds   = model.predict(route, fused, disease,
                                        inp_ids, att_msk)
                all_preds.extend(preds)
                all_gts.extend(batch["answer_raw"])

        # Save for future use
        df = pd.DataFrame({"prediction": all_preds, "ground_truth": all_gts,
                            "correct": [int(p==g)
                                        for p,g in zip(all_preds,all_gts)]})
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"   ✅  Saved predictions → {csv_path}")
        return all_preds, all_gts

    except Exception as e:
        print(f"   ❌  Could not run inference: {e}")
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_text_answer(answer: str, route: int) -> str:
    """
    Mirror of stage4_revised.normalise_answer() — must match exactly so
    that analysis-time comparisons use the same normalisation as training.
    """
    a = answer.strip().lower()
    if route == 0:
        if a == "yes" or a.startswith("yes ") or a.startswith("yes,"):
            return "yes"
        if a == "no" or a.startswith("no ") or a.startswith("no,"):
            return "no"
        return "yes"
    if route == 3:
        colour_map = [
            ("green and black", "green and black"),
            ("green",           "green"),
            ("black",           "black"),
            ("red",             "red"),
            ("pink",            "pink"),
            ("orange",          "orange"),
            ("yellow",          "yellow"),
            ("blue",            "blue"),
            ("purple",          "purple"),
            ("white",           "white"),
            ("brown",           "brown"),
            ("transparent",     "transparent"),
            ("silver",          "mixed"),
            ("metallic",        "mixed"),
        ]
        for keyword, label in colour_map:
            if keyword in a:
                return label
        return "mixed"
    return a


def _yolo_route_correct(p: str, g: str, route: int) -> int:
    """Fuzzy correctness check for YOLO routes (same logic as eval)."""
    import re
    p_low = p.lower(); g_low = g.lower()
    if route == 4:
        # Position keyword match
        POS = ["central","centre","center","middle","upper","top","above",
               "lower","bottom","below","left","right"]
        def keys(t):
            found = set()
            for k in POS:
                if k in t:
                    if k in ("central","centre","center","middle"): found.add("central")
                    elif k in ("upper","top","above"):               found.add("upper")
                    elif k in ("lower","bottom","below"):            found.add("lower")
                    elif k == "left":  found.add("left")
                    elif k == "right": found.add("right")
            return found
        pk, gk = keys(p_low), keys(g_low)
        return 1 if (pk and gk and (pk & gk)) else 0
    elif route == 5:
        WTN = {"no":"0","zero":"0","one":"1","single":"1","two":"2",
               "three":"3","four":"4","five":"5","six":"6","seven":"7",
               "eight":"8","nine":"9","ten":"10",
               "multiple":"many","several":"many","many":"many"}
        def tok(t):
            m = re.search(r"\b(\d+)\b", t)
            if m:
                n = int(m.group(1))
                return str(n) if n <= 10 else "many"
            for w in sorted(WTN, key=len, reverse=True):
                if w in t: return WTN[w]
            return None
        pt, gt = tok(p_low), tok(g_low)
        if pt and gt and pt == gt:                           return 1
        if (pt and gt and pt.isdigit() and gt.isdigit()
            and abs(int(pt)-int(gt)) == 1):                  return 1
        return 0
    return int(p == g)


def _multilabel_f1(preds, gts):
    """
    Compute proper multi-label F1 metrics from comma-separated answer strings.
    Returns (sample_avg_f1, micro_f1, macro_f1).
    """
    def parse(s): return set(t.strip().lower() for t in s.split(",") if t.strip())

    sample_f1s = []
    all_tp, all_fp, all_fn = 0, 0, 0
    per_class_tp = {}; per_class_fp = {}; per_class_fn = {}
    all_classes = set()

    for p, g in zip(preds, gts):
        ps, gs = parse(p), parse(g)
        all_classes |= ps | gs
        tp = len(ps & gs); fp = len(ps - gs); fn = len(gs - ps)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sample_f1s.append(f1)
        all_tp += tp; all_fp += fp; all_fn += fn

        for c in ps & gs:  per_class_tp[c] = per_class_tp.get(c, 0) + 1
        for c in ps - gs:  per_class_fp[c] = per_class_fp.get(c, 0) + 1
        for c in gs - ps:  per_class_fn[c] = per_class_fn.get(c, 0) + 1

    micro_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_f1s = []
    for c in all_classes:
        tp = per_class_tp.get(c, 0)
        fp = per_class_fp.get(c, 0)
        fn = per_class_fn.get(c, 0)
        if tp + fp + fn == 0: continue
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro_f1s.append(f1)
    macro_f1 = sum(macro_f1s) / max(len(macro_f1s), 1)
    sample_avg = sum(sample_f1s) / max(len(sample_f1s), 1)
    return sample_avg, micro_f1, macro_f1


def compute_metrics(preds, gts, route):
    """Returns overall dict and per-class DataFrame."""
    if not preds:
        return None, None

    # Route 2 multi-label: report proper F1 metrics
    if route == 2:
        sample_f1, micro_f1, macro_f1 = _multilabel_f1(preds, gts)
        # Use sample-averaged F1 as the primary "accuracy" for the dashboard
        accuracy = sample_f1
        # Build a per-class DataFrame from the true multi-label classes
        all_classes = sorted({t.strip().lower() for s in gts
                               for t in s.split(",") if t.strip()})
        rows = []
        for cls in all_classes:
            tp = sum(1 for p, g in zip(preds, gts)
                     if cls in {t.strip().lower() for t in p.split(",")}
                     and cls in {t.strip().lower() for t in g.split(",")})
            fp = sum(1 for p, g in zip(preds, gts)
                     if cls in {t.strip().lower() for t in p.split(",")}
                     and cls not in {t.strip().lower() for t in g.split(",")})
            fn = sum(1 for p, g in zip(preds, gts)
                     if cls not in {t.strip().lower() for t in p.split(",")}
                     and cls in {t.strip().lower() for t in g.split(",")})
            sup = tp + fn
            if sup == 0: continue
            pr = tp / (tp + fp + 1e-9)
            re = tp / (tp + fn + 1e-9)
            f1 = 2 * pr * re / (pr + re + 1e-9)
            rows.append({"class": cls, "precision": round(pr, 4),
                          "recall": round(re, 4), "f1": round(f1, 4),
                          "support": sup, "tp": tp, "fp": fp, "fn": fn})
        df = pd.DataFrame(rows)
        overall = {
            "route"      : route, "route_name" : ROUTE_NAMES[route],
            "route_label": ROUTE_LABELS[route], "model" : ROUTE_MODELS[route],
            "accuracy"   : round(sample_f1, 4),
            "n_correct"  : int(sample_f1 * len(preds)),
            "n_total"    : len(preds),
            "macro_f1"   : round(macro_f1, 4),
            "macro_prec" : round(df["precision"].mean(), 4) if len(df) else 0.0,
            "macro_rec"  : round(df["recall"].mean(), 4)    if len(df) else 0.0,
            "n_classes"  : len(df),
            "micro_f1"   : round(micro_f1,   4),
            "sample_f1"  : round(sample_f1,  4),
            "is_multilabel": True,
        }
        return overall, df

    # YOLO routes: intelligent fuzzy matching
    if route in (4, 5):
        correct = [_yolo_route_correct(p, g, route)
                   for p, g in zip(preds, gts)]
        accuracy = sum(correct) / max(len(correct), 1)
        classes  = sorted(set(gts))
    elif route in (0, 3):
        # Routes 0 and 3 need GT normalisation (verbose strings → canonical)
        # Predictions are already normalised by the model output head
        gts_norm = [_normalise_text_answer(g, route) for g in gts]
        correct  = [int(p == g) for p, g in zip(preds, gts_norm)]
        accuracy = sum(correct) / max(len(correct), 1)
        # Use normalised GTs for per-class breakdown too
        gts      = gts_norm
        classes  = sorted(set(gts))
    else:
        # Route 1 — Single Choice: predicted vocab token must appear in
        # the verbose GT string (e.g. predict "ulcerative colitis" matches
        # GT "findings consistent with ulcerative colitis are present")
        correct = []
        gts_norm = []
        for p, g in zip(preds, gts):
            p_low = p.lower().strip()
            g_low = g.lower().strip()
            ok = (p_low == g_low) or (p_low and p_low in g_low)
            correct.append(int(ok))
            gts_norm.append(p_low if ok else g_low)
        accuracy = sum(correct) / max(len(correct), 1)
        gts      = gts_norm
        classes  = sorted(set(gts))

    rows = []
    for cls in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, gts) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, gts) if p != cls and g == cls)
        support = sum(1 for g in gts if g == cls)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({"class": cls, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "support": support, "tp": tp, "fp": fp, "fn": fn})

    df = pd.DataFrame(rows)
    overall = {
        "route"      : route,
        "route_name" : ROUTE_NAMES[route],
        "route_label": ROUTE_LABELS[route],
        "model"      : ROUTE_MODELS[route],
        "accuracy"   : round(accuracy, 4),
        "n_correct"  : sum(correct),
        "n_total"    : len(correct),
        "macro_f1"   : round(df["f1"].mean(), 4),
        "macro_prec" : round(df["precision"].mean(), 4),
        "macro_rec"  : round(df["recall"].mean(), 4),
        "n_classes"  : len(classes),
    }
    return overall, df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE A — CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(preds, gts, overall):
    MAX_CLASSES = 25
    route       = overall["route"]
    route_name  = overall["route_name"]

    gt_counts   = Counter(gts)
    top_classes = [c for c, _ in gt_counts.most_common(MAX_CLASSES)]
    pairs       = [(p, g) for p, g in zip(preds, gts) if g in top_classes]
    if not pairs:
        return None
    preds_f, gts_f = zip(*pairs)
    classes   = sorted(set(gts_f))
    n         = len(classes)
    cls_idx   = {c: i for i, c in enumerate(classes)}
    truncated = len(set(gts)) > MAX_CLASSES

    cm = np.zeros((n, n), dtype=int)
    for p, g in zip(preds_f, gts_f):
        gi = cls_idx.get(g)
        pi = cls_idx.get(p)
        if gi is not None and pi is not None:
            cm[gi, pi] += 1

    cm_norm  = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums,
                          out=np.zeros_like(cm_norm), where=row_sums != 0)

    cell_px = max(0.55, min(1.0, 16/n))
    fig_h   = max(7, n * cell_px + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_h*2.2, fig_h))
    fig.patch.set_facecolor(DARK)
    note = f" (top {n} by support)" if truncated else ""
    fig.suptitle(
        f"Stage 4 Revised  ·  Route {route}: {ROUTE_LABELS[route]}  "
        f"({ROUTE_MODELS[route]})\n"
        f"Confusion Matrix{note}  ·  Test Set  ·  "
        f"Accuracy = {overall['accuracy']*100:.2f}%  "
        f"({overall['n_correct']:,} / {overall['n_total']:,})",
        fontsize=12, fontweight="bold", color=WHITE, y=1.01)

    tick_labels = [c[:22] for c in classes]
    fs          = max(5, min(9, 120//n))

    for ax, (data, title, cmap, fmt) in zip(
            axes,
            [(cm,      "(a)  Raw Counts",         "Blues",  "d"),
             (cm_norm, "(b)  Normalised (Recall)", "YlOrRd", ".2f")]):
        ax.set_facecolor(PANEL)
        sns.heatmap(data, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    linewidths=0.4, linecolor=DARK,
                    cbar_kws={"shrink": 0.7},
                    annot_kws={"size": fs})
        ax.set_title(title, fontsize=11, fontweight="bold", color=WHITE, pad=8)
        ax.set_xlabel("Predicted",    color=WHITE, fontsize=9)
        ax.set_ylabel("Ground Truth", color=WHITE, fontsize=9)
        ax.tick_params(colors=WHITE, labelsize=max(5, fs-1))
        for tick in ax.get_xticklabels():
            tick.set_color(WHITE); tick.set_rotation(45); tick.set_ha("right")
        for tick in ax.get_yticklabels():
            tick.set_color(WHITE); tick.set_rotation(0)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=WHITE, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)

    # Highlight diagonal
    for ax in axes:
        for i in range(n):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1, fill=False,
                edgecolor=GREEN, linewidth=2.0))

    plt.tight_layout()
    path = os.path.join(OUT_DIR,
                        f"route{route}_{route_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅  Confusion matrix   → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE B — PER-CLASS METRICS CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_metrics(per_class_df, overall):
    route      = overall["route"]
    route_name = overall["route_name"]
    df         = per_class_df.copy()

    # Show top 30 classes by support for readability
    df = df.nlargest(30, "support").reset_index(drop=True)
    n  = len(df)
    x  = np.arange(n)
    labels = [c[:22] for c in df["class"].tolist()]

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.5, wspace=0.35,
                            top=0.88, bottom=0.06)
    fig.suptitle(
        f"Stage 4 Revised  ·  Route {route}: {ROUTE_LABELS[route]}  "
        f"({ROUTE_MODELS[route]})\n"
        f"Per-Class Metrics  ·  Test Set  ·  "
        f"Accuracy={overall['accuracy']*100:.2f}%  "
        f"Macro-F1={overall['macro_f1']:.4f}",
        fontsize=13, fontweight="bold", color=WHITE)

    # ── (a) F1 bars ───────────────────────────────────────────────────────────
    ax0 = dark_ax(fig.add_subplot(gs[0, :]))
    bar_colors = [GREEN if v >= 0.90 else TEAL if v >= 0.80
                  else AMBER if v >= 0.65 else RED
                  for v in df["f1"]]
    bars = ax0.bar(x, df["f1"], color=bar_colors, alpha=0.88,
                   edgecolor=DARK, linewidth=0.4, width=0.65)
    ax0.axhline(overall["macro_f1"],  color=WHITE, ls="--", lw=1.2,
                alpha=0.7, label=f"Macro-F1 = {overall['macro_f1']:.4f}")
    ax0.axhline(overall["accuracy"],  color=TEAL,  ls=":",  lw=1.5,
                alpha=0.8, label=f"Accuracy = {overall['accuracy']*100:.2f}%")
    ax0.axhline(0.90, color=GREEN, ls="-.", lw=0.8,
                alpha=0.5, label="90% target")
    ax0.set_title("(a)  F1-Score per Class (top 30 by support)",
                  fontsize=10, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax0.set_ylabel("F1-Score"); ax0.set_ylim(0, 1.15)
    ax0.grid(axis="y", alpha=0.12, color=GREY)
    for bar, val in zip(bars, df["f1"]):
        if val > 0:
            ax0.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", fontsize=6,
                     fontweight="bold", color=WHITE)
    # Legend
    patches = [
        mpatches.Patch(color=GREEN, label="F1 ≥ 90%"),
        mpatches.Patch(color=TEAL,  label="F1 ≥ 80%"),
        mpatches.Patch(color=AMBER, label="F1 ≥ 65%"),
        mpatches.Patch(color=RED,   label="F1 < 65%"),
    ]
    all_handles = patches + ax0.get_legend_handles_labels()[0]
    ax0.legend(handles=all_handles, fontsize=7.5, facecolor=PANEL,
               labelcolor=WHITE, edgecolor="#30363D",
               loc="upper right", ncol=2)

    # ── (b) Grouped P / R / F1 ────────────────────────────────────────────────
    ax1 = dark_ax(fig.add_subplot(gs[1, 0]))
    w   = 0.26
    ax1.bar(x - w, df["precision"], w, label="Precision",
            color=TEAL,  alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.bar(x,     df["recall"],    w, label="Recall",
            color=GREEN, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.bar(x + w, df["f1"],        w, label="F1",
            color=AMBER, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.set_title("(b)  Precision · Recall · F1",
                  fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=7.5, facecolor=PANEL,
               labelcolor=WHITE, edgecolor="#30363D")
    ax1.grid(axis="y", alpha=0.12, color=GREY)

    # ── (c) Support distribution ──────────────────────────────────────────────
    ax2 = dark_ax(fig.add_subplot(gs[1, 1]))
    support_colors = [ROUTE_COLORS.get(route, TEAL)] * n
    ax2.barh(range(n), df["support"][::-1],
             color=support_colors, alpha=0.85,
             edgecolor=DARK, linewidth=0.3)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels[::-1], fontsize=6)
    ax2.set_xlabel("Support (# samples)")
    ax2.set_title("(c)  Class Support", fontsize=9, fontweight="bold")
    ax2.grid(axis="x", alpha=0.12, color=GREY)
    for i, v in enumerate(df["support"][::-1]):
        ax2.text(v + 0.3, i, str(v), va="center",
                 fontsize=6, color=WHITE)

    plt.tight_layout()
    path = os.path.join(OUT_DIR,
                        f"route{route}_{route_name}_per_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅  Per-class metrics  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE C — TRAINING LOSS CURVE
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(route):
    route_name = ROUTE_NAMES[route]
    log_path   = os.path.join(
        PROJECT, "logs", "stage4_revised",
        f"route{route}_{route_name}_log.csv")

    if not os.path.exists(log_path):
        print(f"   ⚠️   No training log found for route {route} — skipping curve")
        return None

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f"Stage 4 Revised  ·  Route {route}: {ROUTE_LABELS[route]}  "
        f"({ROUTE_MODELS[route]})  —  Training Curves",
        fontsize=12, fontweight="bold", color=WHITE)

    # Loss curve
    ax = dark_ax(axes[0])
    ax.plot(df["epoch"], df["tr_loss"], color=TEAL,  lw=2,
            marker="o", ms=4, label="Train Loss")
    ax.plot(df["epoch"], df["va_loss"], color=AMBER, lw=2,
            marker="s", ms=4, label="Val Loss", ls="--")
    best_ep = df.loc[df["va_loss"].idxmin()]
    ax.axvline(best_ep["epoch"], color=GREEN, ls=":", lw=1.5,
               alpha=0.7, label=f"Best epoch {int(best_ep['epoch'])}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("(a)  Loss Curve", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE,
              edgecolor="#30363D")
    ax.grid(alpha=0.12, color=GREY)

    # Accuracy curve (if available)
    ax2 = dark_ax(axes[1])
    if "val_acc" in df.columns:
        ax2.plot(df["epoch"], df["val_acc"] * 100, color=GREEN, lw=2,
                 marker="o", ms=4, label="Val Accuracy")
        ax2.axhline(90, color=RED, ls="--", lw=1.5,
                    alpha=0.7, label="90% target")
        best_acc = df["val_acc"].max()
        ax2.set_title(
            f"(b)  Val Accuracy  (best={best_acc*100:.2f}%)",
            fontsize=10, fontweight="bold")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE,
                   edgecolor="#30363D")
    else:
        ax2.set_title("(b)  Val Accuracy  (not recorded)",
                      fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.grid(alpha=0.12, color=GREY)

    plt.tight_layout()
    path = os.path.join(OUT_DIR,
                        f"route{route}_{route_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅  Training curves    → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE D — SAMPLE PREDICTIONS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_predictions(preds, gts, overall, n=20):
    route      = overall["route"]
    route_name = overall["route_name"]

    rows = []
    for i in range(min(n, len(preds))):
        rows.append({
            "#"           : i + 1,
            "Ground Truth": gts[i][:35] + "…" if len(gts[i]) > 35 else gts[i],
            "Prediction"  : preds[i][:35] + "…" if len(preds[i]) > 35 else preds[i],
            "✓/✗"         : "✓" if preds[i] == gts[i] else "✗",
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, max(4, len(df) * 0.38 + 1.5)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.axis("off")
    fig.suptitle(
        f"Stage 4 Revised  ·  Route {route}: {ROUTE_LABELS[route]}  "
        f"— Sample Predictions (first {len(df)})",
        fontsize=12, fontweight="bold", color=WHITE)

    tbl = ax.table(
        cellText  = df.values.tolist(),
        colLabels = df.columns.tolist(),
        cellLoc   = "left", loc="center",
        bbox      = [0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363D")
        if row == 0:
            cell.set_facecolor("#0D419D")
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            is_correct = df.iloc[row-1]["✓/✗"] == "✓"
            if col == 3:
                cell.set_facecolor(GREEN if is_correct else RED)
                cell.set_text_props(color=DARK, fontweight="bold")
            else:
                cell.set_facecolor("#1C2128" if row % 2 == 0 else PANEL)
                cell.set_text_props(color=WHITE)

    plt.tight_layout()
    path = os.path.join(OUT_DIR,
                        f"route{route}_{route_name}_sample_preds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅  Sample predictions → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE E — ALL-ROUTES SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(all_results: dict):
    """
    all_results : {route_id: overall_dict}
    """
    if not all_results:
        print("   ⚠️   No results to plot summary")
        return None

    routes   = sorted(all_results.keys())
    labels   = [ROUTE_LABELS[r]  for r in routes]
    accs     = [all_results[r]["accuracy"]  * 100 for r in routes]
    f1s      = [all_results[r]["macro_f1"]        for r in routes]
    precs    = [all_results[r]["macro_prec"]       for r in routes]
    recs     = [all_results[r]["macro_rec"]        for r in routes]
    baseline = [BASELINE_MLP.get(r, 0.0) * 100    for r in routes]
    colors   = [ROUTE_COLORS[r] for r in routes]

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            top=0.90, bottom=0.06)
    fig.suptitle(
        "Stage 4 Revised  ·  DistilBERT + YOLO  —  Summary Dashboard\n"
        "Thesis: Advancing Medical AI with Explainable VQA on GI Imaging  "
        "|  Ummay Hani Javed (24i-8211)",
        fontsize=13, fontweight="bold", color=WHITE)

    x = np.arange(len(routes))
    w = 0.35

    # ── (a) Accuracy — revised vs baseline ────────────────────────────────────
    ax0 = dark_ax(fig.add_subplot(gs[0, :2]))
    bars1 = ax0.bar(x - w/2, accs,     w, color=colors,   alpha=0.90,
                    edgecolor=DARK, linewidth=0.4, label="DistilBERT+YOLO")
    bars2 = ax0.bar(x + w/2, baseline, w, color=GREY,      alpha=0.60,
                    edgecolor=DARK, linewidth=0.4, label="MLP Baseline")
    ax0.axhline(90, color=GREEN, ls="--", lw=1.5, alpha=0.7,
                label="90% Target")
    ax0.set_xticks(x); ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("Test Accuracy (%)"); ax0.set_ylim(0, 115)
    ax0.set_title("(a)  Test Accuracy — Revised vs MLP Baseline",
                  fontsize=10, fontweight="bold")
    ax0.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE,
               edgecolor="#30363D")
    ax0.grid(axis="y", alpha=0.12, color=GREY)
    for bar, val in zip(bars1, accs):
        ax0.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", fontsize=8,
                 fontweight="bold", color=WHITE)

    # ── (b) Macro F1 bar chart ─────────────────────────────────────────────────
    ax1 = dark_ax(fig.add_subplot(gs[0, 2]))
    bar_colors_f1 = [GREEN if v >= 0.90 else TEAL if v >= 0.80
                     else AMBER if v >= 0.65 else RED for v in f1s]
    bars = ax1.bar(x, f1s, color=bar_colors_f1, alpha=0.88,
                   edgecolor=DARK, linewidth=0.4)
    ax1.axhline(0.90, color=GREEN, ls="--", lw=1.2,
                alpha=0.7, label="0.90 target")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25,
                                            ha="right", fontsize=8)
    ax1.set_ylabel("Macro F1"); ax1.set_ylim(0, 1.12)
    ax1.set_title("(b)  Macro F1 per Route",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7.5, facecolor=PANEL, labelcolor=WHITE,
               edgecolor="#30363D")
    ax1.grid(axis="y", alpha=0.12, color=GREY)
    for bar, val in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", fontsize=7,
                 fontweight="bold", color=WHITE)

    # ── (c) Grouped P / R / F1 ────────────────────────────────────────────────
    ax2 = dark_ax(fig.add_subplot(gs[1, :2]))
    w2  = 0.25
    ax2.bar(x - w2, precs, w2, label="Precision",
            color=TEAL,  alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax2.bar(x,      recs,  w2, label="Recall",
            color=GREEN, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax2.bar(x + w2, f1s,   w2, label="F1",
            color=AMBER, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.12)
    ax2.set_title("(c)  Macro Precision · Recall · F1 per Route",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE,
               edgecolor="#30363D")
    ax2.grid(axis="y", alpha=0.12, color=GREY)

    # ── (d) Accuracy gain over baseline ───────────────────────────────────────
    ax3 = dark_ax(fig.add_subplot(gs[1, 2]))
    gains = [a - b for a, b in zip(accs, baseline)]
    gain_colors = [GREEN if g >= 0 else RED for g in gains]
    bars = ax3.bar(x, gains, color=gain_colors, alpha=0.88,
                   edgecolor=DARK, linewidth=0.4)
    ax3.axhline(0, color=WHITE, lw=0.8, alpha=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=25,
                                             ha="right", fontsize=8)
    ax3.set_ylabel("Accuracy Gain (pp)")
    ax3.set_title("(d)  Improvement over MLP Baseline",
                  fontsize=10, fontweight="bold")
    ax3.grid(axis="y", alpha=0.12, color=GREY)
    for bar, val in zip(bars, gains):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if val >= 0 else -1.2),
                 f"{val:+.1f}", ha="center", fontsize=8,
                 fontweight="bold", color=WHITE)

    # ── (e) Question type distribution ────────────────────────────────────────
    ax4 = dark_ax(fig.add_subplot(gs[2, 0]))
    sample_counts = [all_results[r]["n_total"] for r in routes]
    wedge_colors  = [ROUTE_COLORS[r] for r in routes]
    wedges, texts, autotexts = ax4.pie(
        sample_counts, labels=labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": WHITE, "fontsize": 7},
        pctdistance=0.82)
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color(DARK)
        at.set_fontweight("bold")
    ax4.set_title("(e)  Test Set Distribution\nby Question Type",
                  fontsize=9, fontweight="bold")

    # ── (f) Metrics summary table ──────────────────────────────────────────────
    ax5 = dark_ax(fig.add_subplot(gs[2, 1:]))
    ax5.axis("off")
    table_data = [
        [ROUTE_LABELS[r],
         ROUTE_MODELS[r],
         f"{all_results[r]['accuracy']*100:.2f}%",
         f"{all_results[r]['macro_f1']:.4f}",
         f"{all_results[r]['macro_prec']:.4f}",
         f"{all_results[r]['macro_rec']:.4f}",
         f"{all_results[r]['n_total']:,}",
         "✅" if all_results[r]['accuracy'] >= 0.90 else
         "⚠️" if all_results[r]['accuracy'] >= 0.85 else "❌"]
        for r in routes
    ]
    # Add weighted average row
    total_n  = sum(all_results[r]["n_total"] for r in routes)
    w_acc    = sum(all_results[r]["accuracy"] * all_results[r]["n_total"]
                   for r in routes) / max(total_n, 1)
    w_f1     = sum(all_results[r]["macro_f1"] * all_results[r]["n_total"]
                   for r in routes) / max(total_n, 1)
    table_data.append([
        "OVERALL (weighted)", "DistilBERT+YOLO",
        f"{w_acc*100:.2f}%", f"{w_f1:.4f}", "—", "—",
        f"{total_n:,}",
        "✅" if w_acc >= 0.90 else "⚠️"])

    col_labels = ["Question Type", "Model", "Accuracy",
                  "Macro F1", "Macro Prec", "Macro Rec",
                  "# Samples", "≥90%?"]
    tbl = ax5.table(
        cellText  = table_data,
        colLabels = col_labels,
        cellLoc   = "center", loc="center",
        bbox      = [0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363D")
        if row == 0:
            cell.set_facecolor("#0D419D")
            cell.set_text_props(color=WHITE, fontweight="bold")
        elif row == len(table_data):
            cell.set_facecolor("#1A3A5C")
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            r_idx = routes[row - 1]
            cell.set_facecolor("#1C2128" if row % 2 == 0 else PANEL)
            cell.set_text_props(color=WHITE)
            if col == 2:
                acc = all_results[r_idx]["accuracy"]
                color = (GREEN if acc >= 0.90 else
                         AMBER if acc >= 0.85 else RED)
                cell.set_facecolor(color)
                cell.set_text_props(color=DARK, fontweight="bold")

    ax5.set_title("(f)  Summary Metrics Table",
                  fontsize=9, fontweight="bold", color=WHITE, pad=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "stage4_revised_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✅  Summary dashboard    → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE F — TRAINING CURVES (ALL ROUTES TOGETHER)
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_training_curves():
    """Overlay training loss curves for all available routes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        "Stage 4 Revised  ·  Training Curves — All Routes",
        fontsize=13, fontweight="bold", color=WHITE)

    found = False
    for route in range(4):   # DistilBERT routes only
        route_name = ROUTE_NAMES[route]
        log_path   = os.path.join(
            PROJECT, "logs", "stage4_revised",
            f"route{route}_{route_name}_log.csv")
        if not os.path.exists(log_path):
            continue
        found = True
        df    = pd.read_csv(log_path)
        c     = ROUTE_COLORS[route]
        label = ROUTE_LABELS[route]

        ax0 = dark_ax(axes[0])
        ax0.plot(df["epoch"], df["tr_loss"], color=c, lw=1.8,
                 marker="o", ms=3, label=f"R{route} {label} (train)",
                 alpha=0.85)
        ax0.plot(df["epoch"], df["va_loss"], color=c, lw=1.8,
                 marker="s", ms=3, ls="--",
                 label=f"R{route} {label} (val)", alpha=0.6)

        if "val_acc" in df.columns:
            ax1 = dark_ax(axes[1])
            ax1.plot(df["epoch"], df["val_acc"] * 100, color=c, lw=1.8,
                     marker="o", ms=3, label=f"R{route} {label}",
                     alpha=0.85)

    if not found:
        print("   ⚠️   No training logs found for combined curve plot")
        plt.close()
        return None

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("(a)  Loss Curves (solid=train, dashed=val)",
                      fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE,
                   edgecolor="#30363D", ncol=2)
    axes[0].grid(alpha=0.12, color=GREY)

    axes[1].axhline(90, color=RED, ls="--", lw=1.5,
                    alpha=0.7, label="90% target")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("(b)  Validation Accuracy per Route",
                      fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE,
                   edgecolor="#30363D")
    axes[1].grid(alpha=0.12, color=GREY)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "stage4_revised_all_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅  All training curves  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURE G — ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_analysis(preds, gts, overall):
    """Shows what the model gets wrong most frequently."""
    route      = overall["route"]
    route_name = overall["route_name"]

    # Collect wrong predictions
    errors = [(g, p) for p, g in zip(preds, gts) if p != g]
    if not errors:
        print(f"   ℹ️   Route {route}: no errors — perfect accuracy!")
        return None

    # Most common errors: (gt, pred) pairs
    error_counts = Counter(errors).most_common(20)
    gt_labels    = [f"{g[:18]}→{p[:18]}" for (g, p), _ in error_counts]
    counts       = [c for _, c in error_counts]

    # Most confused GT classes
    gt_error_counts = Counter([g for g, p in errors]).most_common(15)
    gt_cls  = [g[:25] for g, _ in gt_error_counts]
    gt_errs = [c      for _, c in gt_error_counts]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f"Stage 4 Revised  ·  Route {route}: {ROUTE_LABELS[route]}  "
        f"— Error Analysis\n"
        f"Total errors: {len(errors):,} / {len(preds):,}  "
        f"({len(errors)/len(preds)*100:.1f}%)",
        fontsize=12, fontweight="bold", color=WHITE)

    # Top misclassifications
    ax0 = dark_ax(axes[0])
    ax0.barh(range(len(counts)), counts[::-1],
             color=RED, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax0.set_yticks(range(len(counts)))
    ax0.set_yticklabels(gt_labels[::-1], fontsize=7)
    ax0.set_xlabel("Error Count")
    ax0.set_title("(a)  Top 20 Misclassification Pairs (GT → Pred)",
                  fontsize=9, fontweight="bold")
    ax0.grid(axis="x", alpha=0.12, color=GREY)

    # Most confused classes
    ax1 = dark_ax(axes[1])
    ax1.barh(range(len(gt_errs)), gt_errs[::-1],
             color=AMBER, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.set_yticks(range(len(gt_errs)))
    ax1.set_yticklabels(gt_cls[::-1], fontsize=7)
    ax1.set_xlabel("Number of Wrong Predictions")
    ax1.set_title("(b)  Most Frequently Misclassified GT Classes",
                  fontsize=9, fontweight="bold")
    ax1.grid(axis="x", alpha=0.12, color=GREY)

    plt.tight_layout()
    path = os.path.join(OUT_DIR,
                        f"route{route}_{route_name}_error_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅  Error analysis     → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE SUMMARY CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_summary_csv(all_results):
    rows = []
    for r, res in sorted(all_results.items()):
        rows.append({
            "route"         : r,
            "question_type" : ROUTE_LABELS[r],
            "model"         : ROUTE_MODELS[r],
            "test_accuracy" : f"{res['accuracy']*100:.2f}%",
            "macro_f1"      : res["macro_f1"],
            "macro_precision": res["macro_prec"],
            "macro_recall"  : res["macro_rec"],
            "n_correct"     : res["n_correct"],
            "n_total"       : res["n_total"],
            "n_classes"     : res["n_classes"],
            "pass_90pct"    : "YES" if res["accuracy"] >= 0.90 else "NO",
            "baseline_acc"  : f"{BASELINE_MLP.get(r, 0)*100:.2f}%",
            "improvement_pp": f"{(res['accuracy'] - BASELINE_MLP.get(r,0))*100:+.2f}",
        })
    df   = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "stage4_revised_summary.csv")
    df.to_csv(path, index=False)
    print(f"✅  Summary CSV          → {path}")

    # Print to terminal
    print(f"\n{'='*70}")
    print(f"  STAGE 4 REVISED — RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Route':<4} {'Type':<16} {'Model':<12} {'Accuracy':>10} "
          f"{'Macro F1':>10} {'Pass?':>6}")
    print(f"  {'─'*4} {'─'*16} {'─'*12} {'─'*10} {'─'*10} {'─'*6}")
    for r in sorted(all_results.keys()):
        res = all_results[r]
        status = "✅" if res["accuracy"] >= 0.90 else \
                 "⚠️" if res["accuracy"] >= 0.85 else "❌"
        print(f"  {r:<4} {ROUTE_LABELS[r]:<16} {ROUTE_MODELS[r]:<12} "
              f"{res['accuracy']*100:>9.2f}% {res['macro_f1']:>10.4f} "
              f"{status:>6}")
    total_n = sum(all_results[r]["n_total"] for r in all_results)
    w_acc   = sum(all_results[r]["accuracy"] * all_results[r]["n_total"]
                  for r in all_results) / max(total_n, 1)
    print(f"  {'─'*62}")
    print(f"  {'OVERALL':<21} {'(weighted)':<12} {w_acc*100:>9.2f}%")
    print(f"{'='*70}\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 11. ANALYSE ONE ROUTE
# ─────────────────────────────────────────────────────────────────────────────
def analyse_route(route: int) -> dict:
    route_name = ROUTE_NAMES[route]
    print(f"\n{'='*60}")
    print(f"📊  Analysing Route {route}: {ROUTE_LABELS[route].upper()}")
    print(f"    Model: {ROUTE_MODELS[route]}")
    print(f"{'='*60}")

    preds, gts = load_or_run_predictions(route)
    if not preds:
        print(f"   ❌  No predictions available for route {route}")
        return {}

    overall, per_class_df = compute_metrics(preds, gts, route)
    if overall is None:
        return {}

    # Print per-route summary
    print(f"\n  Test Accuracy : {overall['accuracy']*100:.2f}%  "
          f"({overall['n_correct']:,} / {overall['n_total']:,})")
    print(f"  Macro F1      : {overall['macro_f1']:.4f}")
    print(f"  Macro Prec    : {overall['macro_prec']:.4f}")
    print(f"  Macro Recall  : {overall['macro_rec']:.4f}")
    print(f"  Num Classes   : {overall['n_classes']}")
    status = "✅ PASS" if overall["accuracy"] >= 0.90 else \
             "⚠️  CLOSE" if overall["accuracy"] >= 0.85 else "❌ BELOW TARGET"
    print(f"  Status        : {status} (target: 90%)")

    # Save per-class CSV
    csv_path = os.path.join(
        OUT_DIR, f"route{route}_{route_name}_per_class.csv")
    per_class_df.to_csv(csv_path, index=False)
    print(f"\n  Per-class CSV  → {csv_path}")

    # Generate figures
    print(f"\n  Generating figures ...")
    plot_confusion_matrix(preds, gts, overall)
    plot_per_class_metrics(per_class_df, overall)
    plot_training_curves(route)
    plot_sample_predictions(preds, gts, overall, n=20)
    plot_error_analysis(preds, gts, overall)

    return overall


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 Revised — Analysis & Evaluation Report")
    parser.add_argument("--route", default="all",
                        help="Route: 0-3, all, all_distilbert, summary_only")
    parser.add_argument("--summary_only", action="store_true",
                        help="Only generate summary dashboard from saved CSVs")
    args = parser.parse_args()

    all_results = {}

    if args.summary_only or args.route == "summary_only":
        # Load from existing CSVs only
        for route in range(6):
            route_name = ROUTE_NAMES[route]
            csv_path   = os.path.join(
                PROJECT, "logs", "stage4_revised",
                f"route{route}_{route_name}_eval.csv")
            if os.path.exists(csv_path):
                df   = pd.read_csv(csv_path)
                preds = df["prediction"].astype(str).tolist()
                gts   = df["ground_truth"].astype(str).tolist()
                overall, _ = compute_metrics(preds, gts, route)
                if overall:
                    all_results[route] = overall
                    print(f"   Loaded route {route}: "
                          f"{overall['accuracy']*100:.2f}%")
    else:
        # Determine which routes to run
        r = args.route.lower()
        if r == "all":
            routes = list(range(4))       # DistilBERT routes
        elif r == "all_distilbert":
            routes = [0, 1, 2, 3]
        else:
            routes = [int(r)]

        for route in routes:
            result = analyse_route(route)
            if result:
                all_results[route] = result

    # Combined plots + summary (only if we have ≥2 routes)
    if len(all_results) >= 2:
        print(f"\n{'='*60}")
        print(f"📊  Generating summary for {len(all_results)} routes ...")
        print(f"{'='*60}")
        plot_summary_dashboard(all_results)
        plot_all_training_curves()
        save_summary_csv(all_results)
    elif len(all_results) == 1:
        route = list(all_results.keys())[0]
        save_summary_csv(all_results)
        print(f"\n  (Run with --route all to generate combined dashboard)")

    print(f"\n✅  All analysis saved → {OUT_DIR}\n")


if __name__ == "__main__":
    main()