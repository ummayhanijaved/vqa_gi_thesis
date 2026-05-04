"""
=============================================================================
STAGE 4 — Evaluation Report
Generates confusion matrix, accuracy table, and per-class metrics figures
for any trained transformer head.

USAGE:
    python stage4_eval_report.py --route 0          # Yes/No (FLAN-T5)
    python stage4_eval_report.py --route 1          # Single-Choice (BART)
    python stage4_eval_report.py --route 2          # Multi-Choice (DeBERTa)
    python stage4_eval_report.py --route 3          # Color (ViT-GPT2)
    python stage4_eval_report.py --route 4          # Location (BLIP2)
    python stage4_eval_report.py --route 5          # Count (T5-Large)

Outputs (saved to ~/vqa_gi_thesis/logs/stage4_transformers/):
    route0_yes_no_confusion_matrix.png
    route0_yes_no_metrics_table.png
    route0_yes_no_results.csv
=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # non-interactive — no display server needed, no memory crash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import DataLoader

HOME    = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "vqa_gi_thesis")
sys.path.insert(0, os.path.join(PROJECT, "src"))
sys.path.insert(0, HOME)

# ─────────────────────────────────────────────────────────────────────────────
ROUTE_NAMES = {
    0: "yes_no", 1: "single_choice", 2: "multi_choice",
    3: "color",  4: "location",      5: "count",
}
ROUTE_MODELS = {
    0: "FLAN-T5-base",  1: "BART-Large",     2: "DeBERTa-v3",
    3: "ViT-GPT2",      4: "BLIP2-FLAN-XL",  5: "T5-Large",
}
ROUTE_TO_KEY = {0:"yn", 1:"single", 2:"multi", 3:"color", 4:"loc", 5:"count"}

DARK   = "#0D1117"
PANEL  = "#161B22"
WHITE  = "#E6EDF3"
GREY   = "#8B949E"
TEAL   = "#00B4D8"
GREEN  = "#2ECC71"
RED    = "#E74C3C"
AMBER  = "#F39C12"
PURPLE = "#9B59B6"


# ─────────────────────────────────────────────────────────────────────────────
# 1. RUN PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def run_predictions(route: int):
    """
    Load trained head from checkpoint, run on test cache, collect all
    predictions and ground-truth labels. Returns list of (pred, gt) strings.
    """
    # Import pipeline components
    from stage4_transformers import (
        CFG, S3_CFG, ROUTE_NAMES, ROUTE_TO_KEY,
        Stage4TransformerGenerator, Stage4TransformerDataset,
        cache_stage3_features, collate_fn_for_route,
        FusionExtractor, TextPreprocessor,
    )
    from preprocessing import build_image_transform
    from datasets import load_from_disk

    route_name = ROUTE_NAMES[route]
    head_key   = ROUTE_TO_KEY[route]

    # Fix relative paths
    S3_CFG["stage1_ckpt"]    = os.path.join(PROJECT, "checkpoints", "stage1_best.pt")
    S3_CFG["stage2_ckpt"]    = os.path.join(PROJECT, "checkpoints", "best_model")
    S3_CFG["checkpoint_dir"] = os.path.join(PROJECT, "checkpoints")
    S3_CFG["log_dir"]        = os.path.join(PROJECT, "logs")
    S3_CFG["data_dir"]       = os.path.join(HOME, "data", "kvasir_local")

    print(f"\n🔍  Loading Stage 3 extractor ...")
    extractor = FusionExtractor(CFG["stage3_ckpt"])
    text_prep = TextPreprocessor()
    raw       = load_from_disk(CFG["data_dir"])

    # Test cache
    cache_dir = os.path.join(PROJECT, "cache", "stage3_features")
    te_cache  = cache_stage3_features(
        extractor, text_prep, raw["test"], "test", cache_dir)

    # Load vocab
    vocab_path = CFG.get("vocab_file",
                         os.path.join(HOME, "data", "stage4_vocab.json"))
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
    else:
        vocab = {"yn": ["no","yes"], "multi": [], "color": [], "loc": [], "count": []}

    te_set    = Stage4TransformerDataset(te_cache, "test", vocab, target_route=route)
    te_loader = DataLoader(te_set, batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=0,
                           collate_fn=lambda b: b)

    print(f"   Test samples for route {route} ({route_name}): {len(te_set):,}")

    # Load model
    ckpt_path = os.path.join(
        CFG["ckpt_dir"], f"stage4_{route_name}_best.pt")
    assert os.path.exists(ckpt_path), \
        f"Checkpoint not found: {ckpt_path}\nRun: python stage4_transformers.py --mode train --route {route}"

    print(f"   Loading checkpoint: {ckpt_path}")
    model = Stage4TransformerGenerator(vocab, routes=[route])
    ckpt  = torch.load(ckpt_path, map_location=CFG["device"], weights_only=False)
    model.heads[head_key].load_state_dict(ckpt["model_state"])
    model     = model.to(CFG["device"])
    head      = model.heads[head_key]
    tokenizer = model.tokenizers[head_key]
    head.eval()

    print(f"   Running inference on {len(te_set):,} test samples ...")
    all_preds, all_gts = [], []

    with torch.no_grad():
        for raw_batch in tqdm(te_loader, desc="  Predicting", leave=False):
            batch = collate_fn_for_route(
                raw_batch, route, tokenizer, vocab, CFG["device"])
            if batch is None:
                continue

            if route == 2:
                preds = head.predict(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"],
                    vocab.get("multi", []))
            elif route == 3:
                preds = head.generate(batch["fused"], batch["disease"])
            else:
                preds = head.generate(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"])

            for pred, gt in zip(preds, batch["answers_raw"]):
                all_preds.append(pred.strip().lower())
                all_gts.append(gt.strip().lower())

    print(f"   ✅  Predictions complete: {len(all_preds):,} samples")
    return all_preds, all_gts, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 0b. ANSWER NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def normalise_answers(preds, gts, route):
    """
    Route 0 (yes/no): GT answers are verbose strings like
        "no anatomical landmarks identified" → normalise to "no"
        "yes, a polyp is present"            → normalise to "yes"
    The constrained decoder always outputs bare "yes"/"no", so we
    must reduce GT to the same space before computing metrics.

    Other routes: return unchanged.
    """
    if route != 0:
        return preds, gts

    def _yn(s):
        s = s.strip().lower()
        if s in ("yes", "no"):
            return s
        if s.startswith("yes"):
            return "yes"
        if s.startswith("no"):
            return "no"
        # Fallback: look for yes/no anywhere in short answers
        words = s.split()
        for w in words[:3]:
            w = w.strip(".,;:")
            if w == "yes":
                return "yes"
            if w == "no":
                return "no"
        return s   # unknown — keep as-is (will count as wrong)

    gts_norm   = [_yn(g) for g in gts]
    preds_norm = [p.strip().lower() for p in preds]   # already "yes"/"no"
    return preds_norm, gts_norm

# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds, gts, route):
    """
    Compute exact-match accuracy + per-class precision/recall/F1.
    Returns overall dict and per-class DataFrame.
    """
    route_name = ROUTE_NAMES[route]
    correct    = [int(p == g) for p, g in zip(preds, gts)]
    accuracy   = sum(correct) / max(len(correct), 1)

    # Get unique classes present in GT
    classes = sorted(set(gts))

    rows = []
    for cls in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, gts) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, gts) if p != cls and g == cls)
        tn = sum(1 for p, g in zip(preds, gts) if p != cls and g != cls)
        support = sum(1 for g in gts if g == cls)

        prec  = tp / (tp + fp + 1e-9)
        rec   = tp / (tp + fn + 1e-9)
        f1    = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({
            "class"    : cls,
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "f1"       : round(f1,   4),
            "support"  : support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    df = pd.DataFrame(rows)

    overall = {
        "route"      : route,
        "route_name" : route_name,
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
# 3. CONFUSION MATRIX FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(preds, gts, overall, out_dir):
    route      = overall["route"]
    route_name = overall["route_name"]
    model_name = overall["model"]
    accuracy   = overall["accuracy"]

    # Cap to top-30 classes by support so the figure stays readable
    MAX_CLASSES = 30
    from collections import Counter
    gt_counts = Counter(gts)
    top_classes = [c for c, _ in gt_counts.most_common(MAX_CLASSES)]
    # Filter preds/gts to only these classes
    pairs = [(p, g) for p, g in zip(preds, gts) if g in top_classes]
    if not pairs:
        print("⚠️  No samples for top classes — skipping confusion matrix.")
        return None
    preds_f, gts_f = zip(*pairs)
    classes   = sorted(set(gts_f))
    n         = len(classes)
    cls_idx   = {c: i for i, c in enumerate(classes)}
    truncated = len(set(gts)) > MAX_CLASSES
    title_note = f" (top {n} classes by support)" if truncated else ""

    # Build confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for p, g in zip(preds_f, gts_f):
        gi = cls_idx.get(g)
        pi = cls_idx.get(p)
        if gi is not None:
            if pi is not None:
                cm[gi, pi] += 1
            # else: predicted unknown class — row stays unaccounted

    cm_norm  = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums,
                         out=np.zeros_like(cm_norm),
                         where=row_sums != 0)

    # ── Layout ────────────────────────────────────────────────────────────
    cell_px = max(0.55, min(1.0, 16 / n))   # scale cell size to class count
    fig_h   = max(7, n * cell_px + 2.5)
    fig_w   = max(8, n * cell_px + 3.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f"Stage 4 · Route {route}: {route_name.replace('_',' ').title()}  "
        f"({model_name})\n"
        f"Confusion Matrix{title_note}  ·  Test Set  ·  "
        f"Accuracy = {accuracy*100:.2f}%  ({overall['n_correct']:,}/{overall['n_total']:,})",
        fontsize=12, fontweight="bold", color=WHITE, y=1.01)

    tick_labels = [c[:22] for c in classes]
    font_size   = max(5, min(9, 120 // n))

    for ax, (data, title, cmap, fmt) in zip(
            axes,
            [(cm,      "(a)  Raw Counts",          "Blues",  "d"),
             (cm_norm, "(b)  Normalised (Recall)",  "YlOrRd", ".2f")]):

        ax.set_facecolor(PANEL)
        sns.heatmap(
            data, ax=ax,
            annot=True, fmt=fmt,
            cmap=cmap,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            linewidths=0.4, linecolor=DARK,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": font_size},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", color=WHITE, pad=10)
        ax.set_xlabel("Predicted", color=WHITE, fontsize=9)
        ax.set_ylabel("Ground Truth", color=WHITE, fontsize=9)
        ax.tick_params(colors=WHITE, labelsize=max(5, font_size - 1))
        for tick in ax.get_xticklabels():
            tick.set_color(WHITE); tick.set_rotation(45); tick.set_ha("right")
        for tick in ax.get_yticklabels():
            tick.set_color(WHITE); tick.set_rotation(0)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=WHITE, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)

    # Diagonal highlight — correct predictions
    for ax in axes:
        for i in range(n):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor=GREEN, linewidth=2.0))

    plt.tight_layout()
    path = os.path.join(out_dir, f"route{route}_{route_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅  Confusion matrix saved → {path}")
    return path



# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS TABLE + BAR CHART FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_table(per_class_df, overall, out_dir):
    route      = overall["route"]
    route_name = overall["route_name"]
    model_name = overall["model"]
    df         = per_class_df.copy()

    n_classes  = len(df)
    fig_h      = max(10, n_classes * 0.55 + 5)

    fig = plt.figure(figsize=(20, fig_h))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.35,
                            top=0.90, bottom=0.05)

    fig.suptitle(
        f"Stage 4 · Route {route}: {route_name.replace('_',' ').title()}  "
        f"({model_name})\n"
        f"Per-Class Metrics  ·  Test Set",
        fontsize=14, fontweight="bold", color=WHITE)

    def dark_ax(ax):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        return ax

    x      = np.arange(n_classes)
    labels = [c[:20] for c in df["class"].tolist()]

    # ── (a) F1 bar chart ──────────────────────────────────────────────────
    ax0 = dark_ax(fig.add_subplot(gs[0, :]))
    bar_colors = [GREEN if v >= 0.90 else TEAL if v >= 0.80
                  else AMBER if v >= 0.65 else RED
                  for v in df["f1"]]
    bars = ax0.bar(x, df["f1"], color=bar_colors, alpha=0.85,
                   edgecolor=DARK, linewidth=0.4, width=0.65)
    ax0.axhline(overall["macro_f1"], color=WHITE, ls="--", lw=1.2, alpha=0.7,
                label=f"Macro-F1 = {overall['macro_f1']:.4f}")
    ax0.axhline(overall["accuracy"], color=TEAL,  ls=":", lw=1.5, alpha=0.8,
                label=f"Accuracy = {overall['accuracy']*100:.2f}%")
    ax0.set_title(f"(a)  F1-Score per Class", fontsize=10, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax0.set_ylabel("F1-Score")
    ax0.set_ylim(0, 1.12)
    leg = ax0.legend(fontsize=9, facecolor=PANEL, labelcolor=WHITE,
                     edgecolor="#30363D")
    ax0.grid(axis="y", alpha=0.12, color=GREY)
    for bar, val in zip(bars, df["f1"]):
        ax0.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=7,
                 fontweight="bold", color=WHITE)

    # Legend: colour coding
    patches = [
        mpatches.Patch(color=GREEN, label="≥ 90%"),
        mpatches.Patch(color=TEAL,  label="≥ 80%"),
        mpatches.Patch(color=AMBER, label="≥ 65%"),
        mpatches.Patch(color=RED,   label="< 65%"),
    ]
    ax0.legend(handles=patches + leg.legend_handles,
               fontsize=7.5, facecolor=PANEL, labelcolor=WHITE,
               edgecolor="#30363D", loc="upper right", ncol=2)

    # ── (b) Grouped P/R/F1 ────────────────────────────────────────────────
    ax1 = dark_ax(fig.add_subplot(gs[1, 0]))
    w   = 0.26
    ax1.bar(x - w, df["precision"], w, label="Precision",
            color=TEAL,  alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.bar(x,     df["recall"],    w, label="Recall",
            color=GREEN, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.bar(x + w, df["f1"],        w, label="F1-Score",
            color=AMBER, alpha=0.85, edgecolor=DARK, linewidth=0.3)
    ax1.set_title("(b)  Precision · Recall · F1 per Class",
                  fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=7.5, facecolor=PANEL, labelcolor=WHITE,
               edgecolor="#30363D")
    ax1.grid(axis="y", alpha=0.12, color=GREY)

    # ── (c) Numeric summary table ─────────────────────────────────────────
    ax2 = dark_ax(fig.add_subplot(gs[1, 1]))
    ax2.axis("off")

    # Overall metrics summary card
    summary = [
        ["Metric", "Value"],
        ["Test Accuracy",     f"{overall['accuracy']*100:.2f}%"],
        ["Correct / Total",   f"{overall['n_correct']:,} / {overall['n_total']:,}"],
        ["Macro F1",          f"{overall['macro_f1']:.4f}"],
        ["Macro Precision",   f"{overall['macro_prec']:.4f}"],
        ["Macro Recall",      f"{overall['macro_rec']:.4f}"],
        ["Num Classes",       str(overall["n_classes"])],
        ["Route",             f"{overall['route']} — {route_name.replace('_',' ')}"],
        ["Model",             model_name],
    ]

    tbl = ax2.table(
        cellText  = summary[1:],
        colLabels = summary[0],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1C2128" if row % 2 == 0 else PANEL)
        cell.set_edgecolor("#30363D")
        cell.set_text_props(color=WHITE)
        if row == 0:
            cell.set_facecolor("#0D419D")
            cell.set_text_props(color=WHITE, fontweight="bold")
        if col == 1 and row > 0:
            # Highlight accuracy row
            if "Accuracy" in summary[row][0]:
                acc_val = overall["accuracy"]
                color   = GREEN if acc_val >= 0.90 else AMBER if acc_val >= 0.80 else RED
                cell.set_facecolor(color)
                cell.set_text_props(color=DARK, fontweight="bold")

    ax2.set_title("(c)  Overall Test Metrics",
                  fontsize=9, fontweight="bold", color=WHITE)

    plt.tight_layout()
    path = os.path.join(out_dir, f"route{route}_{route_name}_metrics_table.png")
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅  Metrics table saved   → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAMPLE PREDICTIONS TABLE FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_predictions(preds, gts, questions, overall, out_dir, n=20):
    """Show first n predictions as a readable table."""
    route      = overall["route"]
    route_name = overall["route_name"]

    rows = []
    for i in range(min(n, len(preds))):
        rows.append({
            "#"      : i + 1,
            "Question"      : questions[i][:55] + "…" if len(questions[i]) > 55 else questions[i],
            "Ground Truth"  : gts[i][:20],
            "Prediction"    : preds[i][:20],
            "✓/✗"           : "✓" if preds[i] == gts[i] else "✗",
        })
    df = pd.DataFrame(rows)

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(18, max(4, n_rows * 0.38 + 1.5)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.axis("off")
    fig.suptitle(
        f"Stage 4 · Route {route}: {route_name.replace('_',' ').title()}  "
        f"— Sample Predictions (first {n_rows})",
        fontsize=12, fontweight="bold", color=WHITE)

    cell_text  = df.values.tolist()
    col_labels = df.columns.tolist()

    tbl = ax.table(
        cellText  = cell_text,
        colLabels = col_labels,
        cellLoc   = "left",
        loc       = "center",
        bbox      = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    col_widths = [0.04, 0.45, 0.16, 0.16, 0.06]
    for col_i, w in enumerate(col_widths):
        tbl.auto_set_column_width(col_i)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363D")
        if row == 0:
            cell.set_facecolor("#0D419D")
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            is_correct = cell_text[row-1][4] == "✓"
            if col == 4:
                cell.set_facecolor(GREEN if is_correct else RED)
                cell.set_text_props(color=DARK, fontweight="bold")
            else:
                cell.set_facecolor("#1C2128" if row % 2 == 0 else PANEL)
                cell.set_text_props(color=WHITE)

    plt.tight_layout()
    path = os.path.join(out_dir, f"route{route}_{route_name}_sample_preds.png")
    plt.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅  Sample preds saved    → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(route: int):
    route_name = ROUTE_NAMES[route]
    out_dir    = os.path.join(PROJECT, "logs", "stage4_transformers")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"📊  Stage 4 Evaluation Report — Route {route}: {route_name.upper()}")
    print(f"    Model: {ROUTE_MODELS[route]}")
    print(f"{'='*65}")

    # Step 1 — predictions
    preds, gts, vocab = run_predictions(route)

    # Collect questions from cache for sample table
    cache_path = os.path.join(PROJECT, "cache", "stage3_features",
                              "stage3_cache_test.pt")
    if os.path.exists(cache_path):
        records   = torch.load(cache_path, weights_only=False)
        questions = [r["question"] for r in records
                     if r["route"] == route]
    else:
        questions = ["N/A"] * len(preds)

    # Step 2 — normalise answers then compute metrics
    preds, gts = normalise_answers(preds, gts, route)
    overall, per_class_df = compute_metrics(preds, gts, route)

    # Print summary
    print(f"\n{'─'*50}")
    print(f"  Route {route}  ({route_name})  —  {ROUTE_MODELS[route]}")
    print(f"{'─'*50}")
    print(f"  Test Accuracy  : {overall['accuracy']*100:.2f}%"
          f"  ({overall['n_correct']:,} / {overall['n_total']:,})")
    print(f"  Macro F1       : {overall['macro_f1']:.4f}")
    print(f"  Macro Precision: {overall['macro_prec']:.4f}")
    print(f"  Macro Recall   : {overall['macro_rec']:.4f}")
    print(f"  Classes        : {overall['n_classes']}")
    print(f"{'─'*50}")
    print(f"\n  Per-class breakdown:")
    print(per_class_df[["class","precision","recall","f1","support"]].to_string(index=False))

    # Step 3 — save CSV
    csv_path = os.path.join(out_dir, f"route{route}_{route_name}_results.csv")
    per_class_df.to_csv(csv_path, index=False)
    print(f"\n✅  Results CSV saved     → {csv_path}")

    # Step 4 — figures
    plot_confusion_matrix(preds, gts, overall, out_dir)
    plot_metrics_table(per_class_df, overall, out_dir)
    plot_sample_predictions(preds, gts, questions[:len(preds)], overall, out_dir, n=20)

    # Final verdict
    acc = overall["accuracy"]
    print(f"\n{'='*50}")
    if acc >= 0.90:
        print(f"  ✅  PASS  —  {acc*100:.2f}% ≥ 90% target")
    elif acc >= 0.85:
        print(f"  ⚠️   CLOSE — {acc*100:.2f}% (target: 90%)")
    else:
        print(f"  ❌  BELOW target — {acc*100:.2f}% (need ≥ 90%)")
    print(f"{'='*50}\n")

    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=int, required=True,
                        choices=[0,1,2,3,4,5],
                        help="Route to evaluate (0-5)")
    args = parser.parse_args()
    main(args.route)
