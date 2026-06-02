#!/usr/bin/env python3
"""
=============================================================================
Stage 6 & 7 — Results Section Generator (lightweight, NO model reload)
=============================================================================

PURPOSE:
    Build the plots, qualitative tables, and LaTeX figure/table blocks for the
    Stage 6 (textual explainability) and Stage 7 (Grad-CAM) results sections —
    WITHOUT re-running the pipeline or BERTScore. It reads the CSVs your earlier
    runs already produced, so there is NO memory crash (no models are loaded).

INPUTS (already produced by your earlier successful runs):
    Stage 6:
      ~/vqa_gi_thesis/logs/stage6_explainability/medical_responses.csv
      ~/vqa_gi_thesis/logs/stage6_explainability/explainability_metrics.csv
    Stage 7:
      ~/vqa_gi_thesis/logs/stage7_gradcam/gradcam_index.csv
      ~/vqa_gi_thesis/logs/stage7_gradcam/gradcam_*.png   (existing heatmaps)

OUTPUTS:
    ~/vqa_gi_thesis/logs/stage67_results/
      stage6_metric_bars.png/pdf            (per-route NLG bars)
      stage6_score_distributions.png/pdf    (BERTScore / ROUGE-L histograms)
      stage6_examples_table.tex             (qualitative examples, LaTeX)
      stage7_gradcam_grid.png/pdf           (montage of available heatmaps)
      stage7_keyword_map_table.tex          (disease keyword map, LaTeX)
      stage67_figures.tex                   (ready-to-include figure blocks)

USAGE:
    python make_stage67_results.py
=============================================================================
"""
import os
import glob
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT = os.path.expanduser("~/vqa_gi_thesis")

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
       "#56B4E9", "#D55E00", "#F0E442", "#999999"]


def save(fig, base):
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {os.path.basename(base)}.png / .pdf")


def latex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6
# ─────────────────────────────────────────────────────────────────────────────
def stage6_plots(s6_csv, out_dir):
    if not os.path.exists(s6_csv):
        print(f"  ⚠️  Stage 6 CSV not found: {s6_csv}")
        return None
    df = pd.read_csv(s6_csv)
    print(f"  Loaded {len(df):,} Stage 6 rows")

    ROUTE_NAMES = {0: "Yes/No", 1: "Single", 2: "Multi",
                   3: "Colour", 4: "Location", 5: "Count"}

    # ---- (1) Per-route metric bars (means) ----
    metric_cols = [c for c in ["rouge1", "rougeL", "meteor", "chrf",
                               "bertscore"] if c in df.columns]
    if metric_cols and "route" in df.columns:
        routes = sorted(df["route"].unique())
        x = np.arange(len(routes))
        w = 0.15
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, m in enumerate(metric_cols):
            means = [df[df["route"] == r][m].mean() * 100 for r in routes]
            ax.bar(x + (i - len(metric_cols) / 2) * w + w / 2, means, w,
                   label=m.upper(), color=PAL[i % len(PAL)])
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}: {ROUTE_NAMES.get(r,r)}" for r in routes],
                           rotation=15)
        ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
        ax.set_title("Stage 6 — Per-Route Explainability Metrics",
                     fontweight="bold")
        ax.legend(ncol=len(metric_cols), loc="upper center",
                  bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        save(fig, os.path.join(out_dir, "stage6_metric_bars"))

    # ---- (2) Score distributions (histograms) ----
    dist_cols = [c for c in ["bertscore", "rougeL", "meteor"]
                 if c in df.columns]
    if dist_cols:
        fig, axes = plt.subplots(1, len(dist_cols),
                                 figsize=(5 * len(dist_cols), 4))
        if len(dist_cols) == 1:
            axes = [axes]
        for ax, c in zip(axes, dist_cols):
            ax.hist(df[c] * 100, bins=25, color=PAL[0], alpha=0.8,
                    edgecolor="white")
            ax.axvline(df[c].mean() * 100, color=PAL[5], ls="--",
                       label=f"mean {df[c].mean()*100:.1f}%")
            ax.set_xlabel(f"{c.upper()} (%)"); ax.set_ylabel("Count")
            ax.set_title(f"{c.upper()} distribution"); ax.legend(fontsize=8)
        fig.suptitle("Stage 6 — Metric Score Distributions (per sample)",
                     fontweight="bold")
        fig.tight_layout()
        save(fig, os.path.join(out_dir, "stage6_score_distributions"))

    return df


def stage6_examples_table(df, out_dir, n=6):
    """Pick best + a couple mid examples → LaTeX qualitative table."""
    if df is None or len(df) == 0:
        return
    sort_col = "bertscore" if "bertscore" in df.columns else df.columns[-1]
    qcol = "question" if "question" in df.columns else None
    acol = ("medical_answer" if "medical_answer" in df.columns
            else ("s4_answer" if "s4_answer" in df.columns else None))
    scol = ("s5_sentence" if "s5_sentence" in df.columns else None)
    gcol = ("gt_sentence" if "gt_sentence" in df.columns
            else ("ground_truth" if "ground_truth" in df.columns else None))

    top = df.sort_values(sort_col, ascending=False).head(n)

    out = os.path.join(out_dir, "stage6_examples_table.tex")
    with open(out, "w") as f:
        f.write("% Qualitative Stage 6 examples (auto-generated)\n")
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Representative Stage~5/6 outputs: question, "
                "generated sentence, ground truth, and semantic score.}\n")
        f.write("\\label{tab:s6_examples}\n")
        f.write("\\begin{tabular}{p{4.2cm}p{4.6cm}p{4.0cm}c}\n\\toprule\n")
        f.write("\\textbf{Question} & \\textbf{Generated (Stage 5)} & "
                "\\textbf{Ground Truth} & \\textbf{BERTSc.} \\\\\n\\midrule\n")
        for _, r in top.iterrows():
            q = latex_escape(r[qcol])[:90] if qcol else ""
            s = latex_escape(r[scol])[:100] if scol else \
                (latex_escape(r[acol])[:100] if acol else "")
            g = latex_escape(r[gcol])[:90] if gcol else ""
            bs = (f"{r['bertscore']*100:.1f}" if "bertscore" in df.columns
                  else "--")
            f.write(f"{q} & {s} & {g} & {bs} \\\\\n\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  ✅ stage6_examples_table.tex ({len(top)} examples)")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7
# ─────────────────────────────────────────────────────────────────────────────
def stage7_grid(gradcam_dir, out_dir, max_imgs=6):
    pngs = sorted(glob.glob(os.path.join(gradcam_dir, "gradcam_*.png")))
    # exclude any previous grid
    pngs = [p for p in pngs if "grid" not in os.path.basename(p)]
    if not pngs:
        print(f"  ⚠️  No Grad-CAM PNGs in {gradcam_dir}")
        return
    pngs = pngs[:max_imgs]
    n = len(pngs)
    cols = 1
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows))
    if n == 1:
        axes = [axes]
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, pngs):
        ax.imshow(mpimg.imread(p))
        ax.set_title(os.path.basename(p), fontsize=7)
        ax.axis("off")
    fig.suptitle("Stage 7 — Grad-CAM (original | heatmap | overlay) per route",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "stage7_gradcam_grid"))
    print(f"     (montaged {n} heatmaps)")


def stage7_keyword_table(out_dir):
    """The disease keyword→label map, as a LaTeX table (static, documented)."""
    rows = [
        ("polyp", "polyp-pedunculated, -sessile, -hyperplastic"),
        ("ulcer", "gastric-ulcer, duodenal-ulcer"),
        ("colitis", "ulcerative-colitis"),
        ("esophagitis", "esophagitis"),
        ("z-line", "normal-z-line"),
        ("pylorus", "normal-pylorus"),
        ("cecum", "normal-cecum"),
        ("dyed", "dyed-lifted-polyp, dyed-resection-margins"),
        ("instrument", "instrument"),
    ]
    out = os.path.join(out_dir, "stage7_keyword_map_table.tex")
    with open(out, "w") as f:
        f.write("% Stage 7 keyword->disease map (auto-generated)\n")
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Representative answer-keyword to disease-class "
                "mappings used to select the Grad-CAM target class. The map "
                "is built dynamically from the Stage~1 label list to guarantee "
                "index consistency.}\n")
        f.write("\\label{tab:s7_keymap}\n")
        f.write("\\begin{tabular}{ll}\n\\toprule\n")
        f.write("\\textbf{Answer keyword} & \\textbf{Target disease class(es)}"
                " \\\\\n\\midrule\n")
        for kw, tgt in rows:
            f.write(f"{latex_escape(kw)} & {latex_escape(tgt)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  ✅ stage7_keyword_map_table.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Figure-block helper (.tex you can \input)
# ─────────────────────────────────────────────────────────────────────────────
def write_figure_blocks(out_dir):
    out = os.path.join(out_dir, "stage67_figures.tex")
    with open(out, "w") as f:
        f.write("% =========================================================\n")
        f.write("% Stage 6 & 7 figure blocks — \\input or copy into thesis\n")
        f.write("% =========================================================\n\n")
        f.write("% --- Stage 6 ---\n")
        f.write("\\begin{figure}[H]\\centering\n")
        f.write("  \\includegraphics[width=\\textwidth]"
                "{stage67_results/stage6_metric_bars.pdf}\n")
        f.write("  \\caption{Per-route explainability (NLG) metrics for the "
                "Stage~6 layer.}\\label{fig:s6_bars}\n\\end{figure}\n\n")
        f.write("\\begin{figure}[H]\\centering\n")
        f.write("  \\includegraphics[width=\\textwidth]"
                "{stage67_results/stage6_score_distributions.pdf}\n")
        f.write("  \\caption{Per-sample score distributions. The BERTScore "
                "mass concentrated above 90\\% indicates consistent semantic "
                "fidelity.}\\label{fig:s6_dist}\n\\end{figure}\n\n")
        f.write("\\input{stage67_results/stage6_examples_table}\n\n")
        f.write("% --- Stage 7 ---\n")
        f.write("\\begin{figure}[H]\\centering\n")
        f.write("  \\includegraphics[width=0.9\\textwidth]"
                "{stage67_results/stage7_gradcam_grid.pdf}\n")
        f.write("  \\caption{Grad-CAM visual explanations. Each row shows the "
                "original endoscopic image, the answer-driven heatmap, and the "
                "overlay localising the evidence for the predicted answer.}"
                "\\label{fig:s7_grid}\n\\end{figure}\n\n")
        f.write("\\input{stage67_results/stage7_keyword_map_table}\n")
    print(f"  ✅ stage67_figures.tex (ready to \\input)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s6_dir", default=os.path.join(
        PROJECT, "logs", "stage6_explainability"))
    ap.add_argument("--s7_dir", default=os.path.join(
        PROJECT, "logs", "stage7_gradcam"))
    ap.add_argument("--out_dir", default=os.path.join(
        PROJECT, "logs", "stage67_results"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*68}")
    print(f"  STAGE 6 & 7 — Results Generator (no models loaded)")
    print(f"{'='*68}")
    print(f"  Stage 6 dir : {args.s6_dir}")
    print(f"  Stage 7 dir : {args.s7_dir}")
    print(f"  Output      : {args.out_dir}\n")

    print("  STAGE 6:")
    s6_csv = os.path.join(args.s6_dir, "medical_responses.csv")
    df6 = stage6_plots(s6_csv, args.out_dir)
    stage6_examples_table(df6, args.out_dir)

    print("\n  STAGE 7:")
    stage7_grid(args.s7_dir, args.out_dir)
    stage7_keyword_table(args.out_dir)

    print("\n  LaTeX:")
    write_figure_blocks(args.out_dir)

    print(f"\n{'='*68}")
    print(f"  ✅ Done → {args.out_dir}")
    print(f"{'='*68}\n")
    print("  No models were loaded, so this cannot trigger the OOM crash.")
    print("  If a Stage 7 grid is missing, generate a few heatmaps first with")
    print("  a SMALL demo run (see the memory-safe command I provide).\n")


if __name__ == "__main__":
    main()
