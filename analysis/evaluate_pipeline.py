"""
=============================================================================
END-TO-END PIPELINE EVALUATION
Thesis: Advancing Medical AI with Explainable VQA on GI Imaging

Evaluates the complete 4-stage pipeline on the full test set.

Metrics computed:
    Overall:
        - Exact Match Accuracy
        - Soft Match Accuracy   (predicted in true OR true in predicted)
        - BLEU-1, BLEU-2, BLEU-3, BLEU-4
        - METEOR
        - ROUGE-L

    Per question type (6 routes):
        - All above metrics per route
        - Precision / Recall / F1 for classification routes

    Additional:
        - Routing accuracy  (is question type correctly identified?)
        - Confidence calibration
        - Per-route BLEU breakdown
        - Qualitative examples (correct + wrong per route)

USAGE:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --n_samples 1000   # quick eval
    python evaluate_pipeline.py --save_predictions  # saves all preds to CSV
=============================================================================
"""

import os, sys, json, re, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~"))

LOG_DIR = "./logs/evaluation"
os.makedirs(LOG_DIR, exist_ok=True)

QTYPE_NAMES  = ["yes/no","single-choice","multiple-choice",
                "color","location","numerical count"]
QTYPE_COLORS = ["#2196F3","#4CAF50","#FF9800",
                "#9C27B0","#F44336","#00BCD4"]


# ─────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────
def exact_match(pred: str, true: str) -> float:
    return float(pred.strip().lower() == true.strip().lower())


def soft_match(pred: str, true: str) -> float:
    p = pred.strip().lower()
    t = true.strip().lower()
    return float(p in t or t in p or p == t)


def normalise(text: str) -> str:
    """Normalise answer for fair comparison."""
    t = text.lower().strip()
    # Remove common filler phrases
    for phrase in [
        "there is ", "there are ", "the image shows ",
        "the finding is ", "the lesion is ", "it is ",
        "evidence of ", "consistent with ", "the color is ",
        "the answer is ", "yes, ", "no, ",
        " visible", " present", " detected",
        "the image is from a colonoscopy",
        "the image is from a colonoscopy procedure",
    ]:
        t = t.replace(phrase, "")
    t = re.sub(r'[^\w\s]', '', t).strip()
    return t


def token_overlap_f1(pred: str, true: str) -> float:
    """Token-level F1 — standard VQA metric."""
    pred_toks = set(tokenise(normalise(pred)))
    true_toks = set(tokenise(normalise(true)))
    if not pred_toks or not true_toks:
        return float(pred_toks == true_toks)
    common = pred_toks & true_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(true_toks)
    return 2 * p * r / (p + r)


def tokenise(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def bleu_n(pred: str, true: str, n: int) -> float:
    """Corpus BLEU-N for a single pair."""
    pred_toks = tokenise(pred)
    true_toks = tokenise(true)
    if len(pred_toks) < n or len(true_toks) < n:
        return 0.0
    pred_ngrams = [tuple(pred_toks[i:i+n])
                   for i in range(len(pred_toks)-n+1)]
    true_ngrams = [tuple(true_toks[i:i+n])
                   for i in range(len(true_toks)-n+1)]
    true_set    = defaultdict(int)
    for g in true_ngrams:
        true_set[g] += 1
    matches = 0
    for g in pred_ngrams:
        if true_set[g] > 0:
            matches += 1
            true_set[g] -= 1
    precision = matches / max(len(pred_ngrams), 1)
    # Brevity penalty
    bp = min(1.0, len(pred_toks)/max(len(true_toks), 1))
    return bp * precision


def meteor_score(pred: str, true: str) -> float:
    """Simplified METEOR (unigram F-mean with alpha=0.9)."""
    pred_toks = set(tokenise(pred))
    true_toks = set(tokenise(true))
    if not pred_toks or not true_toks:
        return 0.0
    matches = len(pred_toks & true_toks)
    if matches == 0:
        return 0.0
    p = matches / len(pred_toks)
    r = matches / len(true_toks)
    alpha = 0.9
    return (p * r) / (alpha*p + (1-alpha)*r)


def rouge_l(pred: str, true: str) -> float:
    """ROUGE-L based on longest common subsequence."""
    pred_toks = tokenise(pred)
    true_toks = tokenise(true)
    if not pred_toks or not true_toks:
        return 0.0
    m, n = len(pred_toks), len(true_toks)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pred_toks[i-1] == true_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / max(m, 1)
    r = lcs / max(n, 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_all_metrics(preds: list, trues: list) -> dict:
    """Compute all metrics for a list of (pred, true) pairs."""
    if not preds:
        return {k: 0.0 for k in
                ["exact","soft","norm_exact","norm_soft","token_f1",
                 "bleu1","bleu2","bleu3","bleu4","meteor","rouge_l"]}
    em   = np.mean([exact_match(p,t)   for p,t in zip(preds,trues)])
    sm   = np.mean([soft_match(p,t)    for p,t in zip(preds,trues)])
    # Normalised versions
    nem  = np.mean([exact_match(normalise(p), normalise(t))
                    for p,t in zip(preds,trues)])
    nsm  = np.mean([soft_match(normalise(p), normalise(t))
                    for p,t in zip(preds,trues)])
    tf1  = np.mean([token_overlap_f1(p,t)
                    for p,t in zip(preds,trues)])
    b1   = np.mean([bleu_n(normalise(p),normalise(t),1)
                    for p,t in zip(preds,trues)])
    b2   = np.mean([bleu_n(normalise(p),normalise(t),2)
                    for p,t in zip(preds,trues)])
    b3   = np.mean([bleu_n(normalise(p),normalise(t),3)
                    for p,t in zip(preds,trues)])
    b4   = np.mean([bleu_n(normalise(p),normalise(t),4)
                    for p,t in zip(preds,trues)])
    met  = np.mean([meteor_score(normalise(p),normalise(t))
                    for p,t in zip(preds,trues)])
    rl   = np.mean([rouge_l(normalise(p),normalise(t))
                    for p,t in zip(preds,trues)])
    return dict(exact=em, soft=sm, norm_exact=nem, norm_soft=nsm,
                token_f1=tf1, bleu1=b1, bleu2=b2,
                bleu3=b3, bleu4=b4, meteor=met, rouge_l=rl)


# ─────────────────────────────────────────────────────────────────────────
# LOAD FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────
def load_pipeline():
    from stage4_answer_generation import (
        Stage4AnswerGenerator, Stage4Dataset,
        load_vocabulary, CFG as S4_CFG
    )
    from stage3_multimodal_fusion import FusionExtractor
    from preprocessing import TextPreprocessor
    from datasets import load_from_disk, Image as HFImage

    device = S4_CFG["device"]
    vocab  = load_vocabulary()

    print("   Loading Stage 3 + Stage 4 ...")
    extractor = FusionExtractor(S4_CFG["stage3_ckpt"])
    ckpt      = torch.load("./checkpoints/stage4_best.pt",
                           map_location=device)
    model = Stage4AnswerGenerator(vocab).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("   Loading test dataset ...")
    text_prep = TextPreprocessor()
    raw = load_from_disk(S4_CFG["data_dir"])
    raw = raw.cast_column("image", HFImage())

    return model, extractor, vocab, text_prep, raw["test"], device, S4_CFG


# ─────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────
def evaluate(n_samples: int = None,
             save_predictions: bool = False):

    model, extractor, vocab, text_prep, test_data, device, CFG = \
        load_pipeline()

    from stage4_answer_generation import Stage4Dataset
    from stage3_multimodal_fusion import infer_qtype_label

    test_ds = Stage4Dataset(test_data, "test", extractor, text_prep, vocab)
    loader  = DataLoader(test_ds, batch_size=32,
                         shuffle=False, num_workers=0)

    # Storage
    all_preds, all_trues, all_routes = [], [], []
    all_true_routes, all_confs       = [], []
    route_preds  = defaultdict(list)
    route_trues  = defaultdict(list)
    records      = []   # for CSV

    total = n_samples or len(test_ds)
    count = 0

    print(f"\n🔍  Evaluating {total:,} samples ...\n")

    with torch.no_grad():
        for batch in tqdm(loader, desc="   Eval", leave=False):
            if count >= total:
                break

            fused   = batch["fused_repr"].to(device)
            disease = batch["disease_vec"].to(device)
            routes  = batch["route"]           # predicted route (Stage 3)
            q_raws  = batch["question_raw"]
            a_raws  = batch["answer_raw"]

            # True route from answer content
            true_routes = [
                infer_qtype_label(q, a)
                for q, a in zip(q_raws, a_raws)
            ]

            # Routing confidence
            # We need routing logits — re-run Stage 3 routing head
            # (fused_repr already computed via FusionExtractor in dataset)
            # Use stored route probs from batch
            batch_size = fused.shape[0]

            for i in range(min(batch_size, total-count)):
                r    = routes[i].item()
                f_i  = fused[i:i+1]
                d_i  = disease[i:i+1]

                # Stage 4 predict
                answer = model.predict(f_i, d_i, r, vocab)[0]

                true_ans   = a_raws[i]
                true_route = true_routes[i]

                all_preds      .append(answer)
                all_trues      .append(true_ans)
                all_routes     .append(r)
                all_true_routes.append(true_route)
                route_preds[r] .append(answer)
                route_trues[r] .append(true_ans)

                records.append({
                    "question"   : q_raws[i],
                    "true_answer": true_ans,
                    "pred_answer": answer,
                    "true_route" : QTYPE_NAMES[true_route],
                    "pred_route" : QTYPE_NAMES[r],
                    "route_match": true_route == r,
                    "exact_match": exact_match(answer, true_ans),
                    "soft_match" : soft_match(answer, true_ans),
                    "bleu1"      : bleu_n(normalise(answer), normalise(true_ans), 1),
                    "rouge_l"    : rouge_l(normalise(answer), normalise(true_ans)),
                })
                count += 1

    # ── Overall metrics ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"END-TO-END EVALUATION RESULTS  (n={count:,})")
    print(f"{'='*70}")

    overall = compute_all_metrics(all_preds, all_trues)
    routing_acc = np.mean([p==t for p,t in
                           zip(all_routes, all_true_routes)])

    print(f"\n  OVERALL METRICS:")
    print(f"    Exact Match Accuracy       : {overall['exact']*100:.2f}%")
    print(f"    Soft  Match Accuracy       : {overall['soft']*100:.2f}%")
    print(f"    Normalised Exact Match     : {overall['norm_exact']*100:.2f}%")
    print(f"    Normalised Soft  Match     : {overall['norm_soft']*100:.2f}%")
    print(f"    Token-Level F1 (VQA std)   : {overall['token_f1']*100:.2f}%")
    print(f"    BLEU-1                     : {overall['bleu1']:.4f}")
    print(f"    BLEU-2                     : {overall['bleu2']:.4f}")
    print(f"    BLEU-3                     : {overall['bleu3']:.4f}")
    print(f"    BLEU-4                     : {overall['bleu4']:.4f}")
    print(f"    METEOR                     : {overall['meteor']:.4f}")
    print(f"    ROUGE-L                    : {overall['rouge_l']:.4f}")
    print(f"    Routing Accuracy           : {routing_acc*100:.2f}%")

    # ── Per-route metrics ─────────────────────────────────────────────
    print(f"\n  PER-ROUTE METRICS (normalised):")
    print(f"  {'Route':<18} {'n':>5}  {'Norm-Ex':>7}  {'Tok-F1':>6}"
          f"  {'BLEU-1':>6}  {'METEOR':>6}  {'ROUGE-L':>7}")
    print(f"  {'-'*65}")

    route_metrics = {}
    for r in range(6):
        p_list = route_preds[r]
        t_list = route_trues[r]
        if not p_list:
            continue
        m = compute_all_metrics(p_list, t_list)
        route_metrics[r] = m
        n = len(p_list)
        print(f"  {QTYPE_NAMES[r]:<18} {n:>5}  "
              f"{m['norm_exact']*100:>6.1f}%  "
              f"{m['token_f1']*100:>6.1f}%  "
              f"{m['bleu1']:>6.4f}  "
              f"{m['meteor']:>6.4f}  "
              f"{m['rouge_l']:>7.4f}")

    print(f"\n{'='*70}")

    # ── Save results ──────────────────────────────────────────────────
    df_records = pd.DataFrame(records)
    df_records.to_csv(f"{LOG_DIR}/predictions.csv", index=False)
    print(f"\n✅  Predictions saved → {LOG_DIR}/predictions.csv")

    # Summary CSV
    summary_rows = [
        ["Overall", count,
         overall["exact"], overall["soft"],
         overall["bleu1"], overall["bleu2"],
         overall["bleu3"], overall["bleu4"],
         overall["meteor"], overall["rouge_l"],
         routing_acc]
    ]
    for r in range(6):
        if r not in route_metrics:
            continue
        m = route_metrics[r]
        rout_acc = np.mean([p==r for p in all_routes
                            if all_true_routes[all_routes.index(p)]==r]) \
                   if r in all_true_routes else 0
        summary_rows.append([
            QTYPE_NAMES[r], len(route_preds[r]),
            m["exact"], m["soft"],
            m["bleu1"], m["bleu2"], m["bleu3"], m["bleu4"],
            m["meteor"], m["rouge_l"], None
        ])

    df_summary = pd.DataFrame(summary_rows,
        columns=["Route","N","Exact","Soft","BLEU1","BLEU2",
                 "BLEU3","BLEU4","METEOR","ROUGE-L","Routing Acc"])
    df_summary.to_csv(f"{LOG_DIR}/evaluation_summary.csv", index=False)
    print(f"✅  Summary saved    → {LOG_DIR}/evaluation_summary.csv")

    # ── Generate all figures ──────────────────────────────────────────
    _plot_overall_metrics(overall, routing_acc)
    _plot_per_route_metrics(route_metrics, route_preds)
    _plot_metric_heatmap(route_metrics)
    _plot_bleu_breakdown(route_metrics)
    _plot_qualitative_examples(df_records)
    _plot_routing_confusion(all_routes, all_true_routes)

    return overall, route_metrics, df_records


# ─────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────
def _plot_overall_metrics(overall: dict, routing_acc: float):
    print("\n📊  Plotting overall metrics ...")

    metrics = {
        "Exact Match"  : overall["exact"]*100,
        "Soft Match"   : overall["soft"]*100,
        "BLEU-1"       : overall["bleu1"]*100,
        "BLEU-2"       : overall["bleu2"]*100,
        "BLEU-3"       : overall["bleu3"]*100,
        "BLEU-4"       : overall["bleu4"]*100,
        "METEOR"       : overall["meteor"]*100,
        "ROUGE-L"      : overall["rouge_l"]*100,
        "Routing Acc"  : routing_acc*100,
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("End-to-End Pipeline Evaluation — Overall Metrics\n"
                 "Full 4-Stage VQA Pipeline on Kvasir-VQA-x1 Test Set",
                 fontsize=13, fontweight="bold")

    names  = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#4CAF50" if v >= 70 else
              "#FF9800" if v >= 40 else
              "#F44336" for v in values]

    bars = axes[0].bar(names, values, color=colors,
                       alpha=0.85, edgecolor="white", width=0.7)
    axes[0].set_title("(a) All Metrics (%)", fontweight="bold")
    axes[0].set_ylabel("Score (%)"); axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.8,
                     f"{val:.1f}%", ha="center",
                     fontsize=9, fontweight="bold")

    # Radar chart
    cats    = ["Exact\nMatch","Soft\nMatch","BLEU-1",
               "METEOR","ROUGE-L","Routing\nAcc"]
    vals    = [overall["exact"], overall["soft"],
               overall["bleu1"], overall["meteor"],
               overall["rouge_l"], routing_acc]
    N       = len(cats)
    angles  = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]
    vals_r  = vals + vals[:1]

    ax2 = plt.subplot(122, polar=True)
    ax2.plot(angles, vals_r, "b-o", lw=2, ms=6)
    ax2.fill(angles, vals_r, alpha=0.15, color="blue")
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["20%","40%","60%","80%","100%"],
                        fontsize=6)
    ax2.set_title("(b) Metric Radar", fontweight="bold",
                  pad=20, fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/eval_overall_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


def _plot_per_route_metrics(route_metrics: dict, route_preds: dict):
    print("📊  Plotting per-route metrics ...")

    routes  = sorted(route_metrics.keys())
    names   = [QTYPE_NAMES[r] for r in routes]
    n_each  = [len(route_preds[r]) for r in routes]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("End-to-End Evaluation — Per-Route Metrics",
                 fontsize=13, fontweight="bold")

    metric_list = [
        ("exact",  "Exact Match Accuracy", axes[0,0]),
        ("soft",   "Soft Match Accuracy",  axes[0,1]),
        ("bleu1",  "BLEU-1",               axes[0,2]),
        ("bleu4",  "BLEU-4",               axes[1,0]),
        ("meteor", "METEOR",               axes[1,1]),
        ("rouge_l","ROUGE-L",              axes[1,2]),
    ]

    for metric, title, ax in metric_list:
        vals   = [route_metrics[r][metric]*100 for r in routes]
        colors = [QTYPE_COLORS[r] for r in routes]
        bars   = ax.bar(names, vals, color=colors,
                        alpha=0.85, edgecolor="white", width=0.6)
        ax.set_title(f"{title}", fontweight="bold")
        ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val, n in zip(bars, vals, n_each):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.8,
                    f"{val:.1f}%", ha="center",
                    fontsize=8, fontweight="bold")

    plt.tight_layout()
    path = f"{LOG_DIR}/eval_per_route_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


def _plot_metric_heatmap(route_metrics: dict):
    print("📊  Plotting metric heatmap ...")

    routes  = sorted(route_metrics.keys())
    metrics = ["exact","soft","bleu1","bleu2","bleu3",
               "bleu4","meteor","rouge_l"]
    m_labels= ["Exact","Soft","BLEU-1","BLEU-2","BLEU-3",
               "BLEU-4","METEOR","ROUGE-L"]
    r_labels= [QTYPE_NAMES[r] for r in routes]

    data = np.array([
        [route_metrics[r][m]*100 for m in metrics]
        for r in routes
    ])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(data, ax=ax,
                annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100,
                xticklabels=m_labels, yticklabels=r_labels,
                linewidths=0.3, linecolor="white",
                annot_kws={"size":9})
    ax.set_title("Metric Heatmap — All Routes × All Metrics (%)\n"
                 "End-to-End Pipeline Evaluation",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    path = f"{LOG_DIR}/eval_metric_heatmap.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


def _plot_bleu_breakdown(route_metrics: dict):
    print("📊  Plotting BLEU breakdown ...")

    routes = sorted(route_metrics.keys())
    names  = [QTYPE_NAMES[r] for r in routes]
    x      = np.arange(len(routes))
    w      = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("BLEU Score Breakdown per Route (BLEU-1 to BLEU-4)",
                 fontsize=12, fontweight="bold")

    for i, n in enumerate([1,2,3,4]):
        vals = [route_metrics[r][f"bleu{n}"]*100 for r in routes]
        ax.bar(x + i*w - 1.5*w, vals, w,
               label=f"BLEU-{n}", alpha=0.85,
               edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_ylabel("BLEU Score (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.set_title("BLEU-1 through BLEU-4 per Answer Route",
                 fontweight="bold")

    plt.tight_layout()
    path = f"{LOG_DIR}/eval_bleu_breakdown.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


def _plot_qualitative_examples(df: pd.DataFrame):
    print("📊  Plotting qualitative examples ...")

    fig, axes = plt.subplots(6, 1, figsize=(16, 22))
    fig.suptitle("End-to-End Evaluation — Qualitative Examples\n"
                 "2 correct + 2 wrong per route",
                 fontsize=13, fontweight="bold")

    for row, r in enumerate(range(6)):
        ax = axes[row]
        ax.axis("off")
        ax.set_facecolor("#F8F9FA")

        route_df = df[df["pred_route"] == QTYPE_NAMES[r]]
        correct  = route_df[route_df["exact_match"] == 1].head(2)
        wrong    = route_df[route_df["exact_match"] == 0].head(2)

        ax.set_title(f"Route {r}: [{QTYPE_NAMES[r]}]",
                     fontweight="bold", fontsize=10,
                     color=QTYPE_COLORS[r], loc="left")

        lines = []
        for _, ex in correct.iterrows():
            lines.append(
                f"  ✓  Q: {ex['question'][:70]}\n"
                f"      True: {ex['true_answer'][:50]}"
                f"   |   Pred: {ex['pred_answer'][:50]}"
            )
        for _, ex in wrong.iterrows():
            lines.append(
                f"  ✗  Q: {ex['question'][:70]}\n"
                f"      True: {ex['true_answer'][:50]}"
                f"   |   Pred: {ex['pred_answer'][:50]}"
            )

        text = "\n".join(lines) if lines else "  No examples available"
        ax.text(0.01, 0.5, text, transform=ax.transAxes,
                fontsize=8, va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round",
                          facecolor=QTYPE_COLORS[r]+"15",
                          edgecolor=QTYPE_COLORS[r],
                          linewidth=1.5))

    plt.tight_layout()
    path = f"{LOG_DIR}/eval_qualitative_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


def _plot_routing_confusion(pred_routes: list, true_routes: list):
    print("📊  Plotting routing confusion matrix ...")
    from sklearn.metrics import confusion_matrix

    cm      = confusion_matrix(true_routes, pred_routes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True)+1e-9)
    routing_acc = np.mean(np.array(pred_routes) == np.array(true_routes))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Routing Accuracy Analysis\n"
                 f"Overall Routing Accuracy={routing_acc*100:.2f}%",
                 fontsize=12, fontweight="bold")

    sns.heatmap(cm, ax=axes[0],
                annot=True, fmt="d", cmap="Blues",
                xticklabels=QTYPE_NAMES,
                yticklabels=QTYPE_NAMES,
                linewidths=0.3, linecolor="white",
                annot_kws={"size":9})
    axes[0].set_title("(a) Raw Routing Confusion Matrix",
                      fontweight="bold")
    axes[0].set_xlabel("Predicted Route")
    axes[0].set_ylabel("True Route")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)

    sns.heatmap(cm_norm, ax=axes[1],
                annot=True, fmt=".2f", cmap="Blues",
                xticklabels=QTYPE_NAMES,
                yticklabels=QTYPE_NAMES,
                linewidths=0.3, linecolor="white",
                annot_kws={"size":9})
    axes[1].set_title("(b) Normalised Routing Confusion Matrix",
                      fontweight="bold")
    axes[1].set_xlabel("Predicted Route")
    axes[1].set_ylabel("True Route")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)

    plt.tight_layout()
    path = f"{LOG_DIR}/eval_routing_confusion.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ─────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
        help="Limit samples for quick eval (default: full test set)")
    parser.add_argument("--save_predictions", action="store_true",
        help="Save all predictions to CSV")
    args = parser.parse_args()

    print("\n📊  End-to-End Pipeline Evaluation\n" + "="*60)
    overall, route_metrics, df = evaluate(
        n_samples=args.n_samples,
        save_predictions=args.save_predictions,
    )

    print("\n" + "="*60)
    print("✅  All evaluation outputs saved to ./logs/evaluation/")
    print("    eval_overall_metrics.png    — radar + bar chart")
    print("    eval_per_route_metrics.png  — 6 metrics × 6 routes")
    print("    eval_metric_heatmap.png     — full heatmap")
    print("    eval_bleu_breakdown.png     — BLEU-1 to BLEU-4")
    print("    eval_qualitative_examples.png — correct/wrong examples")
    print("    eval_routing_confusion.png  — routing accuracy CM")
    print("    predictions.csv             — all predictions")
    print("    evaluation_summary.csv      — summary table")
