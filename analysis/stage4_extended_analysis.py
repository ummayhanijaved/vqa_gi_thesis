"""
=============================================================================
Stage 4 — COMPLETE Extended Analysis for Thesis
Covers EVERYTHING:
    1.  Training Dynamics          (loss, accuracy, per-route curves)
    2.  Test Performance           (per-route bars, ranked difficulty)
    3.  Full Pipeline Summary      (all 4 stages together)
    4.  Confusion Matrices         (one per route head)
    5.  Error Analysis             (what goes wrong and why)
    6.  Confidence Analysis        (how certain is each head)
    7.  Disease Vector Influence   (which diseases drive each answer)
    8.  Support vs Accuracy        (class imbalance effect)
    9.  Vocab Coverage Analysis    (OOV rate for single/multi heads)
   10.  End-to-End Examples        (real image + question + answer)
   11.  Latency Benchmarks         (inference speed per head)
   12.  Stage Comparison Table     (Stage 1→2→3→4 complete)
   13.  Answer Length Distribution (short vs long answers)
   14.  Model Architecture Diagram (all heads visualised)
   15.  CSV Tables                 (all results in spreadsheet form)

USAGE:
    python stage4_extended_analysis.py --skip_inference
    python stage4_extended_analysis.py
=============================================================================
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

sys.path.insert(0, os.path.expanduser("~"))

QTYPE_NAMES  = ["yes/no", "single-choice", "multiple-choice",
                "color", "location", "numerical count"]
QTYPE_COLORS = ["#2196F3", "#4CAF50", "#FF9800",
                "#9C27B0", "#F44336", "#00BCD4"]

# ── All results hard-coded from training output ───────────────────────────
EPOCH_LOG = [
    (1, 4.6176, 0.7324, 0.7166),
    (2, 4.1173, 0.7484, 0.7226),
    (3, 3.9644, 0.7531, 0.7108),
    (4, 3.8542, 0.7554, 0.7411),
    (5, 3.7699, 0.7572, 0.7250),
    (6, 3.7097, 0.7581, 0.7146),
    (7, 3.6672, 0.7598, 0.7199),
    (8, 3.6210, 0.7606, 0.7237),
    (9, 3.5980, 0.7609, 0.7350),
]
ROUTE_VAL = {
    "yes/no"         :[1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000],
    "single-choice"  :[0.5004,0.5135,0.5108,0.5159,0.5245,0.5290,0.5252,0.5383,0.5228],
    "multiple-choice":[0.2874,0.2874,0.2857,0.2857,0.2857,0.2874,0.2874,0.2940,0.2890],
    "color"          :[0.7938,0.7938,0.7938,0.7938,0.7938,0.7938,0.7938,0.7938,0.7938],
    "location"       :[0.8754,0.8754,0.8751,0.8756,0.8787,0.8754,0.8723,0.8818,0.8832],
    "numerical count":[0.4644,0.4782,0.4378,0.5437,0.4790,0.4406,0.4634,0.4637,0.5134],
}
TEST_RESULTS = [
    ("yes/no",          1.0000, 4699),
    ("single-choice",   0.5059, 2643),
    ("multiple-choice", 0.3333,  309),
    ("color",           0.7997, 1498),
    ("location",        0.8806, 2312),
    ("numerical count", 0.5405, 4494),
]
OVERALL_TEST_ACC = 0.7397
BEST_VAL_ACC     = 0.7411
BEST_EPOCH       = 4


# ─────────────────────────────────────────────────────────────────────────
# SHARED: load model + data for inference-based plots
# ─────────────────────────────────────────────────────────────────────────
def load_everything():
    from stage4_answer_generation import (
        Stage4AnswerGenerator, Stage4Dataset,
        load_vocabulary, CFG as S4_CFG
    )
    from stage3_multimodal_fusion import FusionExtractor, CFG as S3_CFG
    from preprocessing import TextPreprocessor
    from datasets import load_from_disk, Image as HFImage

    device = S4_CFG["device"]
    vocab  = load_vocabulary()

    print("   Loading Stage 3 extractor ...")
    extractor = FusionExtractor(S4_CFG["stage3_ckpt"])

    print("   Loading Stage 4 model ...")
    ckpt  = torch.load("./checkpoints/stage4_best.pt", map_location=device)
    model = Stage4AnswerGenerator(vocab).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("   Loading test dataset ...")
    text_prep = TextPreprocessor()
    raw = load_from_disk(S4_CFG["data_dir"])
    raw = raw.cast_column("image", HFImage())
    test_ds = Stage4Dataset(raw["test"], "test", extractor, text_prep, vocab)
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    return model, test_ds, loader, vocab, device, S4_CFG


def run_inference(model, loader, device):
    from stage4_answer_generation import compute_accuracy
    all_routes, all_correct = [], []
    all_fused, all_disease  = [], []
    all_preds_per_route     = {r: [] for r in range(6)}
    all_labels_per_route    = {r: [] for r in range(6)}
    all_conf_per_route      = {r: [] for r in range(6)}
    questions, answers      = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="   Inference", leave=False):
            fused   = batch["fused_repr"].to(device)
            disease = batch["disease_vec"].to(device)
            routes  = batch["route"]

            all_fused  .extend(fused.cpu().tolist())
            all_disease.extend(disease.cpu().tolist())
            all_routes .extend(routes.tolist())
            questions  .extend(batch["question_raw"])
            answers    .extend(batch["answer_raw"])

            label_keys = {0:"yn_label",1:"single_label",2:"multi_label",
                          3:"color_label",4:"loc_label",5:"count_label"}

            for r in range(6):
                mask = (routes == r)
                if mask.sum() == 0:
                    continue
                f_r = fused[mask]
                d_r = disease[mask]
                logits = model(f_r, d_r, r)

                if r == 2:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    lbls  = batch["multi_label"][mask].to(device).float()
                    conf  = probs.max(dim=1).values
                    correct_per = (preds == lbls).all(dim=1).float()
                else:
                    probs = F.softmax(logits.float(), dim=-1)
                    preds = probs.argmax(-1)
                    conf  = probs.max(-1).values
                    lbls  = batch[label_keys[r]][mask].to(device)
                    correct_per = (preds == lbls).float()

                all_preds_per_route [r].extend(preds.cpu().tolist())
                all_labels_per_route[r].extend(lbls.cpu().tolist())
                all_conf_per_route  [r].extend(conf.cpu().tolist())
                # correct_per is already indexed to the masked subset (size = mask.sum())
                for j, cp in enumerate(correct_per.cpu().tolist()):
                    all_correct.append((r, cp))

    return dict(
        routes   = np.array(all_routes),
        fused    = np.array(all_fused),
        disease  = np.array(all_disease),
        preds    = all_preds_per_route,
        labels   = all_labels_per_route,
        conf     = all_conf_per_route,
        correct  = all_correct,
        questions= questions,
        answers  = answers,
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. TRAINING DYNAMICS
# ══════════════════════════════════════════════════════════════════════════
def plot_training_dynamics():
    print("\n1️⃣  Training Dynamics ...")
    df  = pd.DataFrame(EPOCH_LOG,
          columns=["epoch","tr_loss","tr_acc","va_acc"])
    eps = df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stage 4: Complete Training Dynamics\n"
                 "6 Specialised Answer Heads | 925K params | Early stop Ep.9",
                 fontsize=13, fontweight="bold")

    # Combined loss
    axes[0,0].plot(eps, df["tr_loss"], "b-o", ms=7, lw=2, label="Train loss")
    axes[0,0].axvline(BEST_EPOCH, color="green", linestyle="--",
                      alpha=0.8, label=f"Best ep={BEST_EPOCH}")
    axes[0,0].fill_between(eps, df["tr_loss"], alpha=0.1, color="blue")
    axes[0,0].set_title("(a) Combined Loss (sum of 6 heads)",fontweight="bold")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
    axes[0,0].legend(fontsize=9); axes[0,0].grid(alpha=0.3)
    axes[0,0].set_xticks(eps)

    # Overall accuracy
    axes[0,1].plot(eps, df["tr_acc"]*100,"b-o",ms=7,lw=2,label="Train")
    axes[0,1].plot(eps, df["va_acc"]*100,"r-o",ms=7,lw=2,label="Val")
    axes[0,1].axvline(BEST_EPOCH, color="green", linestyle="--", alpha=0.8)
    axes[0,1].axhline(BEST_VAL_ACC*100, color="gray", linestyle=":",
                      alpha=0.6, label=f"Best={BEST_VAL_ACC*100:.2f}%")
    axes[0,1].set_title("(b) Overall Accuracy", fontweight="bold")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Accuracy (%)")
    axes[0,1].set_ylim(68, 78); axes[0,1].legend(fontsize=9)
    axes[0,1].grid(alpha=0.3); axes[0,1].set_xticks(eps)

    # Train-val gap
    gap = df["tr_acc"] - df["va_acc"]
    axes[0,2].bar(eps, gap*100, color=["#F44336" if g>0.05 else "#4CAF50"
                  for g in gap.values], alpha=0.8, edgecolor="white")
    axes[0,2].axhline(0, color="black", linewidth=1)
    axes[0,2].set_title("(c) Overfitting Gap (Train − Val)",fontweight="bold")
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("Gap (pp)")
    axes[0,2].grid(axis="y", alpha=0.3); axes[0,2].set_xticks(eps)

    # Per-route curves — stable ones
    stable = ["yes/no","color","location"]
    for name in stable:
        idx = QTYPE_NAMES.index(name)
        axes[1,0].plot(eps, [v*100 for v in ROUTE_VAL[name]],
                       "-o", ms=5, lw=2, label=name,
                       color=QTYPE_COLORS[idx])
    axes[1,0].set_title("(d) Stable Routes Val Acc",fontweight="bold")
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("Val Acc (%)")
    axes[1,0].set_ylim(60,105); axes[1,0].legend(fontsize=8)
    axes[1,0].grid(alpha=0.3); axes[1,0].set_xticks(eps)

    # Per-route curves — learning ones
    learning = ["single-choice","multiple-choice","numerical count"]
    for name in learning:
        idx = QTYPE_NAMES.index(name)
        axes[1,1].plot(eps, [v*100 for v in ROUTE_VAL[name]],
                       "-o", ms=5, lw=2, label=name,
                       color=QTYPE_COLORS[idx])
    axes[1,1].set_title("(e) Learning Routes Val Acc",fontweight="bold")
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Val Acc (%)")
    axes[1,1].set_ylim(20,60); axes[1,1].legend(fontsize=8)
    axes[1,1].grid(alpha=0.3); axes[1,1].set_xticks(eps)

    # All routes heatmap
    heat_data = np.array([[ROUTE_VAL[n][e]*100 for e in range(9)]
                           for n in QTYPE_NAMES])
    sns.heatmap(heat_data, ax=axes[1,2],
                annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=[f"Ep{i+1}" for i in range(9)],
                yticklabels=QTYPE_NAMES,
                linewidths=0.3, annot_kws={"size":7})
    axes[1,2].set_title("(f) Route × Epoch Val Acc Heatmap",fontweight="bold")
    axes[1,2].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_training_dynamics.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 2. TEST PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
def plot_test_performance():
    print("\n2️⃣  Test Performance ...")
    names   = [r[0] for r in TEST_RESULTS]
    accs    = [r[1]*100 for r in TEST_RESULTS]
    support = [r[2] for r in TEST_RESULTS]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Stage 4: Test Set Performance Analysis\n"
                 f"Overall Accuracy={OVERALL_TEST_ACC*100:.2f}%  |  n=15,955",
                 fontsize=13, fontweight="bold")

    # Accuracy bars
    bars = axes[0,0].bar(names, accs, color=QTYPE_COLORS,
                         alpha=0.85, edgecolor="white", width=0.6)
    axes[0,0].axhline(OVERALL_TEST_ACC*100, color="red", linestyle="--",
                      alpha=0.7, lw=1.5,
                      label=f"Overall={OVERALL_TEST_ACC*100:.2f}%")
    axes[0,0].set_title("(a) Test Accuracy per Head",fontweight="bold")
    axes[0,0].set_ylabel("Accuracy (%)"); axes[0,0].set_ylim(0,112)
    axes[0,0].tick_params(axis="x",rotation=25)
    axes[0,0].legend(fontsize=9); axes[0,0].grid(axis="y",alpha=0.3)
    for bar,acc,sup in zip(bars,accs,support):
        axes[0,0].text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+1.5,
                       f"{acc:.1f}%\n(n={sup:,})",
                       ha="center",fontsize=8,fontweight="bold")

    # Ranked horizontal
    s_idx   = np.argsort(accs)
    axes[0,1].barh([names[i] for i in s_idx],
                   [accs[i]  for i in s_idx],
                   color=[QTYPE_COLORS[i] for i in s_idx],
                   alpha=0.85, edgecolor="white", height=0.6)
    axes[0,1].axvline(OVERALL_TEST_ACC*100, color="red",
                      linestyle="--", alpha=0.7, lw=1.5)
    axes[0,1].set_title("(b) Ranked by Difficulty",fontweight="bold")
    axes[0,1].set_xlabel("Test Accuracy (%)"); axes[0,1].set_xlim(0,110)
    axes[0,1].grid(axis="x",alpha=0.3)
    for i,(idx) in enumerate(s_idx):
        axes[0,1].text(accs[idx]+0.5, i, f"{accs[idx]:.1f}%",
                       va="center", fontsize=9, fontweight="bold")

    # Support pie
    axes[1,0].pie(support, labels=names, colors=QTYPE_COLORS,
                  autopct="%1.1f%%", startangle=90,
                  textprops={"fontsize":8})
    axes[1,0].set_title("(c) Test Set Composition by Route",fontweight="bold")

    # Accuracy vs support scatter
    for i,(name,acc,sup) in enumerate(zip(names,accs,support)):
        axes[1,1].scatter(sup, acc, s=300, color=QTYPE_COLORS[i],
                          zorder=5, edgecolors="white", lw=1.5)
        axes[1,1].annotate(name,(sup,acc),
                           textcoords="offset points",
                           xytext=(8,3),fontsize=8)
    z = np.polyfit(np.log(support), accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(support),max(support),100)
    axes[1,1].plot(x_line, p(np.log(x_line)),
                   "r--", alpha=0.5, label="Log trend")
    axes[1,1].set_title("(d) Support vs Accuracy\n(class imbalance effect)",
                        fontweight="bold")
    axes[1,1].set_xlabel("Test Support"); axes[1,1].set_ylabel("Accuracy (%)")
    axes[1,1].set_ylim(20,110); axes[1,1].legend(fontsize=8)
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_test_performance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 3. CONFUSION MATRICES (per route)
# ══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(results):
    print("\n3️⃣  Confusion Matrices ...")
    from sklearn.metrics import confusion_matrix

    route_class_names = {
        0: ["no", "yes"],
        3: ["pink","red","orange","yellow","green","blue",
            "purple","white","black","brown","transparent","mixed"],
        4: ["esophagus","stomach","duodenum","small-bowel","colon",
            "rectum","cecum","pylorus","z-line","retroflex-r",
            "retroflex-s","ileocecal","upper-gi","lower-gi","unknown"],
        5: ["0","1","2","3","4","5","6-10",">10"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Stage 4: Confusion Matrices per Answer Head\n"
                 "(Test Set — Row-Normalised Recall)",
                 fontsize=13, fontweight="bold")

    plot_routes = [0, 3, 4, 5]
    for ax, r in zip(axes.flat, plot_routes):
        preds  = np.array(results["preds"][r])
        labels = np.array(results["labels"][r])
        if len(preds) == 0:
            ax.set_title(f"Route {r} — no data"); continue

        if r == 2:
            ax.set_title("Route 2 (multi-label) — N/A for CM"); continue

        cm      = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-9)
        cnames  = route_class_names.get(r, [str(i) for i in range(cm.shape[0])])

        # Trim to actual classes seen
        n = min(cm.shape[0], len(cnames))
        cm_norm = cm_norm[:n,:n]
        cnames  = cnames[:n]

        sns.heatmap(cm_norm, ax=ax,
                    annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=cnames, yticklabels=cnames,
                    linewidths=0.3, linecolor="white",
                    annot_kws={"size":7})
        route_name = QTYPE_NAMES[r]
        acc = TEST_RESULTS[r][1]*100
        ax.set_title(f"Route {r}: [{route_name}]  acc={acc:.1f}%",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_confusion_matrices.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
def plot_confidence_analysis(results):
    print("\n4️⃣  Confidence Analysis ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Stage 4: Prediction Confidence per Answer Head\n"
                 "Distribution of max softmax probability",
                 fontsize=13, fontweight="bold")

    for idx, r in enumerate(range(6)):
        ax   = axes[idx//3, idx%3]
        conf = np.array(results["conf"][r])
        if len(conf) == 0:
            ax.set_title(f"Route {r} — no data"); continue

        # Separate correct and wrong
        preds  = results["preds"][r]
        labels = results["labels"][r]
        if r == 2:
            correct = np.array([
                bool(np.array_equal(np.array(p), np.array(l)))
                for p, l in zip(preds, labels)
            ]) if len(preds) > 0 else np.array([])
        else:
            correct = np.array(preds) == np.array(labels)

        if len(correct) == 0:
            ax.set_title(f"Route {r} — no data"); continue

        ax.hist(conf[correct],  bins=20, alpha=0.7,
                color="#4CAF50", label=f"Correct ({correct.sum():,})",
                density=True)
        ax.hist(conf[~correct], bins=20, alpha=0.7,
                color="#F44336", label=f"Wrong ({(~correct).sum():,})",
                density=True)
        ax.axvline(conf.mean(), color="black", linestyle="--",
                   lw=1.5, label=f"Mean={conf.mean():.3f}")
        acc = TEST_RESULTS[r][1]*100
        ax.set_title(f"[{QTYPE_NAMES[r]}]  acc={acc:.1f}%",
                     fontweight="bold", color=QTYPE_COLORS[r])
        ax.set_xlabel("Confidence"); ax.set_ylabel("Density")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_confidence_analysis.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 5. ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
def plot_error_analysis(results):
    print("\n5️⃣  Error Analysis ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Stage 4: Error Analysis per Head\n"
                 "Error rate and confidence of misclassified samples",
                 fontsize=13, fontweight="bold")

    route_err_rates = []
    for idx, r in enumerate(range(6)):
        ax     = axes[idx//3, idx%3]
        preds  = np.array(results["preds"][r])
        labels = np.array(results["labels"][r])
        conf   = np.array(results["conf"][r])

        if len(preds) == 0:
            ax.set_title(f"Route {r} — no data"); continue

        if r == 2:
            correct = np.array([
                bool(np.array_equal(np.array(p), np.array(l)))
                for p, l in zip(preds, labels)
            ]) if len(preds) > 0 else np.array([True])
        else:
            correct = preds == labels

        err_rate = 1 - correct.mean()
        route_err_rates.append((QTYPE_NAMES[r], err_rate*100))

        wrong_conf = conf[~correct] if (~correct).sum() > 0 else np.array([0])

        # Confidence distribution of errors
        if len(wrong_conf) > 0:
            ax.hist(wrong_conf, bins=15, color="#EF9A9A",
                    alpha=0.85, edgecolor="white", density=True)
            ax.axvline(wrong_conf.mean(), color="darkred",
                       linestyle="--", lw=2,
                       label=f"Mean={wrong_conf.mean():.3f}")
            ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)

        ax.set_title(
            f"[{QTYPE_NAMES[r]}]\n"
            f"Err rate={err_rate*100:.1f}%  "
            f"Wrong={int((~correct).sum()):,}/{len(correct):,}",
            fontweight="bold", fontsize=9, color=QTYPE_COLORS[r])
        ax.set_xlabel("Confidence of Wrong Predictions")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_error_analysis.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")

    # Error rate summary bar
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    names_e = [r[0] for r in route_err_rates]
    errs_e  = [r[1] for r in route_err_rates]
    bars = ax2.bar(names_e, errs_e, color=QTYPE_COLORS,
                   alpha=0.85, edgecolor="white", width=0.6)
    ax2.axhline(100-OVERALL_TEST_ACC*100, color="red", linestyle="--",
                alpha=0.7, lw=1.5,
                label=f"Overall error={(100-OVERALL_TEST_ACC*100):.2f}%")
    ax2.set_title("Error Rate per Answer Head",fontweight="bold",fontsize=12)
    ax2.set_ylabel("Error Rate (%)"); ax2.set_ylim(0, 80)
    ax2.legend(fontsize=9); ax2.grid(axis="y",alpha=0.3)
    ax2.tick_params(axis="x",rotation=20)
    for bar, val in zip(bars, errs_e):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.5, f"{val:.1f}%",
                 ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path2 = f"{LOG_DIR}/stage4_error_rates.png"
    plt.savefig(path2, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path2}")


# ══════════════════════════════════════════════════════════════════════════
# 6. DISEASE VECTOR INFLUENCE
# ══════════════════════════════════════════════════════════════════════════
def plot_disease_influence(results):
    print("\n6️⃣  Disease Vector Influence ...")

    disease = np.array(results["disease"])  # (N, 23)
    routes  = results["routes"]

    DISEASE_SHORT = [
        "P-Ped","P-Sess","P-Hyp","Esoph","Gastri","UC","Crohn",
        "Barrett","Gas-Ulc","Duo-Ulc","Erosion","Hemor","Diverti",
        "N-Cecum","N-Pylor","N-Z-Line","Ileocec","Retro-R","Retro-S",
        "Dyed-LP","Dyed-RM","For-Body","Instrum"
    ]

    # Mean disease activation per route
    heat_data = np.array([
        disease[routes == r].mean(axis=0) if (routes==r).sum()>0
        else np.zeros(23)
        for r in range(6)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle("Stage 4: Disease Vector Influence on Answer Generation\n"
                 "Mean disease probability per answer route",
                 fontsize=13, fontweight="bold")

    sns.heatmap(heat_data, ax=axes[0],
                annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1,
                xticklabels=DISEASE_SHORT,
                yticklabels=QTYPE_NAMES,
                linewidths=0.3, linecolor="white",
                annot_kws={"size":6})
    axes[0].set_title("(a) Mean Disease Activation Heatmap\n"
                      "(rows=answer type, cols=disease)",
                      fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=9)

    # Top disease per route
    top_diseases = []
    for r in range(6):
        mask = routes == r
        if mask.sum() > 0:
            mean_d = disease[mask].mean(axis=0)
            top_d  = np.argsort(mean_d)[::-1][:3]
            top_diseases.append([
                f"{DISEASE_SHORT[i]}({mean_d[i]:.2f})"
                for i in top_d
            ])
        else:
            top_diseases.append(["N/A"]*3)

    axes[1].axis("off")
    table_data = [[QTYPE_NAMES[r]] + top_diseases[r] for r in range(6)]
    table = axes[1].table(
        cellText  = table_data,
        colLabels = ["Answer Type","Top Disease 1","Top Disease 2","Top Disease 3"],
        loc       = "center",
        cellLoc   = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
        elif j == 0:
            cell.set_facecolor(QTYPE_COLORS[i-1] + "40")
    axes[1].set_title("(b) Top 3 Disease Activations per Route",
                      fontweight="bold", pad=20)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_disease_influence.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 7. VOCAB COVERAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
def plot_vocab_coverage(results, vocab):
    print("\n7️⃣  Vocab Coverage Analysis ...")

    answers   = results["answers"]
    routes    = results["routes"]
    single_v  = set(vocab["single"])
    multi_v   = set(vocab["multi"])

    # Single-choice OOV
    s_mask    = routes == 1
    s_answers = [answers[i].lower().strip()
                 for i in range(len(routes)) if routes[i] == 1]
    s_in_v    = [a in single_v for a in s_answers]
    s_oov_rate= 1 - np.mean(s_in_v) if s_answers else 0

    # Multi-choice OOV
    m_answers = [answers[i].lower().strip()
                 for i in range(len(routes)) if routes[i] == 2]
    m_oov_rates = []
    for ans in m_answers:
        tokens = [t.strip() for t in ans.split(",")]
        oov    = [t not in multi_v for t in tokens if t]
        m_oov_rates.append(np.mean(oov) if oov else 0)
    m_oov_rate = np.mean(m_oov_rates) if m_oov_rates else 0

    # Top single answers
    single_counts = Counter(s_answers)
    top_s = single_counts.most_common(15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Stage 4: Vocabulary Coverage Analysis\n"
                 "Single-choice and Multi-choice answer distributions",
                 fontsize=12, fontweight="bold")

    # OOV rates
    axes[0].bar(["Single-choice\n(200 vocab)","Multi-choice\n(150 vocab)"],
                [s_oov_rate*100, m_oov_rate*100],
                color=["#2196F3","#FF9800"], alpha=0.85,
                edgecolor="white", width=0.4)
    axes[0].set_title("(a) Out-of-Vocabulary Rate",fontweight="bold")
    axes[0].set_ylabel("OOV Rate (%)"); axes[0].set_ylim(0,50)
    axes[0].grid(axis="y",alpha=0.3)
    for i, val in enumerate([s_oov_rate*100, m_oov_rate*100]):
        axes[0].text(i, val+0.5, f"{val:.1f}%",
                     ha="center", fontsize=12, fontweight="bold")

    # Top single answers
    top_names = [t[0][:25] for t in top_s]
    top_vals  = [t[1] for t in top_s]
    axes[1].barh(range(len(top_names)), top_vals[::-1],
                 color="#2196F3", alpha=0.85, edgecolor="white")
    axes[1].set_yticks(range(len(top_names)))
    axes[1].set_yticklabels(top_names[::-1], fontsize=7)
    axes[1].set_title("(b) Top 15 Single-choice Answers\n(test set)",
                      fontweight="bold")
    axes[1].set_xlabel("Count"); axes[1].grid(axis="x",alpha=0.3)

    # Answer length distribution
    all_lens = [len(answers[i].split()) for i in range(len(routes))]
    axes[2].hist(all_lens, bins=20, color="#9C27B0", alpha=0.85,
                 edgecolor="white")
    axes[2].axvline(np.mean(all_lens), color="red", linestyle="--",
                    lw=2, label=f"Mean={np.mean(all_lens):.1f} words")
    axes[2].set_title("(c) Answer Length Distribution\n(all routes)",
                      fontweight="bold")
    axes[2].set_xlabel("Answer length (words)")
    axes[2].set_ylabel("Count"); axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_vocab_coverage.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")

    print(f"   Single-choice OOV rate : {s_oov_rate*100:.1f}%")
    print(f"   Multi-choice  OOV rate : {m_oov_rate*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# 8. END-TO-END EXAMPLES
# ══════════════════════════════════════════════════════════════════════════
def plot_end_to_end_examples(test_ds, model, vocab, device):
    print("\n8️⃣  End-to-End Examples ...")
    from stage4_answer_generation import CFG as S4_CFG

    # Pick one example per route
    selected = {}
    for idx in range(len(test_ds)):
        ex = test_ds[idx]
        r  = ex["route"].item()
        if r not in selected:
            selected[r] = idx
        if len(selected) == 6:
            break

    fig, axes = plt.subplots(6, 3, figsize=(16, 24))
    fig.suptitle("Stage 4: End-to-End Inference Examples\n"
                 "One sample per answer route",
                 fontsize=13, fontweight="bold")

    col_titles = ["Input Image","Question + Route","Answer + Disease Context"]
    for j, t in enumerate(col_titles):
        axes[0,j].set_title(t,fontsize=10,fontweight="bold",pad=8)

    DISEASE_SHORT = [
        "P-Ped","P-Sess","P-Hyp","Esoph","Gastri","UC","Crohn",
        "Barrett","Gas-Ulc","Duo-Ulc","Erosion","Hemor","Diverti",
        "N-Cecum","N-Pylor","N-Z-Line","Ileocec","Retro-R","Retro-S",
        "Dyed-LP","Dyed-RM","For-Body","Instrum"
    ]

    for row, (r, sample_idx) in enumerate(sorted(selected.items())):
        ex      = test_ds[sample_idx]
        raw_img = test_ds.data[sample_idx]["image"].convert("RGB")

        fused   = ex["fused_repr"].unsqueeze(0).to(device)
        disease = ex["disease_vec"].unsqueeze(0).to(device)
        route   = ex["route"].item()

        with torch.no_grad():
            pred_answers = model.predict(fused, disease, route,
                                         vocab, threshold=0.5)
        pred_ans = pred_answers[0]
        true_ans = ex["answer_raw"]

        # Col 0: image
        axes[row,0].imshow(raw_img.resize((224,224)))
        axes[row,0].axis("off")
        axes[row,0].set_xlabel(
            f"Route {route}: {QTYPE_NAMES[route]}",
            fontsize=8)
        axes[row,0].xaxis.set_label_position("bottom")

        # Col 1: question + route info
        axes[row,1].axis("off")
        q = ex["question_raw"]
        q_wrapped = "\n".join([q[i:i+40] for i in range(0,len(q),40)])
        info_text = (
            f"QUESTION:\n{q_wrapped}\n\n"
            f"TRUE ANSWER:\n{true_ans[:60]}\n\n"
            f"PREDICTED:\n{pred_ans[:60]}\n\n"
            f"CORRECT: {'YES ✓' if pred_ans.lower() in true_ans.lower() or true_ans.lower() in pred_ans.lower() else 'NO ✗'}"
        )
        axes[row,1].text(0.05, 0.95, info_text,
                         transform=axes[row,1].transAxes,
                         fontsize=8, va="top",
                         fontfamily="monospace",
                         bbox=dict(boxstyle="round",
                                   facecolor=QTYPE_COLORS[route]+"20",
                                   edgecolor=QTYPE_COLORS[route]))

        # Col 2: disease context
        d_vec = ex["disease_vec"].numpy()
        top5  = np.argsort(d_vec)[::-1][:5]
        axes[row,2].barh(range(5),
                         d_vec[top5][::-1],
                         color=[QTYPE_COLORS[route]]*5,
                         alpha=0.8, edgecolor="white")
        axes[row,2].set_yticks(range(5))
        axes[row,2].set_yticklabels(
            [DISEASE_SHORT[i] for i in top5[::-1]], fontsize=8)
        axes[row,2].set_xlim(0,1.05)
        axes[row,2].set_xlabel("Disease probability", fontsize=8)
        axes[row,2].set_title(
            f"Top 5 active diseases",fontsize=8)
        axes[row,2].axvline(0.5,color="red",linestyle="--",
                            alpha=0.6,lw=1)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_end_to_end_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 9. LATENCY BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════
def plot_latency(model, device):
    print("\n9️⃣  Latency Benchmarks ...")

    batch_sizes = [1, 4, 8, 16, 32, 64]
    route_latencies = {}

    for r in range(6):
        lats = []
        for bs in batch_sizes:
            dummy_f = torch.randn(bs, 512).to(device)
            dummy_d = torch.rand(bs, 23).to(device)
            # warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_f, dummy_d, r)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # measure
            times = []
            for _ in range(30):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(dummy_f, dummy_d, r)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter()-t0)
            lats.append(np.mean(times)*1000)
        route_latencies[QTYPE_NAMES[r]] = lats

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Stage 4: Inference Latency per Head\n"
                 f"Device: {device.upper()}",
                 fontsize=12, fontweight="bold")

    for i, (name, lats) in enumerate(route_latencies.items()):
        axes[0].plot(batch_sizes, lats, "-o", ms=5, lw=2,
                     label=name, color=QTYPE_COLORS[i])
    axes[0].set_title("(a) Latency vs Batch Size",fontweight="bold")
    axes[0].set_xlabel("Batch Size"); axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Per-sample at bs=1
    ps_lats = [route_latencies[n][0] for n in QTYPE_NAMES]
    bars = axes[1].bar(QTYPE_NAMES, ps_lats, color=QTYPE_COLORS,
                       alpha=0.85, edgecolor="white", width=0.6)
    axes[1].set_title("(b) Single-sample Latency per Head",fontweight="bold")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].tick_params(axis="x",rotation=20)
    axes[1].grid(axis="y",alpha=0.3)
    for bar, val in zip(bars, ps_lats):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01,
                     f"{val:.2f}ms", ha="center", fontsize=8)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage4_latency.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")

    df_lat = pd.DataFrame({
        "Batch Size": batch_sizes,
        **{n: [round(l,2) for l in route_latencies[n]]
           for n in QTYPE_NAMES}
    })
    df_lat.to_csv(f"{LOG_DIR}/stage4_latency.csv", index=False)
    print(f"   ✅  {LOG_DIR}/stage4_latency.csv")


# ══════════════════════════════════════════════════════════════════════════
# 10. FULL PIPELINE SUMMARY (all 4 stages)
# ══════════════════════════════════════════════════════════════════════════
def plot_full_pipeline_summary():
    print("\n🔟  Full Pipeline Summary ...")

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Complete 4-Stage VQA Pipeline — Full Results Summary\n"
                 "Thesis: Advancing Medical AI with Explainable VQA on GI Imaging\n"
                 "Author: Ummay Hani Javed (24i-8211)",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Stage accuracy bars
    ax1 = fig.add_subplot(gs[0, :2])
    stages = ["Stage 1\nDisease\nClassifier",
              "Stage 2\nQuestion\nCategoriser",
              "Stage 3\nMultimodal\nFusion",
              "Stage 4\nAnswer\nGeneration"]
    accs   = [96.86, 93.01, 92.33, 73.97]
    f1s    = [96.86, 88.64, 87.47, None]
    colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
    x = np.arange(len(stages)); w = 0.35
    b1 = ax1.bar(x-w/2, accs, w, label="Test Accuracy",
                 color=colors, alpha=0.85, edgecolor="white")
    f1_vals = [f for f in f1s if f is not None]
    ax1.bar(x[:3]+w/2, f1_vals, w, label="Macro-F1/Metric",
            color=colors[:3], alpha=0.5, edgecolor="white",
            hatch="//")
    ax1.set_title("Pipeline Performance — Test Accuracy & Macro-F1",
                  fontweight="bold", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=9)
    ax1.set_ylabel("Score (%)"); ax1.set_ylim(65,104)
    ax1.legend(fontsize=9); ax1.grid(axis="y",alpha=0.3)
    for bar, acc in zip(b1, accs):
        ax1.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.3, f"{acc:.2f}%",
                 ha="center", fontsize=10, fontweight="bold")

    # Stage 4 breakdown
    ax2 = fig.add_subplot(gs[0, 2])
    names4 = [r[0] for r in TEST_RESULTS]
    accs4  = [r[1]*100 for r in TEST_RESULTS]
    ax2.barh(names4, accs4, color=QTYPE_COLORS,
             alpha=0.85, edgecolor="white", height=0.6)
    ax2.axvline(OVERALL_TEST_ACC*100, color="red",
                linestyle="--", alpha=0.7, lw=1.5)
    ax2.set_title("Stage 4\nRoute Breakdown",fontweight="bold",fontsize=10)
    ax2.set_xlabel("Accuracy (%)"); ax2.grid(axis="x",alpha=0.3)
    for i, acc in enumerate(accs4):
        ax2.text(acc+0.5, i, f"{acc:.1f}%", va="center", fontsize=8)

    # Parameter breakdown
    ax3 = fig.add_subplot(gs[0, 3])
    comp = ["ResNet50\n(S1)","DistilBERT\n(S2)",
            "Fusion\n(S3)","Heads\n(S4)"]
    frozen = [23.5, 67.0, 0.0, 0.0]
    train  = [0.7,  0.0,  3.2, 0.9]
    x3 = np.arange(len(comp))
    ax3.bar(x3, frozen, label="Frozen", color="#FFCDD2",
            alpha=0.9, edgecolor="white")
    ax3.bar(x3, train, bottom=frozen, label="Trainable",
            color="#C8E6C9", alpha=0.9, edgecolor="white")
    ax3.set_title("Parameters (M)",fontweight="bold",fontsize=10)
    ax3.set_xticks(x3); ax3.set_xticklabels(comp,fontsize=8)
    ax3.legend(fontsize=8); ax3.grid(axis="y",alpha=0.3)

    # Training epochs
    ax4 = fig.add_subplot(gs[1, 0])
    ep_labs = ["Stage 1","Stage 2","Stage 3","Stage 4"]
    ep_vals = [16, 10, 5, 9]
    bars4 = ax4.bar(ep_labs, ep_vals, color=colors,
                    alpha=0.85, edgecolor="white", width=0.5)
    ax4.set_title("Epochs to Convergence",fontweight="bold",fontsize=10)
    ax4.set_ylabel("Epochs"); ax4.grid(axis="y",alpha=0.3)
    for bar, val in zip(bars4, ep_vals):
        ax4.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.1, str(val),
                 ha="center",fontsize=11,fontweight="bold")

    # Test distribution
    ax5 = fig.add_subplot(gs[1, 1])
    supports = [r[2] for r in TEST_RESULTS]
    ax5.pie(supports, labels=names4, colors=QTYPE_COLORS,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize":7})
    ax5.set_title("Test Set Distribution",fontweight="bold",fontsize=10)

    # Summary table
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis("off")
    table_data = [
        ["Preprocessing","CLAHE+Gamma / DistilBERT","—","—","—"],
        ["Stage 1","ResNet50 + MLP","96.86%","0.9925","16 ep"],
        ["Stage 2","DistilBERT fine-tune","93.01%","0.8864","10 ep"],
        ["Stage 3","CrossAttn+DiseaseGate","92.33%","0.8747","5 ep"],
        ["Stage 4","6 Specialised Heads","73.97%","—","9 ep"],
    ]
    table = ax6.table(
        cellText  = table_data,
        colLabels = ["Stage","Model","Test Acc","Macro-F1","Epochs"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.2)
    for (i,j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
        elif i > 0 and j == 2:
            cell.set_facecolor("#E8F5E9")
    ax6.set_title("Complete Pipeline Summary Table",
                  fontweight="bold", fontsize=11, pad=20)

    path = f"{LOG_DIR}/stage4_full_pipeline_summary.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"   ✅  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 11. ALL CSV TABLES
# ══════════════════════════════════════════════════════════════════════════
def save_all_tables():
    print("\n📋  Saving all CSV tables ...")

    # Epoch log
    df1 = pd.DataFrame(EPOCH_LOG,
          columns=["Epoch","Tr Loss","Tr Acc","Val Acc"])
    df1.to_csv(f"{LOG_DIR}/stage4_epoch_log.csv",index=False)
    print(f"   ✅  stage4_epoch_log.csv")

    # Per-route results
    df2 = pd.DataFrame(TEST_RESULTS,
          columns=["Route","Test Acc","Support"])
    df2["Error Rate"] = 1-df2["Test Acc"]
    df2.to_csv(f"{LOG_DIR}/stage4_results_table.csv",index=False)
    print(f"   ✅  stage4_results_table.csv")

    # Route val per epoch
    df3 = pd.DataFrame(ROUTE_VAL)
    df3.insert(0,"Epoch",range(1,10))
    df3.to_csv(f"{LOG_DIR}/stage4_route_val_per_epoch.csv",index=False)
    print(f"   ✅  stage4_route_val_per_epoch.csv")

    # Full pipeline
    pipeline = [
        ["Preprocessing","CLAHE+Gamma+Aug / DistilBERT tok","—","—","—","—"],
        ["Stage 1","ResNet50+MLP (Disease Classifier)","24.2M","0.7M","96.86%","99.25% F1"],
        ["Stage 2","DistilBERT (Question Categoriser)","67.0M","67.0M","93.01%","88.64% F1"],
        ["Stage 3","CrossAttn+DiseaseGate+FusionMLP","93.7M","3.2M","92.33%","87.47% F1"],
        ["Stage 4","6 Specialised Answer Heads","0.9M","0.9M","73.97%","—"],
    ]
    df4 = pd.DataFrame(pipeline,
          columns=["Stage","Model","Total Params","Trainable",
                   "Test Acc","Best Metric"])
    df4.to_csv(f"{LOG_DIR}/full_pipeline_results.csv",index=False)
    print(f"   ✅  full_pipeline_results.csv")

    # Print summary
    print("\n" + "="*72)
    print("COMPLETE PIPELINE — TEST RESULTS")
    print("="*72)
    print(df4.to_string(index=False))
    print("\nStage 4 Per-Route:")
    print(df2.to_string(index=False))
    print("="*72)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_inference", action="store_true",
        help="Skip plots that need model inference (faster)")
    args = parser.parse_args()

    print("\n📊  Stage 4 COMPLETE Extended Analysis\n" + "="*60)

    save_all_tables()
    plot_training_dynamics()
    plot_test_performance()
    plot_full_pipeline_summary()

    if not args.skip_inference:
        print("\n   Loading model and data for inference-based plots ...")
        model, test_ds, loader, vocab, device, CFG = load_everything()
        print("   Running inference ...")
        results = run_inference(model, loader, device)
        print(f"   Done — {len(results['routes']):,} samples")

        plot_confusion_matrices(results)
        plot_confidence_analysis(results)
        plot_error_analysis(results)
        plot_disease_influence(results)
        plot_vocab_coverage(results, vocab)
        plot_end_to_end_examples(test_ds, model, vocab, device)
        plot_latency(model, device)
    else:
        print("\n   Skipped inference-based plots (--skip_inference flag)")

    print("\n" + "="*60)
    print("✅  ALL Stage 4 outputs saved to ./logs/\n")
    print("  TRAINING:")
    print("    stage4_training_dynamics.png     — 6-panel full training")
    print("  PERFORMANCE:")
    print("    stage4_test_performance.png      — per-route bars + scatter")
    print("    stage4_full_pipeline_summary.png — all 4 stages overview")
    print("  INFERENCE-BASED (requires --no skip_inference):")
    print("    stage4_confusion_matrices.png    — per-route CM")
    print("    stage4_confidence_analysis.png   — confidence histograms")
    print("    stage4_error_analysis.png        — error rates + wrong conf")
    print("    stage4_error_rates.png           — error rate bar chart")
    print("    stage4_disease_influence.png     — disease heatmap + table")
    print("    stage4_vocab_coverage.png        — OOV + answer distribution")
    print("    stage4_end_to_end_examples.png   — 6 real examples")
    print("    stage4_latency.png               — latency benchmarks")
    print("    stage4_latency.csv               — latency numbers")
    print("  TABLES:")
    print("    stage4_epoch_log.csv             — 9-epoch training log")
    print("    stage4_results_table.csv         — per-route test results")
    print("    stage4_route_val_per_epoch.csv   — route×epoch val accuracy")
    print("    full_pipeline_results.csv        — complete pipeline table")
