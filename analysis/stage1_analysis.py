"""
=============================================================================
Stage 1 — Extended Analysis for Thesis Reporting
Run AFTER training is complete:
    python stage1_analysis.py
Generates all figures and tables needed for the thesis chapter.
=============================================================================
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

LOG_DIR  = "./logs"
CKPT_DIR = "./checkpoints/stage1_best.pt"
os.makedirs(LOG_DIR, exist_ok=True)

DISEASE_LABELS = [
    "polyp-pedunculated", "polyp-sessile", "polyp-hyperplastic",
    "esophagitis", "gastritis", "ulcerative-colitis", "crohns-disease",
    "barretts-esophagus", "gastric-ulcer", "duodenal-ulcer",
    "erosion", "hemorrhoid", "diverticulum",
    "normal-cecum", "normal-pylorus", "normal-z-line",
    "ileocecal-valve", "retroflex-rectum", "retroflex-stomach",
    "dyed-lifted-polyp", "dyed-resection-margins",
    "foreign-body", "instrument"
]

SHORT_LABELS = [
    "P-Ped.", "P-Sess.", "P-Hyp.",
    "Esoph.", "Gastri.", "UC", "Crohn's",
    "Barrett's", "Gas-Ulc.", "Duo-Ulc.",
    "Erosion", "Hemor.", "Diverti.",
    "N-Cecum", "N-Pylor.", "N-Z-Line",
    "Ileocec.", "Retro-R.", "Retro-S.",
    "Dyed-LP", "Dyed-RM",
    "For-Body", "Instrum."
]

# ── Hard-coded results from training output ──────────────────────────────────
EPOCH_LOG = [
    (1,  0.3512, 0.7360, 0.1420, 0.9546, 0.9322),
    (2,  0.1130, 0.9510, 0.0809, 0.9812, 0.9472),
    (3,  0.0728, 0.9708, 0.0743, 0.9858, 0.9477),
    (4,  0.0717, 0.9736, 0.0716, 0.9879, 0.9502),
    (5,  0.0584, 0.9785, 0.0723, 0.9884, 0.9520),
    (6,  0.0531, 0.9817, 0.0730, 0.9897, 0.9527),
    (7,  0.0485, 0.9825, 0.0725, 0.9901, 0.9522),
    (8,  0.0452, 0.9844, 0.0741, 0.9907, 0.9501),
    (9,  0.0349, 0.9878, 0.0731, 0.9911, 0.9513),
    (10, 0.0399, 0.9859, 0.0731, 0.9911, 0.9507),
    (11, 0.0337, 0.9877, 0.0724, 0.9918, 0.9526),
    (12, 0.0357, 0.9888, 0.0733, 0.9916, 0.9532),
    (13, 0.0338, 0.9880, 0.0773, 0.9925, 0.9512),
    (14, 0.0271, 0.9908, 0.0780, 0.9922, 0.9490),
    (15, 0.0368, 0.9886, 0.0779, 0.9923, 0.9493),
    (16, 0.0292, 0.9892, 0.0760, 0.9925, 0.9508),
    (17, 0.0291, 0.9905, 0.0791, 0.9924, 0.9510),
    (18, 0.0281, 0.9912, 0.0779, 0.9922, 0.9507),
    (19, 0.0299, 0.9894, 0.0765, 0.9918, 0.9501),
    (20, 0.0308, 0.9896, 0.0785, 0.9920, 0.9499),
]

# Per-class test results (precision, recall, f1, support)
PER_CLASS = [
    ("polyp-pedunculated",    0.97, 1.00, 0.98, 3709),
    ("polyp-sessile",         0.97, 1.00, 0.98, 3709),
    ("polyp-hyperplastic",    0.97, 1.00, 0.98, 3709),
    ("esophagitis",           0.95, 1.00, 0.97, 3635),
    ("gastritis",             0.95, 1.00, 0.97, 3635),
    ("ulcerative-colitis",    0.95, 1.00, 0.97, 3635),
    ("crohns-disease",        0.95, 1.00, 0.97, 3635),
    ("barretts-esophagus",    0.95, 1.00, 0.97, 3635),
    ("gastric-ulcer",         0.95, 1.00, 0.97, 3635),
    ("duodenal-ulcer",        0.95, 1.00, 0.97, 3635),
    ("erosion",               0.95, 1.00, 0.97, 3635),
    ("hemorrhoid",            0.94, 1.00, 0.97, 3586),
    ("diverticulum",          0.94, 1.00, 0.97, 3586),
    ("normal-cecum",          0.92, 1.00, 0.96, 3524),
    ("normal-pylorus",        0.92, 1.00, 0.96, 3524),
    ("normal-z-line",         0.92, 1.00, 0.96, 3524),
    ("ileocecal-valve",       0.92, 1.00, 0.96, 3524),
    ("retroflex-rectum",      0.92, 1.00, 0.96, 3524),
    ("retroflex-stomach",     0.92, 1.00, 0.96, 3524),
    ("dyed-lifted-polyp",     0.93, 1.00, 0.96, 3546),
    ("dyed-resection-margins",0.93, 1.00, 0.96, 3546),
    ("foreign-body",          0.89, 1.00, 0.94, 3387),
    ("instrument",            0.96, 1.00, 0.98, 3681),
]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training Dynamics (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    df   = pd.DataFrame(EPOCH_LOG,
           columns=["epoch","tr_loss","tr_f1","va_loss","va_f1","va_auc"])
    eps  = df["epoch"]
    best = int(df.loc[df["va_f1"].idxmax(), "epoch"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Stage 1: TreeNet Disease Classifier — Training Dynamics",
                 fontsize=13, fontweight="bold", y=1.01)

    # Loss
    axes[0].plot(eps, df["tr_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(eps, df["va_loss"], "r-o", ms=4, label="Validation")
    axes[0].axvline(best, color="green", linestyle="--", alpha=0.7,
                    label=f"Best (Ep.{best})")
    axes[0].set_title("(a) BCE Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Macro-F1
    axes[1].plot(eps, df["tr_f1"], "b-o", ms=4, label="Train Macro-F1")
    axes[1].plot(eps, df["va_f1"], "g-o", ms=4, label="Val Macro-F1")
    axes[1].axvline(best, color="green", linestyle="--", alpha=0.7)
    axes[1].axhline(0.9925, color="gray", linestyle=":", alpha=0.6)
    axes[1].annotate("0.9925", xy=(best, 0.9925),
                     xytext=(best+0.5, 0.991), fontsize=8, color="gray")
    axes[1].set_title("(b) Macro-F1 Score", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.70, 1.01)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # AUC
    axes[2].plot(eps, df["va_auc"], "c-o", ms=4, label="Val AUC-ROC")
    axes[2].axvline(best, color="green", linestyle="--", alpha=0.7)
    axes[2].set_title("(c) Validation AUC-ROC", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC-ROC")
    axes[2].set_ylim(0.90, 0.96)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage1_training_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Per-Class Precision / F1 Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_performance():
    names = [r[0] for r in PER_CLASS]
    prec  = [r[1] for r in PER_CLASS]
    rec   = [r[2] for r in PER_CLASS]
    f1    = [r[3] for r in PER_CLASS]
    x     = np.arange(len(names))

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle("Stage 1: Per-Class Performance on Test Set (n=82,683 labels)",
                 fontsize=13, fontweight="bold")

    # F1 bar chart
    colors = ["#2196F3" if v >= 0.97 else
              "#4CAF50" if v >= 0.96 else
              "#FF9800" if v >= 0.95 else
              "#F44336" for v in f1]
    bars = axes[0].bar(x, f1, color=colors, alpha=0.85, edgecolor="white",
                       linewidth=0.8, width=0.7)
    axes[0].axhline(0.9686, color="red", linestyle="--", alpha=0.7,
                    label="Macro-F1=0.9686")
    axes[0].set_title("(a) F1-Score per Disease Class", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SHORT_LABELS, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.88, 1.03)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, f1):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")

    # Grouped P/R/F1
    w = 0.25
    axes[1].bar(x - w, prec, w, label="Precision", color="#2196F3", alpha=0.8)
    axes[1].bar(x,     rec,  w, label="Recall",    color="#4CAF50", alpha=0.8)
    axes[1].bar(x + w, f1,   w, label="F1-Score",  color="#FF9800", alpha=0.8)
    axes[1].set_title("(b) Precision / Recall / F1 per Disease Class",
                      fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SHORT_LABELS, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.85, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage1_per_class_performance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Precision grouped by disease category
# ══════════════════════════════════════════════════════════════════════════════
def plot_disease_groups():
    groups = {
        "Polyp\n(3 classes)"      : [0.97, 0.97, 0.97],
        "Inflammatory\n(8 classes)": [0.95]*8,
        "Hemorrhoid &\nDiverti. (2)": [0.94, 0.94],
        "Normal\nAnatomy (6)"     : [0.92]*6,
        "Dyed\nPolyp (2)"         : [0.93, 0.93],
        "Foreign Body\n& Instr.(2)": [0.89, 0.96],
    }
    group_names  = list(groups.keys())
    group_mean_p = [np.mean(v) for v in groups.values()]
    group_mean_f1 = [0.98, 0.97, 0.97, 0.96, 0.96, 0.96]  # from report
    colors = ["#2196F3","#E91E63","#9C27B0","#4CAF50","#FF9800","#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 1: Performance by Disease Category",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(group_names))
    bars = axes[0].bar(x, group_mean_p, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.8)
    axes[0].set_title("(a) Mean Precision by Category", fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(group_names, fontsize=9)
    axes[0].set_ylabel("Mean Precision"); axes[0].set_ylim(0.85, 1.01)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, group_mean_p):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    bars2 = axes[1].bar(x, group_mean_f1, color=colors, alpha=0.85,
                        edgecolor="white", linewidth=0.8)
    axes[1].set_title("(b) Mean F1-Score by Category", fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(group_names, fontsize=9)
    axes[1].set_ylabel("Mean F1-Score"); axes[1].set_ylim(0.85, 1.01)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, group_mean_f1):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = f"{LOG_DIR}/stage1_disease_groups.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — 23-Class Confusion Matrix (multi-label → per-class binary)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix():
    """
    For multi-label classification, builds a 23×1 summary heatmap
    showing per-class recall (diagonal of each binary CM).
    Then runs actual inference to get true confusion data.
    """
    import torch
    import torchvision.transforms as T
    from datasets import load_from_disk

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import model from stage1 script
    import sys
    sys.path.insert(0, os.path.expanduser("~"))
    from stage1_disease_classifier import (
        TreeNetDiseaseClassifier, DiseaseClassificationDataset,
        extract_disease_labels, CFG
    )

    print("   Loading model from checkpoint …")
    model = TreeNetDiseaseClassifier().to(device)
    ckpt  = torch.load(CKPT_DIR, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("   Loading test dataset …")
    raw  = load_from_disk("./data/kvasir_local")
    test_ds = DiseaseClassificationDataset(raw["test"], "test")
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["labels"]
            with torch.cuda.amp.autocast():
                out = model(imgs)
            preds = (out["probs"].cpu() > 0.5).float()
            all_preds.append(preds)
            all_labels.append(labels)

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # Per-class TP/FP/FN/TN
    tp = (preds_np * labels_np).sum(0)
    fp = (preds_np * (1-labels_np)).sum(0)
    fn = ((1-preds_np) * labels_np).sum(0)
    tn = ((1-preds_np) * (1-labels_np)).sum(0)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # ── Heatmap: precision & recall side by side ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7, 10))
    fig.suptitle("Stage 1: Per-Class Binary Classification Summary\n"
                 "(Multi-label — each disease evaluated independently)",
                 fontsize=11, fontweight="bold")

    metrics = np.stack([precision, recall, f1], axis=1)   # (23, 3)

    for ax, (col_idx, col_name, cmap) in zip(
            axes,
            [(0, "Precision", "Blues"), (1, "Recall", "Greens")]):
        data = metrics[:, col_idx:col_idx+1]
        sns.heatmap(data, ax=ax,
                    annot=True, fmt=".3f", cmap=cmap,
                    vmin=0.85, vmax=1.0,
                    xticklabels=[col_name],
                    yticklabels=SHORT_LABELS,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.6})
        ax.set_title(f"({chr(97+col_idx)}) {col_name}", fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = f"{LOG_DIR}/stage1_per_class_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  {path}")

    # ── Full precision/recall summary bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(23)
    w = 0.35
    ax.bar(x - w/2, precision, w, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, recall,    w, label="Recall",    color="#4CAF50", alpha=0.85)
    ax.axhline(precision.mean(), color="blue",  linestyle="--",
               alpha=0.6, label=f"Mean Prec={precision.mean():.3f}")
    ax.axhline(recall.mean(),    color="green", linestyle="--",
               alpha=0.6, label=f"Mean Rec={recall.mean():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0.85, 1.03)
    ax.set_title("Stage 1: Precision & Recall per Disease Class — Test Set",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = f"{LOG_DIR}/stage1_precision_recall_bars.png"
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  {path2}")

    return preds_np, labels_np


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Disease Vector Visualisation (sample from test set)
# ══════════════════════════════════════════════════════════════════════════════
def plot_disease_vector_example():
    """
    Shows what the 23-D output disease probability vector looks like
    for a real test image — key figure for thesis methodology section.
    """
    import torch
    import torchvision.transforms as T
    from datasets import load_from_disk
    from stage1_disease_classifier import TreeNetDiseaseClassifier, CFG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TreeNetDiseaseClassifier().to(device)
    ckpt   = torch.load(CKPT_DIR, map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    tfm = T.Compose([
        T.Resize((224,224)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    raw = load_from_disk("./data/kvasir_local")
    # Pick 3 diverse examples
    indices = [0, 500, 1500]
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, 5*len(indices)))
    fig.suptitle("Stage 1: 23-D Disease Probability Vector Output Examples",
                 fontsize=13, fontweight="bold")

    for row, idx in enumerate(indices):
        ex    = raw["test"][idx]
        img   = ex["image"].convert("RGB")
        tensor= tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
        probs = out["probs"][0].cpu().numpy()

        # Left: image
        axes[row, 0].imshow(img.resize((224,224)))
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"Input Image (Test #{idx})\n"
                               f"Q: {ex['question'][:60]}",
                               fontsize=8)

        # Right: disease vector bar chart
        colors = ["#F44336" if p >= 0.5 else "#90CAF9" for p in probs]
        bars   = axes[row, 1].barh(range(23), probs, color=colors,
                                    edgecolor="white", linewidth=0.5)
        axes[row, 1].axvline(0.5, color="red", linestyle="--",
                             alpha=0.7, linewidth=1.5, label="Threshold=0.5")
        axes[row, 1].set_yticks(range(23))
        axes[row, 1].set_yticklabels(SHORT_LABELS, fontsize=7)
        axes[row, 1].set_xlabel("Probability")
        axes[row, 1].set_xlim(0, 1.05)
        axes[row, 1].set_title(f"23-D Disease Probability Vector  d = [p₁...p₂₃]",
                               fontsize=9, fontweight="bold")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(axis="x", alpha=0.3)
        # Annotate active diseases
        active = [SHORT_LABELS[i] for i, p in enumerate(probs) if p >= 0.5]
        axes[row, 1].text(0.52, 22,
                          f"Active: {', '.join(active) if active else 'None'}",
                          fontsize=7, color="darkred", va="top")

    plt.tight_layout()
    path = f"{LOG_DIR}/stage1_disease_vector_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
def print_tables():
    # Epoch log
    df = pd.DataFrame(EPOCH_LOG,
         columns=["Epoch","Tr Loss","Tr F1","Val Loss","Val Macro-F1","Val AUC"])
    df.to_csv(f"{LOG_DIR}/stage1_epoch_log.csv", index=False)
    print(f"✅  {LOG_DIR}/stage1_epoch_log.csv")

    # Per-class results
    df2 = pd.DataFrame(PER_CLASS,
          columns=["Class","Precision","Recall","F1","Support"])
    df2.to_csv(f"{LOG_DIR}/stage1_results_table.csv", index=False)

    print("\n" + "="*72)
    print("STAGE 1 TEST SET RESULTS  (TreeNet, frozen ResNet50 + MLP head)")
    print("="*72)
    print(df2.to_string(index=False))
    print(f"\n  Test Macro-F1  : 0.9686")
    print(f"  Test Micro-F1  : 0.9687")
    print(f"  Test AUC-ROC   : 0.7550")
    print(f"  Test Exact Acc : 0.8903  (89.03%)")
    print(f"  Best Val F1    : 0.9925  (Epoch 16)")
    print("="*72)
    print(f"✅  {LOG_DIR}/stage1_results_table.csv")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_inference", action="store_true",
        help="Skip confusion matrix (requires model+data). Use for quick plots only.")
    args = parser.parse_args()

    print("\n📊  Generating Stage 1 thesis figures …\n")
    print_tables()
    plot_training_curves()
    plot_per_class_performance()
    plot_disease_groups()

    if not args.skip_inference:
        print("\n   Running inference for confusion heatmap (needs GPU + data) …")
        plot_confusion_matrix()
        plot_disease_vector_example()
    else:
        print("\n   Skipped inference plots (--skip_inference flag set)")

    print("\n✅  All Stage 1 outputs saved to ./logs/")
    print("    stage1_training_curves.png        — loss / F1 / AUC per epoch")
    print("    stage1_per_class_performance.png  — F1 + grouped P/R/F1 bars")
    print("    stage1_disease_groups.png         — performance by disease category")
    print("    stage1_per_class_heatmap.png      — precision & recall heatmap (23 classes)")
    print("    stage1_precision_recall_bars.png  — P/R bars with mean lines")
    print("    stage1_disease_vector_examples.png— 23-D vector output examples")
    print("    stage1_epoch_log.csv              — full training log")
    print("    stage1_results_table.csv          — per-class test results")
