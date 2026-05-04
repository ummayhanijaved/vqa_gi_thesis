"""
Stage 2 — Extended Analysis for Thesis Reporting
Run AFTER training is complete:
    python stage2_analysis.py
Generates all tables and figures needed for the thesis chapter.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Load training history ──────────────────────────────────────────────────
LOG_DIR  = "./logs"
CKPT_DIR = "./checkpoints/best_model"

with open(f"{LOG_DIR}/history.json") as f:
    history = json.load(f)

CLASS_NAMES = ["yes/no", "single-choice", "multiple-choice",
               "color", "location", "numerical count"]

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Training Curves (publication quality, 3-panel)
# ══════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Stage 2: DistilBERT Question Type Classifier — Training Dynamics",
                 fontsize=13, fontweight="bold", y=1.02)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", ms=5, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", ms=5, label="Validation")
    axes[0].axvline(x=5, color="green", linestyle="--", alpha=0.6, label="Best epoch (5)")
    axes[0].set_title("(a) Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(list(epochs))

    # Accuracy
    axes[1].plot(epochs, [v*100 for v in history["train_acc"]], "b-o", ms=5, label="Train")
    axes[1].plot(epochs, [v*100 for v in history["val_acc"]],   "r-o", ms=5, label="Validation")
    axes[1].axvline(x=5, color="green", linestyle="--", alpha=0.6, label="Best epoch (5)")
    axes[1].axhline(y=92.77, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_title("(b) Token Accuracy (%)", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(list(epochs))

    # Macro-F1
    axes[2].plot(epochs, history["val_f1"], "g-o", ms=5, label="Val Macro-F1")
    axes[2].axvline(x=5, color="green", linestyle="--", alpha=0.6, label="Best epoch (5)")
    axes[2].set_title("(c) Validation Macro-F1", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0.85, 0.92)
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_xticks(list(epochs))

    plt.tight_layout()
    path = f"{LOG_DIR}/fig_training_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Confusion Matrix (normalised + raw counts side-by-side)
# ══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_dataset
    from sklearn.metrics import confusion_matrix

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained(CKPT_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(CKPT_DIR).to(device)
    model.eval()

    # Reload test data
    def infer_label(q, a):
        q, a = q.lower().strip(), a.lower().strip()
        if any(w in q for w in ("color","colour","what color","what colour")): return 3
        if any(w in q for w in ("where","location","located","position",
                                 "which part","which area","which region","which side")): return 4
        if any(w in q for w in ("how many","count","number of","total number")): return 5
        if a in {"yes","no"} or a.startswith("yes") or a.startswith("no"): return 0
        if "," in a and len(a) < 200: return 2
        return 1

    raw     = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir="./data")
    records = [{"question": ex["question"], "label": infer_label(ex["question"], ex["answer"])}
               for ex in raw["test"]]
    test_df = pd.DataFrame(records)

    class SimpleDS(Dataset):
        def __init__(self, df):
            self.q = df["question"].tolist(); self.l = df["label"].tolist()
        def __len__(self): return len(self.q)
        def __getitem__(self, i):
            enc = tokenizer(self.q[i], max_length=128, padding="max_length",
                            truncation=True, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "label": self.l[i]}

    loader = DataLoader(SimpleDS(test_df), batch_size=64, shuffle=False, num_workers=2)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device)).logits
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"])

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    short = ["Y/N", "Single", "Multi", "Color", "Loc.", "Count"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 2: Confusion Matrices on Test Set (n=15,955)",
                 fontsize=13, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short, ax=axes[0])
    axes[0].set_title("(a) Raw Counts", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=short, yticklabels=short,
                vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("(b) Normalised (row %)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    path = f"{LOG_DIR}/fig_confusion_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved: {path}")
    return cm, cm_norm, all_preds, all_labels

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-class F1 bar chart
# ══════════════════════════════════════════════════════════════════════════
def plot_per_class_f1():
    f1_scores   = [0.89, 0.83, 0.60, 1.00, 1.00, 1.00]
    precision   = [0.91, 0.82, 0.53, 1.00, 1.00, 1.00]
    recall      = [0.88, 0.84, 0.69, 1.00, 1.00, 1.00]
    support     = [4657, 2618, 377, 1912, 3192, 3199]
    colors      = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336","#00BCD4"]

    x = np.arange(len(CLASS_NAMES))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 2: Per-Class Performance — Test Set",
                 fontsize=13, fontweight="bold")

    bars = axes[0].bar(x, f1_scores, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    axes[0].set_title("(a) F1-Score per Class", fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    axes[0].set_ylabel("F1-Score"); axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=0.8864, color="red", linestyle="--", alpha=0.7, label="Macro-F1=0.886")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, f1_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(x - w, precision, w, label="Precision", color="#2196F3", alpha=0.8)
    axes[1].bar(x,     recall,    w, label="Recall",    color="#4CAF50", alpha=0.8)
    axes[1].bar(x + w, f1_scores, w, label="F1-Score",  color="#FF9800", alpha=0.8)
    axes[1].set_title("(b) Precision / Recall / F1 per Class", fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1.1)
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{LOG_DIR}/fig_per_class_f1.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Class distribution bar chart
# ══════════════════════════════════════════════════════════════════════════
def plot_class_distribution():
    data = {
        "Class"         : CLASS_NAMES * 3,
        "Count"         : [33250,19009,2867,13730,23153,22866,
                            8313, 4752, 716, 3433, 5788, 5717,
                            4657, 2618, 377, 1912, 3192, 3199],
        "Split"         : ["Train"]*6 + ["Val"]*6 + ["Test"]*6,
    }
    df  = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    x   = np.arange(len(CLASS_NAMES))
    w   = 0.25
    splits = ["Train", "Val", "Test"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (split, color) in enumerate(zip(splits, colors)):
        counts = df[df["Split"]==split]["Count"].values
        ax.bar(x + i*w - w, counts, w, label=split, color=color, alpha=0.85)

    ax.set_title("Stage 2: Class Distribution Across Splits",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Number of Samples"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{LOG_DIR}/fig_class_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE — Full results summary (printed + saved as CSV)
# ══════════════════════════════════════════════════════════════════════════
def print_results_table():
    rows = [
        ["yes/no",           0.91, 0.88, 0.89, 4657],
        ["single-choice",    0.82, 0.84, 0.83, 2618],
        ["multiple-choice",  0.53, 0.69, 0.60,  377],
        ["color",            1.00, 1.00, 1.00, 1912],
        ["location",         1.00, 1.00, 1.00, 3192],
        ["numerical count",  1.00, 1.00, 1.00, 3199],
        ["— macro avg —",    0.88, 0.90, 0.89, 15955],
        ["— weighted avg —", 0.93, 0.93, 0.93, 15955],
    ]
    df = pd.DataFrame(rows, columns=["Class","Precision","Recall","F1","Support"])
    print("\n" + "="*65)
    print("STAGE 2 TEST SET RESULTS  (DistilBERT fine-tuned, 8 epochs)")
    print("="*65)
    print(df.to_string(index=False))
    print(f"\n  Overall Accuracy : 93.01%")
    print(f"  Macro-F1         : 0.8864")
    print(f"  Best Val Acc     : 92.77%  (Epoch 5, early stopping at 8)")
    print("="*65 + "\n")
    df.to_csv(f"{LOG_DIR}/results_table.csv", index=False)
    print(f"✅  Saved: {LOG_DIR}/results_table.csv")

# ══════════════════════════════════════════════════════════════════════════
# TRAINING EPOCH TABLE
# ══════════════════════════════════════════════════════════════════════════
def print_epoch_table():
    tr_loss = history["train_loss"]
    va_loss = history["val_loss"]
    tr_acc  = history["train_acc"]
    va_acc  = history["val_acc"]
    va_f1   = history["val_f1"]

    print("\n" + "="*75)
    print("EPOCH-BY-EPOCH TRAINING LOG")
    print(f"{'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Va Loss':>8}  {'Va Acc':>7}  {'Va F1':>7}  {'Note'}")
    print("-"*75)
    best = max(va_acc)
    for i, (trl, vl, tra, va, f1) in enumerate(
            zip(tr_loss, va_loss, tr_acc, va_acc, va_f1), 1):
        note = "← best" if va == best else ""
        print(f"{i:>5}  {trl:>8.4f}  {tra*100:>6.2f}%  {vl:>8.4f}  {va*100:>6.2f}%  {f1:>7.4f}  {note}")
    print("="*75 + "\n")

    # Save as CSV for thesis
    epoch_df = pd.DataFrame({
        "Epoch"       : range(1, len(tr_loss)+1),
        "Train Loss"  : tr_loss,
        "Train Acc"   : [round(v*100,2) for v in tr_acc],
        "Val Loss"    : va_loss,
        "Val Acc"     : [round(v*100,2) for v in va_acc],
        "Val Macro-F1": va_f1,
    })
    epoch_df.to_csv(f"{LOG_DIR}/epoch_log.csv", index=False)
    print(f"✅  Saved: {LOG_DIR}/epoch_log.csv")

# ══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    print("\n📊  Generating all thesis figures and tables …\n")

    print_epoch_table()
    print_results_table()
    plot_training_curves()
    plot_per_class_f1()
    plot_class_distribution()
    plot_confusion_matrices()   # this re-runs inference on test set

    print("\n✅  All outputs saved to ./logs/")
    print("    fig_training_curves.png   — training dynamics (3-panel)")
    print("    fig_per_class_f1.png      — per-class F1 bar chart")
    print("    fig_class_distribution.png — dataset class balance")
    print("    fig_confusion_matrix.png  — normalised + raw CM")
    print("    results_table.csv         — precision/recall/F1 table")
    print("    epoch_log.csv             — full epoch-by-epoch log")
