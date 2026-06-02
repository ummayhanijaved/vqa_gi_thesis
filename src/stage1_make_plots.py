#!/usr/bin/env python3
"""Stage 1 plot generator — reads real epoch CSV + runs model on test set."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
SRC     = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)

OUT = os.path.join(PROJECT, "figures", "stage1_plots")
os.makedirs(OUT, exist_ok=True)
EPOCH_CSV = os.path.join(PROJECT, "results", "stage1_epoch_log.csv")

import stage1_disease_classifier as s1
import torch
from sklearn.metrics import f1_score, multilabel_confusion_matrix, roc_auc_score


def plot_training_curves():
    if not os.path.exists(EPOCH_CSV):
        print(f"  [skip] {EPOCH_CSV} not found.")
        return
    df = pd.read_csv(EPOCH_CSV)
    print(f"  epoch log columns: {list(df.columns)}")
    ep = df["epoch"] if "epoch" in df.columns else df[df.columns[0]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for c in df.columns:
        cl = c.lower()
        if "loss" in cl and "train" in cl:
            ax[0].plot(ep, df[c], "o-", label="Train loss", color="tab:blue")
        if "loss" in cl and ("val" in cl or "valid" in cl):
            ax[0].plot(ep, df[c], "s-", label="Val loss", color="tab:red")
    ax[0].set_title("Stage 1 - Loss per Epoch"); ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("BCE loss"); ax[0].grid(alpha=0.3); ax[0].legend()
    for c in df.columns:
        cl = c.lower()
        if "macro" in cl and "f1" in cl:
            ax[1].plot(ep, df[c], "o-", label="Val macro-F1", color="tab:green")
        elif "micro" in cl and "f1" in cl:
            ax[1].plot(ep, df[c], "s-", label="Val micro-F1", color="tab:purple")
    ax[1].set_title("Stage 1 - F1 per Epoch"); ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1"); ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout()
    p = os.path.join(OUT, "stage1_training_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {p}")


def run_test_inference():
    labels = s1.DISEASE_LABELS
    device = s1.CFG["device"]
    model = s1.TreeNetDiseaseClassifier().to(device)
    ckpt_path = os.path.join(PROJECT, "checkpoints", "stage1_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  loaded checkpoint (best_f1={ckpt.get('best_f1','?')})")

    from datasets import load_from_disk
    from datasets import Image as HFImage
    import torch.utils.data as tud

    raw = load_from_disk(os.path.expanduser("~/vqa_gi_thesis/data/kvasir_local"))
    test_hf = raw["test"] if "test" in raw else raw["train"]
    try:
        test_hf = test_hf.cast_column("image", HFImage())
    except Exception:
        pass

    ds = s1.DiseaseClassificationDataset(test_hf, "test")
    loader = tud.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            out = model(imgs)
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    probs = np.concatenate(all_probs); y = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)

    from sklearn.metrics import f1_score
    macro = f1_score(y, preds, average="macro", zero_division=0)
    micro = f1_score(y, preds, average="micro", zero_division=0)
    print(f"  TEST macro-F1={macro:.4f}  micro-F1={micro:.4f}  (n={len(y)} images)")
    return labels, y, probs, preds


def plot_per_class_f1(labels, y, preds):
    f1s = f1_score(y, preds, average=None, zero_division=0)
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(labels)[order], f1s[order], color="teal")
    ax.set_xlabel("F1-score"); ax.set_xlim(0, 1.05)
    ax.set_title("Stage 1 - Per-Class F1 (test set)")
    for i, v in enumerate(f1s[order]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3); plt.tight_layout()
    p = os.path.join(OUT, "stage1_per_class_f1.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {p}")


def plot_confusion_grid(labels, y, preds):
    mcm = multilabel_confusion_matrix(y, preds)
    n = len(labels); cols = 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.4))
    axes = axes.flatten()
    for i in range(n):
        cm = mcm[i]; ax = axes[i]; ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > cm.max()/2 else "black")
        ax.set_title(labels[i], fontsize=7)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["N","P"], fontsize=6); ax.set_yticklabels(["N","P"], fontsize=6)
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.suptitle("Stage 1 - Per-Class Confusion Matrices (multi-label)", fontsize=12, y=1.005)
    plt.tight_layout()
    p = os.path.join(OUT, "stage1_confusion_grid.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {p}")


def plot_support_auc(labels, y, probs):
    support = y.sum(axis=0).astype(int); aucs = []
    for i in range(len(labels)):
        try:
            aucs.append(roc_auc_score(y[:, i], probs[:, i]) if y[:, i].sum() > 0 else np.nan)
        except Exception:
            aucs.append(np.nan)
    aucs = np.array(aucs); order = np.argsort(support)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].barh(np.array(labels)[order], support[order], color="tab:orange")
    ax[0].set_title("Per-Class Support (test set)"); ax[0].set_xlabel("# positive")
    ax[0].grid(axis="x", alpha=0.3)
    ax[1].barh(np.array(labels)[order], aucs[order], color="tab:blue")
    ax[1].set_title("Per-Class ROC-AUC"); ax[1].set_xlim(0, 1.05)
    ax[1].set_xlabel("ROC-AUC"); ax[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT, "stage1_per_class_support.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {p}")


def main():
    print(f"\nStage 1 plots -> {OUT}\n")
    plot_training_curves()
    res = run_test_inference()
    if res is not None:
        labels, y, probs, preds = res
        plot_per_class_f1(labels, y, preds)
        plot_confusion_grid(labels, y, preds)
        plot_support_auc(labels, y, probs)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
