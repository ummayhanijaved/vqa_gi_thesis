#!/usr/bin/env python3
"""
Stage 4 — Additional Analysis Plots

Adds three thesis-grade analyses on top of the existing dashboard:
  1. ROC curves for Route 0 (Yes/No) and Route 2 (Multi-Choice)
  2. Per-disease accuracy heatmap for Route 0 (uses Stage 3 disease vec)
  3. Inference latency benchmarks per route

Outputs PNG files to:
  ~/vqa_gi_thesis/logs/stage4_revised/analysis/extra/

Usage:
  python stage4_extra_analysis.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Reuse setup from main pipeline
SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

from stage4_revised import (
    CFG, ROUTE_NAMES, DistilBERTAnswerModel,
    DistilBERTRouteDataset, FusionExtractor, TextPreprocessor,
    cache_stage3_features, infer_route, build_vocab,
    YOLOLocationModel, YOLOCountModel,
)

OUT_DIR = os.path.expanduser(
    "~/vqa_gi_thesis/logs/stage4_revised/analysis/extra")
os.makedirs(OUT_DIR, exist_ok=True)

# Disease vocab — must match Stage 1 / Stage 3
DISEASES = [
    "polyp", "ulcerative-colitis", "esophagitis-a", "esophagitis-b-d",
    "barretts", "barretts-short-segment", "hemorrhoids", "ileum",
    "cecum", "pylorus", "z-line", "retroflex-stomach", "retroflex-rectum",
    "bbps-0-1", "bbps-2-3", "impacted-stool", "dyed-lifted-polyps",
    "dyed-resection-margins", "instruments", "normal-cecum",
    "normal-pylorus", "normal-z-line", "other",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROC CURVES for Routes 0 and 2
# ─────────────────────────────────────────────────────────────────────────────
def compute_roc_route(route: int, model, tokenizer, vocab, test_records):
    """Run inference + collect probabilities for ROC curve."""
    from torch.utils.data import DataLoader

    test_ds = DistilBERTRouteDataset(
        test_records, route, tokenizer, vocab, CFG["max_input_len"])
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            fused   = batch["fused"].to(CFG["device"])
            disease = batch["disease"].to(CFG["device"])
            inp_ids = batch["input_ids"].to(CFG["device"])
            att_msk = batch["attention_mask"].to(CFG["device"])
            labels  = batch["label"]

            cls_repr = model._encode(fused, disease, inp_ids, att_msk)
            logits   = model.heads[str(route)](cls_repr)

            if route == 0:
                # Binary: probability of "yes" class
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
            elif route == 2:
                # Multi-label: probability per class
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return probs, labels


def plot_roc_route0(probs, labels):
    """ROC curve for binary Yes/No classification."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color="#2E75B6", lw=2.5,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--",
             label="Random (AUC = 0.500)")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#2E75B6")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — Route 0 (Yes/No)\nDistilBERT", fontsize=13,
              fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "route0_roc_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Route 0 ROC saved → {out_path}  (AUC={roc_auc:.4f})")
    return roc_auc


def plot_roc_route2(probs, labels):
    """Macro-averaged ROC curve for multi-label Multi-Choice."""
    from sklearn.metrics import roc_curve, auc

    n_classes = probs.shape[1]
    plt.figure(figsize=(8, 7))

    # Per-class ROC for top 5 most-supported classes + macro average
    class_supports = labels.sum(axis=0)
    top_classes = np.argsort(class_supports)[::-1][:5]

    aucs = []
    for cls_idx in range(n_classes):
        if labels[:, cls_idx].sum() < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(labels[:, cls_idx], probs[:, cls_idx])
            cls_auc = auc(fpr, tpr)
            aucs.append(cls_auc)
            if cls_idx in top_classes:
                plt.plot(fpr, tpr, lw=1.5, alpha=0.7,
                         label=f"Class {cls_idx} (AUC={cls_auc:.3f})")
        except Exception:
            continue

    macro_auc = np.mean(aucs) if aucs else 0.0
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--",
             label="Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves — Route 2 (Multi-Choice)\n"
              f"Top 5 classes shown   |   Macro AUC = {macro_auc:.3f}",
              fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "route2_roc_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Route 2 ROC saved → {out_path}  (macro AUC={macro_auc:.4f})")
    return macro_auc


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-DISEASE ACCURACY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_disease_heatmap(test_records):
    """
    Build heatmap of accuracy per (disease × route) pair.
    Uses the disease one-hot from Stage 3 features in the cache.
    """
    print("\n  Building per-disease × per-route accuracy matrix ...")

    # Load all 4 saved eval CSVs and Route 4/5 yolo eval CSVs
    eval_dir = os.path.expanduser("~/vqa_gi_thesis/logs/stage4_revised")
    csv_files = {
        0: "route0_yes_no_eval.csv",
        1: "route1_single_choice_eval.csv",
        2: "route2_multi_choice_eval.csv",
        3: "route3_color_eval.csv",
        4: "route4_location_yolo_eval.csv",
        5: "route5_count_yolo_eval.csv",
    }

    # Index test records by their position so we can join with CSV results
    # (CSV preserves original test order)
    matrix = np.full((len(DISEASES), 6), np.nan)
    counts = np.zeros((len(DISEASES), 6), dtype=int)

    for route, fname in csv_files.items():
        path = os.path.join(eval_dir, fname)
        if not os.path.exists(path):
            print(f"    ⚠️   Missing {fname} — skipping route {route}")
            continue
        df = pd.read_csv(path)
        n_rows = len(df)

        # Get disease vec for the corresponding test records
        # Filter test_records to those for this route
        route_records = [r for r in test_records
                         if infer_route(r.get("question", "")) == route]
        n_route = min(n_rows, len(route_records))

        per_disease_correct = {d: 0 for d in range(len(DISEASES))}
        per_disease_total   = {d: 0 for d in range(len(DISEASES))}

        for i in range(n_route):
            rec = route_records[i]
            disease_vec = rec.get("disease", rec.get("disease_vec"))
            if disease_vec is None:
                continue
            if hasattr(disease_vec, "numpy"):
                disease_vec = disease_vec.numpy()
            disease_vec = np.asarray(disease_vec).flatten()
            if disease_vec.size != len(DISEASES):
                continue

            # Find the dominant disease (argmax of one-hot)
            disease_idx = int(np.argmax(disease_vec))

            pred = str(df.iloc[i]["prediction"]).lower().strip()
            gt   = str(df.iloc[i]["ground_truth"]).lower().strip()

            # Simple correctness: pred substring or exact in gt
            correct = (pred == gt) or (pred and pred in gt)

            per_disease_total[disease_idx]   += 1
            per_disease_correct[disease_idx] += int(correct)

        for d_idx in range(len(DISEASES)):
            if per_disease_total[d_idx] > 0:
                matrix[d_idx, route] = (per_disease_correct[d_idx] /
                                         per_disease_total[d_idx])
                counts[d_idx, route] = per_disease_total[d_idx]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix * 100, cmap="RdYlGn", aspect="auto",
                    vmin=0, vmax=100)

    # Annotate
    for i in range(len(DISEASES)):
        for j in range(6):
            if counts[i, j] > 0:
                val = matrix[i, j] * 100
                color = "white" if val < 40 or val > 75 else "black"
                ax.text(j, i, f"{val:.0f}%\n(n={counts[i, j]})",
                        ha="center", va="center", fontsize=7,
                        color=color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="gray")

    ax.set_xticks(range(6))
    ax.set_xticklabels([f"R{r}\n{ROUTE_NAMES[r][:8]}" for r in range(6)],
                       fontsize=10)
    ax.set_yticks(range(len(DISEASES)))
    ax.set_yticklabels(DISEASES, fontsize=9)
    ax.set_xlabel("Question Route", fontsize=12)
    ax.set_ylabel("GI Disease / Finding", fontsize=12)
    ax.set_title("Per-Disease Test Accuracy Heatmap\n"
                  "Stage 4 — All 6 Routes × 23 GI Diseases",
                  fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "per_disease_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Heatmap saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFERENCE LATENCY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_latency(test_records):
    """Measure single-sample inference time per route."""
    from transformers import DistilBertTokenizerFast

    print("\n  Benchmarking inference latency per route ...")
    results = []

    # ── DistilBERT routes (0, 1, 2, 3) ───────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        DistilBERTAnswerModel.MODEL_NAME)

    for route in [0, 1, 2, 3]:
        ckpt_path = os.path.join(
            CFG["ckpt_dir"],
            f"stage4_revised_{ROUTE_NAMES[route]}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"    ⚠️   No checkpoint for route {route}, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=CFG["device"],
                           weights_only=False)
        vocab = ckpt["vocab"]
        model = DistilBERTAnswerModel(vocab_per_route={route: vocab})
        model.load_state_dict(ckpt["model_state"])
        model = model.to(CFG["device"]).eval()

        # Filter to this route's records
        route_recs = [r for r in test_records
                      if infer_route(r.get("question", "")) == route][:50]
        if not route_recs:
            continue

        # Build a single batch of size 1 for true latency
        ds = DistilBERTRouteDataset(
            route_recs, route, tokenizer, vocab, CFG["max_input_len"])

        # Warm up
        for i in range(3):
            sample = ds[i]
            with torch.no_grad():
                _ = model._encode(
                    sample["fused"].unsqueeze(0).to(CFG["device"]),
                    sample["disease"].unsqueeze(0).to(CFG["device"]),
                    sample["input_ids"].unsqueeze(0).to(CFG["device"]),
                    sample["attention_mask"].unsqueeze(0).to(CFG["device"]),
                )

        # Measure
        times = []
        for i in range(min(50, len(ds))):
            sample = ds[i]
            torch.cuda.synchronize() if CFG["device"] == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                cls = model._encode(
                    sample["fused"].unsqueeze(0).to(CFG["device"]),
                    sample["disease"].unsqueeze(0).to(CFG["device"]),
                    sample["input_ids"].unsqueeze(0).to(CFG["device"]),
                    sample["attention_mask"].unsqueeze(0).to(CFG["device"]),
                )
                _ = model.heads[str(route)](cls)
            torch.cuda.synchronize() if CFG["device"] == "cuda" else None
            times.append((time.perf_counter() - t0) * 1000)  # ms

        results.append({
            "route"   : route,
            "name"    : ROUTE_NAMES[route],
            "model"   : "DistilBERT",
            "mean_ms" : float(np.mean(times)),
            "p50_ms"  : float(np.percentile(times, 50)),
            "p95_ms"  : float(np.percentile(times, 95)),
            "n_samples": len(times),
        })
        print(f"    Route {route} ({ROUTE_NAMES[route]:<15}): "
              f"mean={np.mean(times):6.2f} ms  p95={np.percentile(times, 95):6.2f} ms")

        del model
        torch.cuda.empty_cache()

    # ── YOLO routes (4, 5) ────────────────────────────────────────────────────
    image_dir = CFG["image_dir"]
    if os.path.isdir(image_dir):
        sample_images = [os.path.join(image_dir, f)
                         for f in os.listdir(image_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))][:30]

        for route, ModelCls in [(4, YOLOLocationModel),
                                  (5, YOLOCountModel)]:
            ft_path = os.path.join(
                CFG["ckpt_dir"],
                "yolo_seg_finetuned" if route == 4 else "yolo_det_finetuned",
                "weights", "best.pt")
            yolo = ModelCls(weights_path=ft_path
                              if os.path.exists(ft_path) else None)
            if not yolo.available:
                continue

            # Warm up
            for img in sample_images[:3]:
                yolo.predict(img)

            times = []
            for img in sample_images:
                t0 = time.perf_counter()
                yolo.predict(img)
                times.append((time.perf_counter() - t0) * 1000)

            results.append({
                "route"   : route,
                "name"    : ROUTE_NAMES[route],
                "model"   : "YOLO-Seg" if route == 4 else "YOLO-Det",
                "mean_ms" : float(np.mean(times)),
                "p50_ms"  : float(np.percentile(times, 50)),
                "p95_ms"  : float(np.percentile(times, 95)),
                "n_samples": len(times),
            })
            print(f"    Route {route} ({ROUTE_NAMES[route]:<15}): "
                  f"mean={np.mean(times):6.2f} ms  "
                  f"p95={np.percentile(times, 95):6.2f} ms")

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "inference_latency.csv")
    df.to_csv(csv_path, index=False)

    # Bar chart
    if results:
        fig, ax = plt.subplots(figsize=(11, 6))
        routes  = [f"R{r['route']}\n{r['name'][:10]}" for r in results]
        means   = [r["mean_ms"] for r in results]
        p95s    = [r["p95_ms"] for r in results]
        colors  = ["#2E75B6" if r["model"].startswith("DistilBERT")
                    else "#E67E22" for r in results]

        x = np.arange(len(routes))
        bars1 = ax.bar(x - 0.2, means, 0.4, label="Mean (ms)",
                        color=colors, alpha=0.8, edgecolor="black")
        bars2 = ax.bar(x + 0.2, p95s, 0.4, label="P95 (ms)",
                        color=colors, alpha=0.4, edgecolor="black",
                        hatch="//")

        for b, v in zip(bars1, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(routes, fontsize=10)
        ax.set_ylabel("Latency (milliseconds)", fontsize=12)
        ax.set_title("Stage 4 Per-Route Inference Latency\n"
                      f"NVIDIA RTX 5070  |  Single-sample inference  |  "
                      f"50 trials per route",
                      fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, "inference_latency.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅  Latency benchmark saved → {out_path}")
        print(f"  ✅  Latency CSV saved       → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print(f"📊  Stage 4 Extra Analysis")
    print(f"{'='*65}")
    print(f"  Output dir: {OUT_DIR}\n")

    # Load shared resources once
    print("  Loading test features and records ...")
    from datasets import load_from_disk

    raw = load_from_disk(CFG["data_dir"])
    print(f"  Dataset splits: {list(raw.keys())}")

    fusion_extractor = FusionExtractor(CFG["stage3_ckpt"])
    text_processor   = TextPreprocessor()

    # We only need the test split for analysis
    test_records = cache_stage3_features(
        fusion_extractor, text_processor,
        raw["test"], "test", CFG["cache_dir"])

    # ── 1. ROC curves for Routes 0 and 2 ──────────────────────────────────────
    print(f"\n[1/3]  ROC Curves")
    print(f"{'-'*45}")
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        DistilBERTAnswerModel.MODEL_NAME)

    for route in [0, 2]:
        ckpt_path = os.path.join(
            CFG["ckpt_dir"],
            f"stage4_revised_{ROUTE_NAMES[route]}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️   No checkpoint for route {route}")
            continue

        ckpt = torch.load(ckpt_path, map_location=CFG["device"],
                           weights_only=False)
        vocab = ckpt["vocab"]
        model = DistilBERTAnswerModel(vocab_per_route={route: vocab})
        model.load_state_dict(ckpt["model_state"])
        model = model.to(CFG["device"]).eval()

        probs, labels = compute_roc_route(
            route, model, tokenizer, vocab, test_records)

        if route == 0:
            plot_roc_route0(probs, labels)
        else:
            plot_roc_route2(probs, labels)

        del model
        torch.cuda.empty_cache()

    # ── 2. Per-disease heatmap ────────────────────────────────────────────────
    print(f"\n[2/3]  Per-Disease Accuracy Heatmap")
    print(f"{'-'*45}")
    plot_disease_heatmap(test_records)

    # ── 3. Inference latency ──────────────────────────────────────────────────
    print(f"\n[3/3]  Inference Latency Benchmark")
    print(f"{'-'*45}")
    benchmark_latency(test_records)

    print(f"\n{'='*65}")
    print(f"✅  All extra analyses complete")
    print(f"{'='*65}")
    print(f"\n  Output files in: {OUT_DIR}")
    print(f"  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"    {f:<35}  {size:>7.1f} KB")
    print()


if __name__ == "__main__":
    main()