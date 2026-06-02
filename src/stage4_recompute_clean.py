#!/usr/bin/env python3
"""
Stage 4 — Clean re-evaluation of ALL routes from saved eval CSVs.
Uses the SAME compute_metrics() logic from stage4_revised_analysis.py
(proper normalization per route) and prints one honest table.

This recomputes from the raw prediction,ground_truth pairs already on disk
in logs/stage4_revised/. Nothing invented.

USAGE: python src/stage4_recompute_clean.py
"""
import os, sys
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
SRC = os.path.join(PROJECT, "src"); sys.path.insert(0, SRC)
LOGDIR = os.path.join(PROJECT, "logs", "stage4_revised")

import stage4_revised_analysis as A

ROUTE_FILES = {
    0: "route0_yes_no_eval.csv",
    1: "route1_single_choice_eval.csv",
    2: "route2_multi_choice_eval.csv",
    3: "route3_color_eval.csv",
    4: "route4_location_yolo_eval.csv",
    5: "route5_count_yolo_eval.csv",
}

def load_preds(route):
    path = os.path.join(LOGDIR, ROUTE_FILES[route])
    if not os.path.exists(path):
        print(f"  [route {route}] MISSING: {path}")
        return None, None
    df = pd.read_csv(path)
    # columns may be prediction,ground_truth(,correct,...)
    p = df["prediction"].fillna("").astype(str).tolist()
    g = df["ground_truth"].fillna("").astype(str).tolist()
    return p, g

print("\n" + "="*78)
print("  STAGE 4 — CLEAN RE-EVALUATION (recomputed from saved eval CSVs)")
print("="*78)
print(f"{'Route':<22}{'Metric':<16}{'Value':>10}{'N':>10}")
print("-"*78)

results = {}
for route in range(6):
    p, g = load_preds(route)
    if p is None:
        continue
    overall, df = A.compute_metrics(p, g, route)
    if overall is None:
        print(f"  route {route}: no metrics")
        continue
    if route == 2:
        metric_name = "sample-F1"
        val = overall["sample_f1"]
    elif route in (4, 5):
        metric_name = "fuzzy-acc"
        val = overall["accuracy"]
    else:
        metric_name = "accuracy"
        val = overall["accuracy"]
    results[route] = (overall["route_name"], metric_name, val,
                      overall["n_total"])
    print(f"{overall['route_name']:<22}{metric_name:<16}"
          f"{val*100:>9.2f}%{overall['n_total']:>10}")

print("-"*78)
print("\nThese are the HONEST, recomputed numbers from your saved predictions.")
print("Compare to the hard-coded thesis values: 88.65/36.70/84.20/81.71/54.80/70.00\n")
