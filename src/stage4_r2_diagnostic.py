#!/usr/bin/env python3
"""
Stage 4 Route 2 (multi-choice) diagnostic.
Shows raw prediction vs ground_truth pairs and how the multi-label F1
parser splits them, so we can see if 6.31% is real or a format mismatch.
USAGE: python src/stage4_r2_diagnostic.py
"""
import os, pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
path = os.path.join(PROJECT, "logs", "stage4_revised", "route2_multi_choice_eval.csv")
df = pd.read_csv(path)
print(f"Loaded {len(df)} rows from {os.path.basename(path)}")
print(f"Columns: {list(df.columns)}\n")

def parse(s):
    return set(t.strip().lower() for t in str(s).split(",") if t.strip())

print("="*90)
print("FIRST 15 PAIRS — how the comma-split multi-label parser sees them:")
print("="*90)
exact_fmt_match = 0
some_overlap = 0
for i in range(min(15, len(df))):
    p = df.iloc[i]["prediction"]
    g = df.iloc[i]["ground_truth"]
    ps, gs = parse(p), parse(g)
    overlap = ps & gs
    print(f"\n[{i}]")
    print(f"  PRED raw: {p!r}")
    print(f"  GT   raw: {g!r}")
    print(f"  PRED set ({len(ps)}): {ps}")
    print(f"  GT   set ({len(gs)}): {gs}")
    print(f"  overlap : {overlap if overlap else 'NONE'}")

# aggregate stats over all rows
tot_overlap = 0
tot_pred_tokens = 0
tot_gt_tokens = 0
rows_with_any_overlap = 0
for _, r in df.iterrows():
    ps, gs = parse(r["prediction"]), parse(r["ground_truth"])
    ov = ps & gs
    tot_overlap += len(ov)
    tot_pred_tokens += len(ps)
    tot_gt_tokens += len(gs)
    if ov: rows_with_any_overlap += 1

print("\n" + "="*90)
print("AGGREGATE over all", len(df), "rows:")
print(f"  Avg tokens per PRED: {tot_pred_tokens/len(df):.2f}")
print(f"  Avg tokens per GT  : {tot_gt_tokens/len(df):.2f}")
print(f"  Rows with ANY token overlap: {rows_with_any_overlap}/{len(df)} "
      f"({rows_with_any_overlap/len(df)*100:.1f}%)")
print("="*90)
print("\nINTERPRETATION:")
print(" - If PRED sets look like full SENTENCES (not comma-separated findings),")
print("   the comma-split is wrong -> 6.31% is an EVAL ARTIFACT, fixable.")
print(" - If PRED sets are short findings but rarely overlap GT findings,")
print("   then 6.31% is REAL -> the route genuinely underperforms.\n")
