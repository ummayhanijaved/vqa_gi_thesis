"""
Inspect Route 1 eval CSV to see what's in preds vs gts.
Run: python debug_route1.py
"""
import pandas as pd
import os

csv_path = os.path.expanduser(
    "~/vqa_gi_thesis/logs/stage4_revised/route1_single_choice_eval.csv")
df = pd.read_csv(csv_path)

print(f"Total rows: {len(df)}")
print()
print("First 15 predictions vs ground truths:")
print("=" * 80)
for i in range(min(15, len(df))):
    p = str(df.iloc[i]['prediction']).strip()
    g = str(df.iloc[i]['ground_truth']).strip()
    p_in_g = p.lower() in g.lower() if p else False
    match = "✅" if p_in_g else "❌"
    print(f"{match} Pred: {p[:35]:<35} | GT: {g[:40]}")

print()
print("Unique prediction tokens (top 20):")
print("=" * 80)
print(df['prediction'].value_counts().head(20))

print()
print("Length statistics:")
print(f"  Avg pred length: {df['prediction'].astype(str).str.len().mean():.1f} chars")
print(f"  Avg GT length:   {df['ground_truth'].astype(str).str.len().mean():.1f} chars")

# Check substring match rate manually
preds = df['prediction'].astype(str).tolist()
gts   = df['ground_truth'].astype(str).tolist()
strict = sum(1 for p, g in zip(preds, gts) if p.strip().lower() == g.strip().lower())
substr = sum(1 for p, g in zip(preds, gts)
             if p.strip() and p.strip().lower() in g.strip().lower())
print()
print(f"Strict match:    {strict}/{len(df)} = {100*strict/len(df):.2f}%")
print(f"Substring match: {substr}/{len(df)} = {100*substr/len(df):.2f}%")