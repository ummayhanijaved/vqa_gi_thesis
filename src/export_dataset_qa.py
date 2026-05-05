"""
Export all Q&A pairs from Kvasir-VQA-x1 dataset to CSV.
Includes: split, image_id, question, answer, question_type (route)

USAGE:
    python export_dataset_qa.py

OUTPUT:
    ~/vqa_gi_thesis/results/dataset_all_qa.csv
"""

import os, sys
import pandas as pd
from datasets import load_from_disk

HOME    = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "vqa_gi_thesis")
sys.path.insert(0, os.path.join(PROJECT, "src"))

# ── Route classifier (same logic as stage2) ──────────────────────────────────
def classify_route(question: str) -> str:
    q = question.lower().strip()
    yn_keywords   = ["is ", "are ", "was ", "were ", "has ", "have ",
                     "does ", "do ", "did ", "can ", "could ", "will "]
    color_kw      = ["colour", "color", "coloured", "colored"]
    location_kw   = ["where", "location", "located", "position", "region",
                     "quadrant", "area", "part of"]
    count_kw      = ["how many", "count", "number of", "size", "millimeter",
                     "mm", "how large", "how big"]
    multi_kw      = ["what findings", "which findings", "what abnormalities",
                     "which abnormalities", "what features", "what is visible",
                     "what can be seen", "what are visible"]

    if any(q.startswith(kw) for kw in yn_keywords):
        return "Yes/No"
    for kw in color_kw:
        if kw in q:
            return "Color"
    for kw in location_kw:
        if kw in q:
            return "Location"
    for kw in count_kw:
        if kw in q:
            return "Count"
    for kw in multi_kw:
        if kw in q:
            return "Multi-Choice"
    return "Single-Choice"

# ── Load dataset ──────────────────────────────────────────────────────────────
data_dir = os.path.join(HOME, "data", "kvasir_local")
print(f"Loading dataset from: {data_dir}")
dataset  = load_from_disk(data_dir)
print(f"Splits available: {list(dataset.keys())}")

rows = []
for split_name, split_data in dataset.items():
    print(f"Processing [{split_name}]: {len(split_data):,} samples ...")
    for i, sample in enumerate(split_data):
        question = sample.get("question", sample.get("Question", ""))
        answer   = sample.get("answer",   sample.get("Answer",   ""))
        img_id   = sample.get("image_id", sample.get("img_id",
                   sample.get("image",    f"img_{i}")))

        # If image_id is not a string (e.g. it's the actual image), use index
        if not isinstance(img_id, str):
            img_id = f"{split_name}_{i:06d}"

        rows.append({
            "split"        : split_name,
            "index"        : i,
            "image_id"     : str(img_id),
            "question"     : question.strip(),
            "answer"       : answer.strip().lower(),
            "question_type": classify_route(question),
        })

df = pd.DataFrame(rows)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Total Q&A pairs  : {len(df):,}")
print(f"  Splits           : {df['split'].value_counts().to_dict()}")
print(f"\n  Question type breakdown:")
for qtype, count in df["question_type"].value_counts().items():
    pct = 100 * count / len(df)
    print(f"    {qtype:<20} {count:>7,}  ({pct:.1f}%)")
print(f"{'='*55}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(PROJECT, "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "dataset_all_qa.csv")
df.to_csv(out_path, index=False)
print(f"\n✅  Saved {len(df):,} rows → {out_path}")

# Also save one CSV per question type for easy reference
type_dir = os.path.join(out_dir, "qa_by_type")
os.makedirs(type_dir, exist_ok=True)
for qtype in df["question_type"].unique():
    sub = df[df["question_type"] == qtype]
    fname = qtype.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    sub.to_csv(os.path.join(type_dir, f"qa_{fname}.csv"), index=False)
    print(f"   Saved {len(sub):,} rows → qa_by_type/qa_{fname}.csv")

print(f"\nDone!")
