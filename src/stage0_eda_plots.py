#!/usr/bin/env python3
"""
Stage 0 — EDA + preprocessing visualisations from REAL Kvasir-VQA-x1 data.
- Images loaded from LOCAL disk: data/kvasir_raw/images/<img_id>.jpg
- Question-type plot uses your REAL Stage 2 infer_label() (authoritative 6-way)
USAGE: python src/stage0_eda_plots.py
Requires: datasets, matplotlib, opencv-python, numpy, pillow
"""
import os, textwrap, random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
SRC = os.path.join(PROJECT, "src"); 
import sys; sys.path.insert(0, SRC)
OUT = os.path.join(PROJECT, "figures", "stage0"); os.makedirs(OUT, exist_ok=True)
IMGDIR = os.path.join(PROJECT, "data", "kvasir_raw", "images")

from datasets import load_dataset
print("Loading Kvasir-VQA-x1 ...")
raw = load_dataset("SimulaMet/Kvasir-VQA-x1")
train = raw["train"]
print(f"  train size: {len(train):,}")

# ---- local image loader ----
def load_img(img_id, size=None):
    p = os.path.join(IMGDIR, f"{img_id}.jpg")
    im = Image.open(p).convert("RGB")
    if size: im = im.resize(size)
    return im

# ---- REAL 6-way category labels via your stage2 logic ----
USE_REAL = True
try:
    import stage2_question_categorizer as s2
    ID2LABEL = s2.ID2LABEL
    def cat(q, a): return ID2LABEL[s2.infer_label(q, a)]
    print("  using REAL s2.infer_label() for categories")
except Exception as e:
    USE_REAL = False
    print(f"  [warn] could not import stage2 ({e}); falling back to answer heuristic")
    YES_NO = {"yes","no"}
    def cat(q, a):
        a=str(a).strip().lower()
        if a in YES_NO or a.startswith("yes") or a.startswith("no"): return "yes/no"
        if "," in a and len(a)<200: return "multiple-choice"
        return "single-choice"

# ============ 1. QUESTION-TYPE (6 categories, real logic) ============
N = min(len(train), 50000)
print(f"  categorising {N:,} samples ...")
cats = [cat(train[i]["question"], train[i]["answer"]) for i in range(N)]
s = pd.Series(cats).value_counts()
plt.figure(figsize=(7,4))
s.plot(kind="bar", color="#2962FF", edgecolor="black")
plt.title(f"Question-Type Distribution ({N:,} sample)" + ("" if USE_REAL else " [heuristic]"))
plt.ylabel("Count"); plt.xlabel("Question category")
plt.xticks(rotation=25, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT}/eda_question_types.pdf"); plt.close()
print("  saved eda_question_types.png\n", s)

# ============ 2. COMPLEXITY ============
comp = pd.Series([train[i]["complexity"] for i in range(len(train))]).value_counts().sort_index()
plt.figure(figsize=(5,4))
comp.plot(kind="bar", color="#00897B", edgecolor="black")
plt.title("Question Complexity Distribution (train)")
plt.ylabel("Count"); plt.xlabel("Complexity level"); plt.tight_layout()
plt.savefig(f"{OUT}/eda_complexity.pdf"); plt.close()
print("  saved eda_complexity.pdf")

# ============ 3. ANSWER LENGTH ============
ans_len = [len(str(train[i]["answer"]).split()) for i in range(min(len(train),30000))]
plt.figure(figsize=(6,4))
plt.hist(ans_len, bins=40, color="#7C4DFF", edgecolor="black")
plt.title("Answer Length Distribution (words)")
plt.xlabel("Words per answer"); plt.ylabel("Frequency"); plt.tight_layout()
plt.savefig(f"{OUT}/eda_answer_length.pdf"); plt.close()
print("  saved eda_answer_length.pdf")

# ============ 4. SAMPLE GRID (local images) ============
random.seed(0)
idxs = random.sample(range(len(train)), 6)
fig, axes = plt.subplots(2,3, figsize=(13,8))
for ax,i in zip(axes.ravel(), idxs):
    ex = train[i]
    try: ax.imshow(load_img(ex["img_id"]))
    except Exception as e: ax.text(0.5,0.5,f"[load failed]\n{e}",ha="center",va="center",fontsize=7)
    ax.axis("off")
    ax.set_title(textwrap.fill(str(ex["question"]),38), fontsize=8)
    ax.text(0.5,-0.08,textwrap.fill("A: "+str(ex["answer"]),38),
            transform=ax.transAxes, fontsize=8, ha="center", va="top", color="#003")
fig.suptitle("Kvasir-VQA-x1 --- Sample Image / Question / Answer Triplets", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/sample_grid.pdf"); plt.close()
print("  saved sample_grid.pdf")

# ============ 5. PREPROCESSING PIPELINE ============
import cv2
img = np.array(load_img(train[idxs[0]]["img_id"], size=(224,224)))
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])
img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
mean_int = img_clahe.mean()/255.0
gamma = 0.7 if mean_int>0.7 else (1.5 if mean_int<0.3 else 1.0)
img_gamma = (255*((img_clahe/255.0)**gamma)).astype(np.uint8)
img_aug = np.clip(cv2.flip(img_gamma,1)*1.1,0,255).astype(np.uint8)
stages=[("Original",img),("+ CLAHE",img_clahe),(f"+ Gamma ({gamma})",img_gamma),("+ Augment",img_aug)]
fig,axes=plt.subplots(1,4,figsize=(14,4))
for ax,(t,im) in zip(axes,stages):
    ax.imshow(im); ax.set_title(t,fontsize=11); ax.axis("off")
fig.suptitle("Stage 0 --- Image Preprocessing Pipeline", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/preproc_pipeline.pdf"); plt.close()
print("  saved preproc_pipeline.pdf")

print("\nAll Stage 0 figures saved to", OUT)
