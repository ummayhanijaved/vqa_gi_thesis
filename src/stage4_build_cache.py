#!/usr/bin/env python3
"""
=============================================================================
PHASE 1 — Build Stage 4 Cache (for Stage 5 retraining)
=============================================================================

PURPOSE:
    Run original Phase 2 Stage 4 models on all 159,549 samples (train + val
    + test) and save predictions to disk. This cache will then be used to
    train Stage 5 (T5) without re-running Stage 4 every epoch.

SAFETY:
    - Reads existing Stage 3 cache files (read-only)
    - Reads existing Stage 4 trained checkpoints (read-only)
    - Writes NEW files to: ~/vqa_gi_thesis/cache/stage4_predictions/
    - NEVER overwrites any existing checkpoint or eval CSV

OUTPUT FILES (NEW):
    ~/vqa_gi_thesis/cache/stage4_predictions/
      ├── stage4_cache_train.pt  (~129K records)
      ├── stage4_cache_val.pt    (~14K records)
      └── stage4_cache_test.pt   (~16K records)

USAGE:
    python stage4_build_cache.py --split all
    python stage4_build_cache.py --split train
    python stage4_build_cache.py --split val
    python stage4_build_cache.py --split test
=============================================================================
"""
import os
import sys
import time
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

from transformers import DistilBertModel, DistilBertTokenizerFast
from stage4_revised import (
    CFG as S4_CFG, ROUTE_NAMES, infer_route,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/vqa_gi_thesis")
CFG = {
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size":  64,
    "stage3_cache_dir": os.path.join(PROJECT, "cache", "stage3_features"),
    "stage4_cache_dir": os.path.join(PROJECT, "cache", "stage4_predictions"),
    "data_dir":    S4_CFG["data_dir"],
    "image_dir":   S4_CFG.get("image_dir", ""),
    "ckpt_dir":    None,  # auto-detected
}
os.makedirs(CFG["stage4_cache_dir"], exist_ok=True)

# Auto-detect Stage 4 ckpt directory
for cand in [os.path.join(PROJECT, "checkpoints", "stage4_revised"),
              S4_CFG.get("ckpt_dir", "")]:
    if os.path.exists(os.path.join(cand, "stage4_revised_yes_no_best.pt")):
        CFG["ckpt_dir"] = cand
        break

if not CFG["ckpt_dir"]:
    print(f"❌  Could not find Stage 4 checkpoints")
    sys.exit(1)

print(f"✅  Stage 4 ckpt dir: {CFG['ckpt_dir']}")

S4_CHECKPOINTS = {
    0: "stage4_revised_yes_no_best.pt",
    1: "stage4_revised_single_choice_best.pt",
    2: "stage4_revised_multi_choice_best.pt",
    3: "stage4_revised_color_best.pt",
}
YOLO_SEG_CKPT = os.path.join(CFG["ckpt_dir"], "yolo_seg_finetuned",
                              "weights", "best.pt")
YOLO_DET_CKPT = os.path.join(CFG["ckpt_dir"], "yolo_det_finetuned",
                              "weights", "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 DistilBERT model (matches your original architecture)
# ─────────────────────────────────────────────────────────────────────────────
class Stage3Projector(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(S4_CFG["head_input_dim"], hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, fused, disease):
        x = torch.cat([fused, disease], dim=-1)
        return self.proj(x).unsqueeze(1)


class DistilBERTRouteModel(nn.Module):
    HIDDEN = 768
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, n_classes):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(self.MODEL_NAME)
        self.projector = Stage3Projector(self.HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN // 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(self.HIDDEN // 2, n_classes),
        )

    def forward(self, fused, disease, input_ids, attention_mask):
        emb = self.distilbert.embeddings
        word_emb = emb.word_embeddings(input_ids)
        prefix = self.projector(fused, disease).to(word_emb.dtype)
        combined = torch.cat([prefix, word_emb], dim=1)
        pos_ids = torch.arange(combined.size(1), dtype=torch.long,
                                device=combined.device).unsqueeze(0)
        combined = combined + emb.position_embeddings(pos_ids)
        combined = emb.LayerNorm(combined); combined = emb.dropout(combined)
        prefix_mask = torch.ones(fused.size(0), 1,
                                  dtype=attention_mask.dtype,
                                  device=attention_mask.device)
        ext_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        try:
            out = self.distilbert.transformer(
                hidden_states=combined, attn_mask=ext_mask)
        except TypeError:
            try:
                out = self.distilbert.transformer(combined, attn_mask=ext_mask)
            except TypeError:
                out = self.distilbert.transformer(x=combined, attn_mask=ext_mask)
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        return self.head(hidden[:, 1, :])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapping the cache records
# ─────────────────────────────────────────────────────────────────────────────
class CacheRouteDataset(Dataset):
    def __init__(self, records, route, tokenizer, max_len=128):
        self.records = [r for r in records if r.get("route") == route]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        fused = r.get("fused_repr", r.get("fused"))
        disease = r.get("disease_vec", r.get("disease"))
        if fused.dim() == 1: fused = fused
        if disease.dim() == 1: disease = disease
        enc = self.tokenizer(r["question"], return_tensors="pt",
                              max_length=self.max_len,
                              padding="max_length", truncation=True)
        return {
            "fused": fused,
            "disease": disease,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "_idx": idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference functions per route
# ─────────────────────────────────────────────────────────────────────────────
def run_distilbert_routes(records, route_models_vocab, tokenizer):
    """Run Routes 0, 1, 2, 3 in batches using Stage 3 cache."""
    predictions = {}  # idx → prediction string

    for route, (model, vocab) in route_models_vocab.items():
        # Find records for this route
        route_indices = [i for i, r in enumerate(records)
                          if r.get("route") == route]
        if not route_indices:
            print(f"  Route {route}: no records, skipping")
            continue

        print(f"\n  Route {route} ({ROUTE_NAMES[route]}): "
              f"{len(route_indices):,} records")

        # Build a small dataset of just these records
        ds = CacheRouteDataset(records, route, tokenizer)
        dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False,
                         num_workers=0)

        # Map original idx → ds idx
        idx_map = {ds_idx: original_idx
                    for ds_idx, original_idx in enumerate(route_indices)}

        pred_counter = Counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dl, desc=f"   Route {route} inference")):
                fused = batch["fused"].to(CFG["device"])
                disease = batch["disease"].to(CFG["device"])
                inp_ids = batch["input_ids"].to(CFG["device"])
                att_msk = batch["attention_mask"].to(CFG["device"])

                logits = model(fused, disease, inp_ids, att_msk)

                if route == 2:
                    # Multi-label with adaptive threshold
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for bi in range(probs.shape[0]):
                        ds_idx = batch_idx * CFG["batch_size"] + bi
                        if ds_idx >= len(route_indices): break
                        p_vec = probs[bi]
                        mean_p = float(np.mean(p_vec))
                        std_p = float(np.std(p_vec))
                        thr = max(0.7, mean_p + 0.5 * std_p)
                        above = sorted(
                            [(i, float(p)) for i, p in enumerate(p_vec)
                             if p >= thr],
                            key=lambda x: x[1], reverse=True)[:5]
                        if not above:
                            top_i = int(np.argmax(p_vec))
                            above = [(top_i, float(p_vec[top_i]))]
                        picks = [vocab[i] for i, _ in above]
                        pred_str = ", ".join(picks)
                        predictions[idx_map[ds_idx]] = pred_str
                        pred_counter[picks[0]] += 1
                else:
                    preds = logits.argmax(dim=-1).cpu().tolist()
                    confs = torch.softmax(logits, dim=-1).max(dim=-1)[0].cpu().tolist()
                    for bi, (p_idx, conf) in enumerate(zip(preds, confs)):
                        ds_idx = batch_idx * CFG["batch_size"] + bi
                        if ds_idx >= len(route_indices): break
                        pred_str = vocab[p_idx] if p_idx < len(vocab) else "?"
                        predictions[idx_map[ds_idx]] = pred_str
                        pred_counter[pred_str] += 1

        # Show diversity diagnostic
        print(f"   Top 5 predictions for Route {route}:")
        for p, c in pred_counter.most_common(5):
            pct = c / len(route_indices) * 100
            print(f"     '{p[:40]:<40}' {c:>6} ({pct:.1f}%)")

    return predictions


def run_yolo_route_with_images(records, yolo_seg, yolo_det, hf_dataset,
                                  split_name):
    """
    Route 4 + Route 5 need actual images (YOLO is image-based).
    Since Stage 3 cache may not have img_id, we iterate the HF dataset
    and match by position to records of routes 4 and 5.
    """
    predictions = {}

    # Build mapping from records position → HF dataset position
    # The Stage 3 cache was built by iterating HF dataset in order, so the
    # original positions are preserved per route.
    route4_indices = [i for i, r in enumerate(records)
                       if r.get("route") == 4]
    route5_indices = [i for i, r in enumerate(records)
                       if r.get("route") == 5]

    # Iterate HF dataset and find images matching routes 4/5 records
    # We need to track which HF samples correspond to which cache record
    print(f"\n  Building HF dataset map for YOLO routes ...")

    # Re-iterate HF dataset, route-classifying each
    r4_hf_idx = 0
    r5_hf_idx = 0
    hf_route4_samples = []  # list of HF samples for route 4 in original order
    hf_route5_samples = []
    for s in tqdm(hf_dataset, desc=f"   Routing HF {split_name}"):
        q = s.get("question", "")
        if not q: continue
        try:
            r = infer_route(q)
        except Exception:
            continue
        if r == 4:
            hf_route4_samples.append(s)
        elif r == 5:
            hf_route5_samples.append(s)

    print(f"  HF Route 4 samples: {len(hf_route4_samples):,} "
          f"(cache has {len(route4_indices):,})")
    print(f"  HF Route 5 samples: {len(hf_route5_samples):,} "
          f"(cache has {len(route5_indices):,})")

    # Sanity: should match
    if len(hf_route4_samples) != len(route4_indices):
        print(f"  ⚠️   Route 4 count mismatch; using min")
    if len(hf_route5_samples) != len(route5_indices):
        print(f"  ⚠️   Route 5 count mismatch; using min")

    image_dir = CFG["image_dir"]

    # ── Route 4: YOLO-Seg (location) ──────────────────────────────────────
    if yolo_seg is not None:
        n4 = min(len(hf_route4_samples), len(route4_indices))
        print(f"\n  Route 4 (location): running YOLO-Seg on {n4:,} images ...")
        pred_counter = Counter()
        for i in tqdm(range(n4), desc="   YOLO-Seg"):
            sample = hf_route4_samples[i]
            cache_idx = route4_indices[i]
            img_id = sample.get("img_id", sample.get("image_id", ""))
            image_path = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
                p = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    image_path = p; break
            if image_path is None:
                predictions[cache_idx] = "no image"
                continue
            try:
                res = yolo_seg(image_path, verbose=False)[0]
                if res.masks is None or len(res.masks) == 0:
                    pred = "no region detected"
                else:
                    confs = (res.boxes.conf.cpu().numpy()
                              if res.boxes is not None else None)
                    if confs is None or len(confs) == 0:
                        pred = "no region detected"
                    else:
                        best = int(np.argmax(confs))
                        mask = res.masks.xy[best]
                        if len(mask) == 0:
                            pred = "no region detected"
                        else:
                            cx = float(np.mean(mask[:, 0])) / res.orig_shape[1]
                            cy = float(np.mean(mask[:, 1])) / res.orig_shape[0]
                            vert = ("upper" if cy < 0.33
                                     else ("lower" if cy > 0.67 else "central"))
                            horiz = ("left" if cx < 0.33
                                      else ("right" if cx > 0.67 else "central"))
                            pred = (f"{vert}-{horiz}"
                                     if vert != horiz else vert)
                predictions[cache_idx] = pred
                pred_counter[pred] += 1
            except Exception as e:
                predictions[cache_idx] = f"error: {str(e)[:30]}"

        print(f"   Top 5 Route 4 predictions:")
        for p, c in pred_counter.most_common(5):
            pct = c / max(n4, 1) * 100
            print(f"     '{p[:40]:<40}' {c:>6} ({pct:.1f}%)")

    # ── Route 5: YOLO-Det (count) ─────────────────────────────────────────
    if yolo_det is not None:
        n5 = min(len(hf_route5_samples), len(route5_indices))
        print(f"\n  Route 5 (count): running YOLO-Det on {n5:,} images ...")
        pred_counter = Counter()
        for i in tqdm(range(n5), desc="   YOLO-Det"):
            sample = hf_route5_samples[i]
            cache_idx = route5_indices[i]
            img_id = sample.get("img_id", sample.get("image_id", ""))
            image_path = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
                p = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    image_path = p; break
            if image_path is None:
                predictions[cache_idx] = "no image"
                continue
            try:
                res = yolo_det(image_path, verbose=False)[0]
                if res.boxes is None:
                    pred = "0"
                else:
                    n_obj = len(res.boxes)
                    if n_obj > 10: pred = "more than 10"
                    elif n_obj > 5: pred = "6-10"
                    else: pred = str(n_obj)
                predictions[cache_idx] = pred
                pred_counter[pred] += 1
            except Exception as e:
                predictions[cache_idx] = f"error: {str(e)[:30]}"

        print(f"   Top 5 Route 5 predictions:")
        for p, c in pred_counter.most_common(5):
            pct = c / max(n5, 1) * 100
            print(f"     '{p[:40]:<40}' {c:>6} ({pct:.1f}%)")

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main per-split processing
# ─────────────────────────────────────────────────────────────────────────────
def build_cache_for_split(split, distilbert_models, tokenizer,
                            yolo_seg, yolo_det):
    print(f"\n{'='*72}")
    print(f"  Building Stage 4 cache for [{split}] split")
    print(f"{'='*72}\n")

    # Load Stage 3 cache for this split
    s3_path = os.path.join(CFG["stage3_cache_dir"],
                            f"stage3_cache_{split}.pt")
    if not os.path.exists(s3_path):
        print(f"❌  Stage 3 cache not found: {s3_path}")
        return

    print(f"  Loading Stage 3 cache: {s3_path}")
    records = torch.load(s3_path, map_location="cpu", weights_only=False)
    print(f"  ✅  Loaded {len(records):,} records\n")

    # Inspect record structure
    if records:
        print(f"  Record fields: {list(records[0].keys())}")

    # Route distribution
    route_dist = Counter(r.get("route", -1) for r in records)
    print(f"  Route distribution:")
    for r in range(6):
        n = route_dist.get(r, 0)
        print(f"     Route {r} ({ROUTE_NAMES[r]:<15}): {n:>7,}")

    # Run DistilBERT routes (0, 1, 2, 3)
    print(f"\n  Running DistilBERT inference on Routes 0, 1, 2, 3 ...")
    distilbert_preds = run_distilbert_routes(records, distilbert_models,
                                                tokenizer)

    # Run YOLO routes (4, 5)
    print(f"\n  Running YOLO inference on Routes 4 and 5 ...")
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])
    if split == "train" or split == "val":
        # Both come from raw["train"] originally
        hf_data = raw["train"]
    else:
        hf_data = raw["test"]
    yolo_preds = run_yolo_route_with_images(records, yolo_seg, yolo_det,
                                              hf_data, split)

    # Combine predictions into cache records
    print(f"\n  Combining all predictions ...")
    all_preds = {**distilbert_preds, **yolo_preds}

    output_records = []
    n_with_pred = 0
    for i, r in enumerate(records):
        pred = all_preds.get(i, "")
        if pred and "error" not in pred.lower():
            n_with_pred += 1
        output_records.append({
            "route":      r.get("route", -1),
            "route_name": ROUTE_NAMES.get(r.get("route", -1), "?"),
            "question":   r.get("question", ""),
            "gt_answer":  r.get("answer", ""),
            "stage4_pred": pred,
        })
    print(f"  ✅  {n_with_pred:,} records with valid predictions "
          f"(out of {len(records):,})")

    # Save cache
    out_path = os.path.join(CFG["stage4_cache_dir"],
                              f"stage4_cache_{split}.pt")
    torch.save(output_records, out_path)
    file_size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  ✅  Saved Stage 4 cache → {out_path}")
    print(f"     File size: {file_size_mb:.1f} MB")
    print(f"     Records:   {len(output_records):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all",
                         choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    print(f"\n{'█'*72}")
    print(f"  PHASE 1 — Building Stage 4 Predictions Cache")
    print(f"  Output dir: {CFG['stage4_cache_dir']}")
    print(f"{'█'*72}\n")

    # Load all Stage 4 models once
    print(f"  Loading 4 DistilBERT routes ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased")

    distilbert_models = {}
    for route, fname in S4_CHECKPOINTS.items():
        path = os.path.join(CFG["ckpt_dir"], fname)
        if not os.path.exists(path):
            print(f"     ⚠️   {fname} not found, skipping")
            continue
        ckpt = torch.load(path, map_location=CFG["device"],
                          weights_only=False)
        vocab = ckpt.get("vocab", [])
        n_classes = ckpt.get("n_classes", len(vocab) if vocab else 2)
        model = DistilBERTRouteModel(n_classes)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model = model.to(CFG["device"]).eval()
        distilbert_models[route] = (model, vocab)
        print(f"     ✅  Route {route} ({ROUTE_NAMES[route]}): "
              f"{n_classes} classes")

    print(f"\n  Loading 2 YOLO routes ...")
    yolo_seg = yolo_det = None
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_SEG_CKPT):
            yolo_seg = YOLO(YOLO_SEG_CKPT)
            print(f"     ✅  YOLO-Seg loaded")
        if os.path.exists(YOLO_DET_CKPT):
            yolo_det = YOLO(YOLO_DET_CKPT)
            print(f"     ✅  YOLO-Det loaded")
    except ImportError:
        print(f"     ⚠️   ultralytics not installed")

    # Process splits
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        start = time.time()
        build_cache_for_split(split, distilbert_models, tokenizer,
                                yolo_seg, yolo_det)
        print(f"\n  ⏱️   {split.upper()} took {(time.time()-start)/60:.1f} min")

    print(f"\n{'█'*72}\n  PHASE 1 COMPLETE\n{'█'*72}\n")


if __name__ == "__main__":
    main()
