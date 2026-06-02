#!/usr/bin/env python3
"""
=============================================================================
PHASE 1 — Build Stage 4 Cache (FIXED: OOM-safe YOLO + resumable)
=============================================================================
CHANGES vs original:
  1. YOLO routing: questions-only pass first (no image loading), then
     image inference in batches with checkpoint every CHECKPOINT_EVERY images
  2. Resume support: if killed, restart and it picks up where it left off
  3. gc.collect() + cuda empty_cache() every GC_EVERY images
  4. Saves partial DistilBERT results too, so those don't re-run on restart
=============================================================================
"""
import os
import sys
import gc
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
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size":       64,
    "stage3_cache_dir": os.path.join(PROJECT, "cache", "stage3_features"),
    "stage4_cache_dir": os.path.join(PROJECT, "cache", "stage4_predictions"),
    "data_dir":         S4_CFG["data_dir"],
    "image_dir":        S4_CFG.get("image_dir", ""),
    "ckpt_dir":         None,
    "CHECKPOINT_EVERY": 500,   # save partial YOLO results every N images
    "GC_EVERY":         200,   # call gc.collect() every N images
}
os.makedirs(CFG["stage4_cache_dir"], exist_ok=True)

for cand in [os.path.join(PROJECT, "checkpoints", "stage4_revised"),
             S4_CFG.get("ckpt_dir", "")]:
    if os.path.exists(os.path.join(cand, "stage4_revised_yes_no_best.pt")):
        CFG["ckpt_dir"] = cand
        break

if not CFG["ckpt_dir"]:
    print("❌  Could not find Stage 4 checkpoints"); sys.exit(1)
print(f"✅  Stage 4 ckpt dir: {CFG['ckpt_dir']}")

S4_CHECKPOINTS = {
    0: "stage4_revised_yes_no_best.pt",
    1: "stage4_revised_single_choice_best.pt",
    2: "stage4_revised_multi_choice_best.pt",
    3: "stage4_revised_color_best.pt",
}
YOLO_SEG_CKPT = os.path.join(CFG["ckpt_dir"], "yolo_seg_finetuned", "weights", "best.pt")
YOLO_DET_CKPT = os.path.join(CFG["ckpt_dir"], "yolo_det_finetuned", "weights", "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class Stage3Projector(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(S4_CFG["head_input_dim"], hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
        )
    def forward(self, fused, disease):
        return self.proj(torch.cat([fused, disease], dim=-1)).unsqueeze(1)


class DistilBERTRouteModel(nn.Module):
    HIDDEN = 768
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, n_classes):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(self.MODEL_NAME)
        self.projector  = Stage3Projector(self.HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN // 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(self.HIDDEN // 2, n_classes),
        )

    def forward(self, fused, disease, input_ids, attention_mask):
        emb      = self.distilbert.embeddings
        word_emb = emb.word_embeddings(input_ids)
        prefix   = self.projector(fused, disease).to(word_emb.dtype)
        combined = torch.cat([prefix, word_emb], dim=1)
        pos_ids  = torch.arange(combined.size(1), dtype=torch.long,
                                device=combined.device).unsqueeze(0)
        combined = combined + emb.position_embeddings(pos_ids)
        combined = emb.LayerNorm(combined); combined = emb.dropout(combined)
        prefix_mask = torch.ones(fused.size(0), 1,
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        ext_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        try:
            out = self.distilbert.transformer(hidden_states=combined, attn_mask=ext_mask)
        except TypeError:
            try:
                out = self.distilbert.transformer(combined, attn_mask=ext_mask)
            except TypeError:
                out = self.distilbert.transformer(x=combined, attn_mask=ext_mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state")
                  else out[0] if isinstance(out, tuple) else out)
        return self.head(hidden[:, 1, :])


class CacheRouteDataset(Dataset):
    def __init__(self, records, route, tokenizer, max_len=128):
        self.records   = [r for r in records if r.get("route") == route]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        enc = self.tokenizer(r["question"], return_tensors="pt",
                             max_length=self.max_len,
                             padding="max_length", truncation=True)
        return {
            "fused":          r.get("fused_repr", r.get("fused")),
            "disease":        r.get("disease_vec", r.get("disease")),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DistilBERT inference (routes 0-3) — unchanged logic, added partial cache
# ─────────────────────────────────────────────────────────────────────────────
def run_distilbert_routes(records, route_models_vocab, tokenizer, split):
    partial_path = os.path.join(CFG["stage4_cache_dir"],
                                f"partial_distilbert_{split}.pt")

    # Resume if partial exists
    if os.path.exists(partial_path):
        predictions = torch.load(partial_path, map_location="cpu",
                                 weights_only=False)
        done_routes = set(records[i].get("route") for i in predictions.keys())
        print(f"  Resuming DistilBERT — already done routes: {done_routes}")
    else:
        predictions = {}
        done_routes = set()

    for route, (model, vocab) in route_models_vocab.items():
        if route in done_routes:
            print(f"  Route {route}: already cached, skipping")
            continue

        route_indices = [i for i, r in enumerate(records)
                         if r.get("route") == route]
        if not route_indices:
            print(f"  Route {route}: no records, skipping"); continue

        print(f"\n  Route {route} ({ROUTE_NAMES[route]}): {len(route_indices):,} records")
        ds  = CacheRouteDataset(records, route, tokenizer)
        dl  = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
        idx_map = {ds_idx: orig for ds_idx, orig in enumerate(route_indices)}

        pred_counter = Counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                    tqdm(dl, desc=f"   Route {route} inference")):
                fused   = batch["fused"].to(CFG["device"])
                disease = batch["disease"].to(CFG["device"])
                inp_ids = batch["input_ids"].to(CFG["device"])
                att_msk = batch["attention_mask"].to(CFG["device"])
                logits  = model(fused, disease, inp_ids, att_msk)

                if route == 2:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for bi in range(probs.shape[0]):
                        ds_idx = batch_idx * CFG["batch_size"] + bi
                        if ds_idx >= len(route_indices): break
                        p_vec  = probs[bi]
                        thr    = max(0.7, float(np.mean(p_vec)) + 0.5 * float(np.std(p_vec)))
                        above  = sorted([(i, float(p)) for i, p in enumerate(p_vec)
                                         if p >= thr],
                                        key=lambda x: x[1], reverse=True)[:5]
                        if not above:
                            top_i = int(np.argmax(p_vec))
                            above = [(top_i, float(p_vec[top_i]))]
                        picks = [vocab[i] for i, _ in above]
                        predictions[idx_map[ds_idx]] = ", ".join(picks)
                        pred_counter[picks[0]] += 1
                else:
                    preds = logits.argmax(dim=-1).cpu().tolist()
                    for bi, p_idx in enumerate(preds):
                        ds_idx = batch_idx * CFG["batch_size"] + bi
                        if ds_idx >= len(route_indices): break
                        pred_str = vocab[p_idx] if p_idx < len(vocab) else "?"
                        predictions[idx_map[ds_idx]] = pred_str
                        pred_counter[pred_str] += 1

        print(f"   Top 5 predictions for Route {route}:")
        for p, c in pred_counter.most_common(5):
            print(f"     '{p[:40]:<40}' {c:>6} ({c/len(route_indices)*100:.1f}%)")

        # Save partial after each route completes
        torch.save(predictions, partial_path)
        print(f"   Partial DistilBERT cache saved.")

    # Clean up partial
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# YOLO inference (routes 4-5) — FIXED: questions-only pass + checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo_route_with_images(records, yolo_seg, yolo_det,
                               hf_dataset, split_name):
    predictions  = {}
    image_dir    = CFG["image_dir"]

    route4_indices = [i for i, r in enumerate(records) if r.get("route") == 4]
    route5_indices = [i for i, r in enumerate(records) if r.get("route") == 5]

    # ── STEP 1: questions-only pass (no images loaded) ────────────────────
    # This is fast — just string matching, no PIL/YOLO overhead
    print(f"\n  Step 1: questions-only routing pass (no image loading) ...")
    hf_route4_samples = []
    hf_route5_samples = []

    for s in tqdm(hf_dataset, desc=f"   Routing questions {split_name}"):
        q = s.get("question", "")
        if not q:
            continue
        try:
            r = infer_route(q)
        except Exception:
            continue
        # Store only the lightweight fields needed for YOLO (img_id + question)
        # Do NOT append the full sample — avoids accumulating PIL images in RAM
        if r == 4:
            hf_route4_samples.append({
                "img_id":   s.get("img_id", s.get("image_id", "")),
                "question": q,
            })
        elif r == 5:
            hf_route5_samples.append({
                "img_id":   s.get("img_id", s.get("image_id", "")),
                "question": q,
            })

    print(f"  Route 4 questions: {len(hf_route4_samples):,} "
          f"(cache has {len(route4_indices):,})")
    print(f"  Route 5 questions: {len(hf_route5_samples):,} "
          f"(cache has {len(route5_indices):,})")

    # ── STEP 2: YOLO-Seg (Route 4) with checkpointing ────────────────────
    if yolo_seg is not None:
        partial_path = os.path.join(CFG["stage4_cache_dir"],
                                    f"partial_r4_{split_name}.pt")
        # Resume
        start_i = 0
        if os.path.exists(partial_path):
            partial = torch.load(partial_path, map_location="cpu",
                                 weights_only=False)
            predictions.update(partial)
            start_i = len(partial)
            print(f"  Resuming Route 4 from image {start_i:,}")

        n4 = min(len(hf_route4_samples), len(route4_indices))
        print(f"\n  Route 4 (location): YOLO-Seg on {n4 - start_i:,} remaining images ...")
        pred_counter = Counter()

        for i in tqdm(range(start_i, n4), desc="   YOLO-Seg"):
            cache_idx = route4_indices[i]
            img_id    = hf_route4_samples[i]["img_id"]
            image_path = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
                p = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    image_path = p; break

            if image_path is None:
                predictions[cache_idx] = "no image"
            else:
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
                                cx   = float(np.mean(mask[:, 0])) / res.orig_shape[1]
                                cy   = float(np.mean(mask[:, 1])) / res.orig_shape[0]
                                vert = ("upper" if cy < 0.33 else
                                        "lower" if cy > 0.67 else "central")
                                horiz = ("left" if cx < 0.33 else
                                         "right" if cx > 0.67 else "central")
                                pred = f"{vert}-{horiz}" if vert != horiz else vert
                    predictions[cache_idx] = pred
                    pred_counter[pred] += 1
                except Exception as e:
                    predictions[cache_idx] = f"error: {str(e)[:30]}"

            # Periodic checkpoint
            done = i - start_i + 1
            if done % CFG["CHECKPOINT_EVERY"] == 0:
                r4_preds = {k: v for k, v in predictions.items()
                            if k in route4_indices[:i + 1]}
                torch.save(r4_preds, partial_path)
                print(f"   Checkpoint at {i+1}/{n4}")

            # Periodic GC
            if done % CFG["GC_EVERY"] == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final checkpoint then clean up
        r4_preds = {k: v for k, v in predictions.items()
                    if k in route4_indices}
        torch.save(r4_preds, partial_path)
        print(f"   Top 5 Route 4 predictions:")
        for p, c in pred_counter.most_common(5):
            print(f"     '{p[:40]:<40}' {c:>6} ({c/max(n4,1)*100:.1f}%)")
        os.remove(partial_path)

    # ── STEP 3: YOLO-Det (Route 5) with checkpointing ────────────────────
    if yolo_det is not None:
        partial_path = os.path.join(CFG["stage4_cache_dir"],
                                    f"partial_r5_{split_name}.pt")
        start_i = 0
        if os.path.exists(partial_path):
            partial = torch.load(partial_path, map_location="cpu",
                                 weights_only=False)
            predictions.update(partial)
            start_i = len(partial)
            print(f"  Resuming Route 5 from image {start_i:,}")

        n5 = min(len(hf_route5_samples), len(route5_indices))
        print(f"\n  Route 5 (count): YOLO-Det on {n5 - start_i:,} remaining images ...")
        pred_counter = Counter()

        for i in tqdm(range(start_i, n5), desc="   YOLO-Det"):
            cache_idx  = route5_indices[i]
            img_id     = hf_route5_samples[i]["img_id"]
            image_path = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
                p = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    image_path = p; break

            if image_path is None:
                predictions[cache_idx] = "no image"
            else:
                try:
                    res   = yolo_det(image_path, verbose=False)[0]
                    n_obj = len(res.boxes) if res.boxes is not None else 0
                    pred  = ("more than 10" if n_obj > 10 else
                             "6-10"         if n_obj > 5  else str(n_obj))
                    predictions[cache_idx] = pred
                    pred_counter[pred] += 1
                except Exception as e:
                    predictions[cache_idx] = f"error: {str(e)[:30]}"

            done = i - start_i + 1
            if done % CFG["CHECKPOINT_EVERY"] == 0:
                r5_preds = {k: v for k, v in predictions.items()
                            if k in route5_indices[:i + 1]}
                torch.save(r5_preds, partial_path)
                print(f"   Checkpoint at {i+1}/{n5}")

            if done % CFG["GC_EVERY"] == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        r5_preds = {k: v for k, v in predictions.items()
                    if k in route5_indices}
        torch.save(r5_preds, partial_path)
        print(f"   Top 5 Route 5 predictions:")
        for p, c in pred_counter.most_common(5):
            print(f"     '{p[:40]:<40}' {c:>6} ({c/max(n5,1)*100:.1f}%)")
        os.remove(partial_path)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main per-split processing
# ─────────────────────────────────────────────────────────────────────────────
def build_cache_for_split(split, distilbert_models, tokenizer,
                          yolo_seg, yolo_det):
    print(f"\n{'='*72}")
    print(f"  Building Stage 4 cache for [{split}] split")
    print(f"{'='*72}\n")

    s3_path = os.path.join(CFG["stage3_cache_dir"], f"stage3_cache_{split}.pt")
    if not os.path.exists(s3_path):
        print(f"❌  Stage 3 cache not found: {s3_path}"); return

    print(f"  Loading Stage 3 cache: {s3_path}")
    records = torch.load(s3_path, map_location="cpu", weights_only=False)
    print(f"  ✅  Loaded {len(records):,} records")
    if records:
        print(f"  Record fields: {list(records[0].keys())}")

    route_dist = Counter(r.get("route", -1) for r in records)
    print(f"  Route distribution:")
    for r in range(6):
        print(f"     Route {r} ({ROUTE_NAMES[r]:<15}): {route_dist.get(r, 0):>7,}")

    # DistilBERT routes 0-3
    print(f"\n  Running DistilBERT inference on Routes 0-3 ...")
    distilbert_preds = run_distilbert_routes(
        records, distilbert_models, tokenizer, split)

    # YOLO routes 4-5
    print(f"\n  Running YOLO inference on Routes 4-5 ...")
    from datasets import load_from_disk
    raw     = load_from_disk(CFG["data_dir"])
    hf_data = raw["train"] if split in ("train", "val") else raw["test"]
    yolo_preds = run_yolo_route_with_images(
        records, yolo_seg, yolo_det, hf_data, split)

    # Combine
    all_preds    = {**distilbert_preds, **yolo_preds}
    output_records = []
    n_with_pred  = 0
    for i, r in enumerate(records):
        pred = all_preds.get(i, "")
        if pred and "error" not in pred.lower():
            n_with_pred += 1
        output_records.append({
            "route":       r.get("route", -1),
            "route_name":  ROUTE_NAMES.get(r.get("route", -1), "?"),
            "question":    r.get("question", ""),
            "gt_answer":   r.get("answer", ""),
            "stage4_pred": pred,
        })
    print(f"  ✅  {n_with_pred:,} records with valid predictions "
          f"(out of {len(records):,})")

    out_path = os.path.join(CFG["stage4_cache_dir"],
                            f"stage4_cache_{split}.pt")
    torch.save(output_records, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  ✅  Saved → {out_path}  ({size_mb:.1f} MB, {len(output_records):,} records)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    print(f"\n{'█'*72}")
    print(f"  PHASE 1 — Building Stage 4 Predictions Cache (FIXED)")
    print(f"  Output dir: {CFG['stage4_cache_dir']}")
    print(f"{'█'*72}\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    distilbert_models = {}
    for route, fname in S4_CHECKPOINTS.items():
        path = os.path.join(CFG["ckpt_dir"], fname)
        if not os.path.exists(path):
            print(f"     ⚠️   {fname} not found, skipping"); continue
        ckpt  = torch.load(path, map_location=CFG["device"], weights_only=False)
        vocab = ckpt.get("vocab", [])
        n_cls = ckpt.get("n_classes", len(vocab) if vocab else 2)
        model = DistilBERTRouteModel(n_cls)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model = model.to(CFG["device"]).eval()
        distilbert_models[route] = (model, vocab)
        print(f"     ✅  Route {route} ({ROUTE_NAMES[route]}): {n_cls} classes")

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
        print("     ⚠️   ultralytics not installed")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        start = time.time()
        build_cache_for_split(split, distilbert_models, tokenizer,
                              yolo_seg, yolo_det)
        print(f"\n  ⏱️   {split.upper()} took {(time.time()-start)/60:.1f} min")

    print(f"\n{'█'*72}\n  PHASE 1 COMPLETE\n{'█'*72}\n")


if __name__ == "__main__":
    main()
