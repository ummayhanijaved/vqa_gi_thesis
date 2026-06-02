#!/usr/bin/env python3
"""
=============================================================================
Stage 5 — Complete End-to-End Pipeline Testing (Self-Contained)
=============================================================================

This version BUILDS ITS OWN Stage 4 predictor inline to avoid the
build_image_transform() signature mismatch in stage4_revised.py.

It directly loads all six trained Stage 4 checkpoints:
   - stage4_revised_yes_no_best.pt        (DistilBERT)
   - stage4_revised_single_choice_best.pt (DistilBERT)
   - stage4_revised_multi_choice_best.pt  (DistilBERT)
   - stage4_revised_color_best.pt         (DistilBERT)
   - yolo_seg_finetuned/weights/best.pt   (YOLO-Seg)
   - yolo_det_finetuned/weights/best.pt   (YOLO-Det)

Plus Stage 5 T5-small verbalizer:
   - stage5_verbalizer/stage5_verbalizer_best.pt

USAGE:
    python stage5_pipeline_test.py --mode all --n_samples 300
=============================================================================
"""
import os
import sys
import time
import re
import argparse
import warnings
import random
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

try:
    from transformers import (
        T5Tokenizer, T5ForConditionalGeneration,
        DistilBertModel, DistilBertTokenizerFast,
    )
except ImportError as e:
    print(f"❌  Failed to import transformers: {e}")
    sys.exit(1)

try:
    from rapidfuzz import fuzz as _fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False

# Import only the SAFE things from stage4_revised (no Stage4RevisedPredictor)
from stage4_revised import (
    CFG as S4_CFG, ROUTE_NAMES, DISTILBERT_ROUTES, YOLO_ROUTES,
    FusionExtractor, infer_route, normalise_answer,
)
from preprocessing import TextPreprocessor, build_image_transform


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
    "s5_model_name"     : "t5-small",
    "s5_ckpt_path"      : os.path.expanduser(
        "~/vqa_gi_thesis/checkpoints/stage5_verbalizer/stage5_verbalizer_best.pt"),
    "s5_max_input_len"  : 128,
    "s5_max_output_len" : 96,
    "s5_num_beams"      : 4,
    "data_dir"          : S4_CFG["data_dir"],
    "image_dir"         : S4_CFG.get("image_dir", ""),
    # Auto-detect: stage4 ckpts live in <ckpt_dir>/stage4_revised/ OR
    # in <ckpt_dir>/ directly. Choose the one that exists.
    "ckpt_dir"          : (S4_CFG["ckpt_dir"] + "/stage4_revised"
                            if os.path.exists(
                                os.path.join(S4_CFG["ckpt_dir"],
                                              "stage4_revised",
                                              "stage4_revised_yes_no_best.pt"))
                            else (S4_CFG["ckpt_dir"]
                                  if os.path.exists(
                                      os.path.join(S4_CFG["ckpt_dir"],
                                                    "stage4_revised_yes_no_best.pt"))
                                  else os.path.join(
                                      os.path.dirname(S4_CFG["ckpt_dir"]),
                                      "stage4_revised"))),
    "out_dir"           : os.path.expanduser(
        "~/vqa_gi_thesis/logs/stage5_verbalizer"),
}
os.makedirs(CFG["out_dir"], exist_ok=True)

# Stage 4 checkpoint filenames
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

STOPWORDS = set("""a an the is are was were be been being am have has had do
does did will would could should may might must can could shall and or but if
then so of in on at by for with from to into onto upon between within without
""".split())

MEDICAL_SYNONYMS = [
    {"polyp", "polyps", "polypoid", "lesion", "lesions", "growth",
     "neoplasm", "abnormality", "abnormalities"},
    {"colonoscopy", "colonoscopic"},
    {"gastroscopy", "gastroscopic", "endoscopy"},
    {"instrument", "instruments", "tube", "tubes", "scope", "device", "tool"},
    {"identified", "detected", "observed", "noted", "visible", "present",
     "seen", "found", "shown", "displayed"},
    {"no", "not", "absent", "without", "none", "negative", "lack"},
    {"upper", "above", "top", "superior"},
    {"lower", "below", "bottom", "inferior"},
    {"central", "centre", "center", "middle", "centrally"},
    {"left", "leftward"}, {"right", "rightward"},
    {"colon", "rectum", "intestine", "bowel", "rectal", "colonic", "intestinal"},
    {"esophagus", "oesophagus", "esophageal", "oesophageal"},
    {"stomach", "gastric"},
    {"red", "reddish", "erythematous", "hyperaemic", "hyperemic"},
    {"pink", "rose"},
    {"yes", "affirmative", "positive", "confirmed"},
    {"removed", "resected", "extracted", "excised"},
    {"size", "sized", "millimeter", "millimetre", "mm",
     "small", "medium", "large", "tiny"},
    {"landmark", "landmarks", "anatomical", "anatomic"},
    {"finding", "findings", "feature", "features"},
    {"artifact", "artifacts", "artefact", "artefacts"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained Stage 4 model (mirrors stage4_revised's DistilBERTAnswerModel)
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
    """Single-route DistilBERT — for inference only (no training)."""
    HIDDEN     = 768
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, n_classes):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(self.MODEL_NAME)
        self.projector  = Stage3Projector(self.HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.HIDDEN // 2, n_classes),
        )

    def forward(self, fused, disease, input_ids, attention_mask):
        emb      = self.distilbert.embeddings
        word_emb = emb.word_embeddings(input_ids)
        prefix   = self.projector(fused, disease).to(word_emb.dtype)
        combined = torch.cat([prefix, word_emb], dim=1)

        seq_len = combined.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long,
                                device=combined.device).unsqueeze(0)
        combined = combined + emb.position_embeddings(pos_ids)
        combined = emb.LayerNorm(combined)
        combined = emb.dropout(combined)

        prefix_mask = torch.ones(fused.size(0), 1,
                                  dtype=attention_mask.dtype,
                                  device=attention_mask.device)
        ext_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # New-API-first signature handling
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
        cls = hidden[:, 1, :]
        return self.head(cls)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Predictor (self-contained — no Stage4RevisedPredictor)
# ─────────────────────────────────────────────────────────────────────────────
class FullPipelinePredictor:
    def __init__(self):
        print(f"\n{'='*72}")
        print(f"  Initialising Full 5-Stage Pipeline (Self-Contained)")
        print(f"{'='*72}\n")

        # Stages 1-3: shared FusionExtractor (provides 535-D features)
        print(f"  Loading Stages 1-3 (FusionExtractor) ...")
        self.extractor = FusionExtractor(S4_CFG["stage3_ckpt"])
        self.text_prep = TextPreprocessor()
        # Diagnostic — print available extraction methods
        ext_methods = [m for m in dir(self.extractor)
                       if not m.startswith("_") and
                       callable(getattr(self.extractor, m, None))]
        relevant = [m for m in ext_methods if any(
            k in m.lower() for k in ["extract", "forward", "predict", "call"])]
        print(f"  FusionExtractor methods detected: {relevant}")
        # Build image transform — use "test" or fallback
        try:
            self.transform = build_image_transform("test")
        except TypeError:
            try:
                self.transform = build_image_transform(is_train=False)
            except TypeError:
                self.transform = build_image_transform()
        print(f"  ✅  Stages 1-3 ready\n")

        # Stage 4: load 4 DistilBERT + 2 YOLO models
        print(f"  Loading Stage 4 DistilBERT routes ...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased")
        self.distilbert_models = {}
        for route, fname in S4_CHECKPOINTS.items():
            path = os.path.join(CFG["ckpt_dir"], fname)
            if not os.path.exists(path):
                print(f"     ⚠️   {fname} not found, skipping route {route}")
                continue
            ckpt = torch.load(path, map_location=CFG["device"],
                              weights_only=False)
            vocab = ckpt.get("vocab", ckpt.get("vocab_single", []))
            if not vocab:
                # Try alternate key
                for k in ["vocab_list", "classes", "labels"]:
                    if k in ckpt:
                        vocab = ckpt[k]
                        break
            n_classes = ckpt.get("n_classes", len(vocab) if vocab else 2)
            model = DistilBERTRouteModel(n_classes)
            try:
                model.load_state_dict(ckpt["model_state"], strict=False)
            except Exception as e:
                print(f"     ⚠️   Couldn't load {fname} strictly: {e}")
                try:
                    model.load_state_dict(ckpt["model_state"], strict=False)
                except Exception:
                    pass
            model = model.to(CFG["device"]).eval()
            self.distilbert_models[route] = (model, vocab)
            print(f"     ✅  Route {route} ({ROUTE_NAMES[route]:<15}): "
                  f"{n_classes} classes, vocab={len(vocab)}")

        # YOLO models
        print(f"\n  Loading Stage 4 YOLO routes ...")
        self.yolo_seg = None
        self.yolo_det = None
        try:
            from ultralytics import YOLO
            if os.path.exists(YOLO_SEG_CKPT):
                self.yolo_seg = YOLO(YOLO_SEG_CKPT)
                print(f"     ✅  Route 4 (location): YOLO-Seg loaded")
            else:
                print(f"     ⚠️   YOLO-Seg not found at {YOLO_SEG_CKPT}")
            if os.path.exists(YOLO_DET_CKPT):
                self.yolo_det = YOLO(YOLO_DET_CKPT)
                print(f"     ✅  Route 5 (count):    YOLO-Det loaded")
            else:
                print(f"     ⚠️   YOLO-Det not found at {YOLO_DET_CKPT}")
        except ImportError:
            print(f"     ⚠️   ultralytics not installed — YOLO disabled")

        # Stage 5: T5-small verbalizer
        print(f"\n  Loading Stage 5 T5-small ...")
        if not os.path.exists(CFG["s5_ckpt_path"]):
            raise FileNotFoundError(
                f"Stage 5 ckpt not found: {CFG['s5_ckpt_path']}")
        self.s5_tokenizer = T5Tokenizer.from_pretrained(CFG["s5_model_name"])
        self.s5_model = T5ForConditionalGeneration.from_pretrained(
            CFG["s5_model_name"])
        ckpt = torch.load(CFG["s5_ckpt_path"], map_location=CFG["device"],
                          weights_only=False)
        self.s5_model.load_state_dict(ckpt["model_state"])
        self.s5_model = self.s5_model.to(CFG["device"]).eval()
        print(f"     ✅  Stage 5 ready (epoch {ckpt['epoch']}, "
              f"val_loss={ckpt['val_loss']:.4f})\n")

    def _extract_features(self, img, question):
        """
        Extract Stage 3 features.
        Verified signatures (from stage4_answer_generation.py):
          - text_prep.preprocess(text) → dict with input_ids, attention_mask (1-D tensors)
          - extractor.extract(image_tensor, input_ids, attention_mask)
              → dict with fused_repr, disease_vec, routing_label, routing_probs
            All inputs must be (B, ...) — i.e. unsqueeze(0) for single sample.
        """
        # 1. Tokenise question via TextPreprocessor
        tp = self.text_prep.preprocess(question)
        input_ids = tp["input_ids"].unsqueeze(0).to(CFG["device"])     # (1, L)
        attn_mask = tp["attention_mask"].unsqueeze(0).to(CFG["device"]) # (1, L)

        # 2. Transform image
        img_t = self.transform(img).unsqueeze(0).to(CFG["device"])     # (1, C, H, W)

        # 3. Run FusionExtractor — try extract() then extract_batch()
        with torch.no_grad():
            if hasattr(self.extractor, "extract"):
                out = self.extractor.extract(img_t, input_ids, attn_mask)
            elif hasattr(self.extractor, "extract_batch"):
                # Older API expecting (images, questions, transform, text_prep)
                out = self.extractor.extract_batch(
                    [img], [question], self.transform, self.text_prep)
            else:
                methods = [m for m in dir(self.extractor)
                           if not m.startswith("_")]
                raise RuntimeError(
                    f"FusionExtractor has no extract or extract_batch. "
                    f"Available methods: {methods}")

        # 4. Pull fused + disease from output dict
        if isinstance(out, dict):
            fused = out.get("fused_repr", out.get("fused"))
            disease = out.get("disease_vec", out.get("disease"))
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            fused, disease = out[0], out[1]
        else:
            raise RuntimeError(
                f"Unexpected FusionExtractor output type: {type(out)}")

        # Strip batch dim (extract returns (1, 512) and (1, 23))
        if fused.dim() == 2:
            fused = fused[0]
        if disease.dim() == 2:
            disease = disease[0]

        return fused, disease

    def _predict_distilbert(self, route, image_path, question):
        model, vocab = self.distilbert_models[route]
        img = Image.open(image_path).convert("RGB")

        # Get Stage 3 features via robust extraction
        fused_v, disease_v = self._extract_features(img, question)
        fused   = fused_v.unsqueeze(0).to(CFG["device"])
        disease = disease_v.unsqueeze(0).to(CFG["device"])

        enc = self.tokenizer(
            question, return_tensors="pt", max_length=128,
            padding="max_length", truncation=True)
        inp_ids = enc["input_ids"].to(CFG["device"])
        att_msk = enc["attention_mask"].to(CFG["device"])

        with torch.no_grad():
            logits = model(fused, disease, inp_ids, att_msk)
        if route == 2:  # Multi-label — fixed: cap to top-K most confident
            probs = torch.sigmoid(logits[0]).cpu().numpy()
            # Use adaptive threshold: only keep labels significantly above mean
            mean_prob = float(np.mean(probs))
            std_prob  = float(np.std(probs))
            # Threshold = max(0.7, mean + 0.5*std) — much stricter than 0.5
            adaptive_thr = max(0.7, mean_prob + 0.5 * std_prob)
            # Get all classes above adaptive threshold
            above = [(i, float(p)) for i, p in enumerate(probs)
                     if p >= adaptive_thr]
            # Sort by confidence and cap to top 5 (avoid full-vocab dumps)
            above.sort(key=lambda x: x[1], reverse=True)
            top_k = above[:5]
            if not top_k:
                # No class above threshold — fallback to single top-1
                top_idx = int(np.argmax(probs))
                top_k = [(top_idx, float(probs[top_idx]))]
            picks = [vocab[i] for i, _ in top_k]
            answer = ", ".join(picks)
        else:
            pred_idx = logits[0].argmax().item()
            answer = vocab[pred_idx] if pred_idx < len(vocab) else "?"
        return answer

    def _predict_yolo_location(self, image_path):
        if self.yolo_seg is None:
            return "yolo not available"
        res = self.yolo_seg(image_path, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            return "no region detected"
        # Use most confident mask
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
        if confs is None or len(confs) == 0:
            return "no region detected"
        best = int(np.argmax(confs))
        mask = res.masks.xy[best]
        if len(mask) == 0:
            return "no region detected"
        # Compute centroid
        cx = float(np.mean(mask[:, 0])) / res.orig_shape[1]
        cy = float(np.mean(mask[:, 1])) / res.orig_shape[0]
        vert = "upper" if cy < 0.33 else ("lower" if cy > 0.67 else "central")
        horiz = "left" if cx < 0.33 else ("right" if cx > 0.67 else "central")
        return f"{vert}-{horiz}" if vert != horiz else vert

    def _predict_yolo_count(self, image_path):
        if self.yolo_det is None:
            return "yolo not available"
        res = self.yolo_det(image_path, verbose=False)[0]
        if res.boxes is None: return "0"
        n = len(res.boxes)
        if n > 10: return "more than 10"
        if n > 5:  return "6-10"
        return str(n)

    def predict(self, image_path, question, verbose=False):
        timings = {}
        t0 = time.time()
        try:
            route = infer_route(question)
        except Exception:
            route = -1

        # Surface disease vector for explainability grounding (ALL routes).
        # Stage 6 reads result["disease_vec"] to name the predicted disease
        # in the templated medical explanation. If extraction fails for any
        # reason, disease_vec stays None and the explanation falls back to a
        # generic phrase (handled downstream).
        disease_vec = None
        try:
            _img = Image.open(image_path).convert("RGB")
            _fused, _disease = self._extract_features(_img, question)
            disease_vec = _disease.detach().cpu()
        except Exception as e:
            print(f"  disease_vec extraction failed: {type(e).__name__}: {e}")
            disease_vec = None

        s4_model = "unknown"
        try:
            if route in self.distilbert_models:
                s4_answer = self._predict_distilbert(route, image_path, question)
                s4_model  = "DistilBERT"
            elif route == 4:
                s4_answer = self._predict_yolo_location(image_path)
                s4_model  = "YOLO-Seg"
            elif route == 5:
                s4_answer = self._predict_yolo_count(image_path)
                s4_model  = "YOLO-Det"
            else:
                s4_answer = "unknown route"
        except Exception as e:
            s4_answer = f"(error: {str(e)[:30]})"
        timings["s4_total_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        inp_text = (
            f"verbalize | route: {ROUTE_NAMES.get(route, 'unknown')} "
            f"| question: {question[:100]} "
            f"| answer: {str(s4_answer)[:80]}"
        )
        inp = self.s5_tokenizer(
            inp_text, max_length=CFG["s5_max_input_len"],
            truncation=True, return_tensors="pt").to(CFG["device"])
        with torch.no_grad():
            gen_ids = self.s5_model.generate(
                **inp, max_length=CFG["s5_max_output_len"],
                num_beams=CFG["s5_num_beams"], early_stopping=True)
        sentence = self.s5_tokenizer.decode(gen_ids[0],
                                              skip_special_tokens=True)
        timings["s5_ms"] = (time.time() - t0) * 1000

        result = {
            "question": question, "image_path": image_path,
            "route": route, "route_name": ROUTE_NAMES.get(route, "?"),
            "s4_model": s4_model, "s4_answer": s4_answer,
            "s5_sentence": sentence, "s5_input": inp_text,
            "disease_vec": disease_vec,
            "timings": timings,
        }

        if verbose:
            print(f"\n{'─'*72}")
            print(f"   Q : {question}")
            print(f"   Image: {os.path.basename(image_path)}")
            print(f"   Route: {route} ({ROUTE_NAMES.get(route, '?')})")
            print(f"   S4 Model:    {s4_model}")
            print(f"   S4 Answer:   {s4_answer}")
            print(f"   S5 Sentence: {sentence}")
            print(f"   Timing: S1-4={timings['s4_total_ms']:.0f}ms  "
                  f"S5={timings['s5_ms']:.0f}ms")
            print(f"{'─'*72}\n")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced NLG metrics
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [t for t in text.split() if t]


def content_tokens(text):
    return set(tokenize(text)) - STOPWORDS


def expand_synonyms(tokens):
    out = set(tokens)
    for tok in list(tokens):
        for group in MEDICAL_SYNONYMS:
            if tok in group:
                out.update(group); break
    return out


def soft_match(pred, gt):
    p = pred.strip().lower(); g = gt.strip().lower()
    if not p or not g: return 0.0
    if p in g or g in p: return 1.0
    p_tok = content_tokens(p); g_tok = content_tokens(g)
    if not g_tok: return 0.0
    return 1.0 if len(p_tok & g_tok) / len(g_tok) >= 0.5 else 0.0


def clinical_adequacy(pred, gt):
    p_tok = content_tokens(pred); g_tok = content_tokens(gt)
    if not g_tok: return 0.0
    NEG = {"no", "not", "absent", "without", "none", "negative", "lack"}
    neg_p = bool(p_tok & NEG); neg_g = bool(g_tok & NEG)
    if neg_p != neg_g: return 0.0
    p_exp = expand_synonyms(p_tok)
    g_concepts = g_tok - NEG
    if not g_concepts: return 1.0
    covered = sum(1 for c in g_concepts if c in p_exp)
    return 1.0 if (covered / len(g_concepts)) >= 0.6 else 0.0


def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_n(pred, gt, n):
    p = tokenize(pred); g = tokenize(gt)
    if not p or len(p) < n: return 0.0
    p_ng = _ngrams(p, n); g_ng = _ngrams(g, n)
    if not p_ng or not g_ng: return 0.0
    g_set = set(g_ng)
    return sum(1 for ng in p_ng if ng in g_set) / len(p_ng)


def fuzzy_match(pred, gt):
    if not HAVE_RAPIDFUZZ: return 0.0
    return _fuzz.token_set_ratio(str(pred), str(gt)) / 100.0


def token_f1(pred, gt):
    p = set(tokenize(pred)); g = set(tokenize(gt))
    if not p or not g: return 0.0
    common = p & g
    if not common: return 0.0
    prec = len(common) / len(p); rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred, gt):
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


def categorize_quality(pred, gt, s4_answer):
    if exact_match(pred, gt) == 1.0: return "PERFECT"
    if soft_match(pred, gt) == 1.0 and clinical_adequacy(pred, gt) == 1.0:
        return "FLUENT_CORRECT"
    if soft_match(pred, gt) == 1.0:
        return "FLUENT_INCORRECT"
    return "POOR"


def compute_all_metrics(pred, gt, s4_answer):
    return {
        "soft_match":         soft_match(pred, gt),
        "clinical_adequacy":  clinical_adequacy(pred, gt),
        "bleu_1":             bleu_n(pred, gt, 1),
        "bleu_2":             bleu_n(pred, gt, 2),
        "fuzzy_match":        fuzzy_match(pred, gt),
        "token_f1":           token_f1(pred, gt),
        "exact":              exact_match(pred, gt),
        "category":           categorize_quality(pred, gt, s4_answer),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Interactive / Bulk modes
# ─────────────────────────────────────────────────────────────────────────────
def _find_image(img_id, image_dir):
    for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
        p = os.path.join(image_dir, f"{img_id}{ext}")
        if os.path.exists(p): return p
    return None


def demo_mode(predictor):
    print(f"\n{'='*72}\n  Demo Mode (1 per route)\n{'='*72}\n")
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]
    by_route = defaultdict(list)
    for s in test_split:
        q = s.get("question", "")
        if not q: continue
        try: r = infer_route(q)
        except Exception: continue
        by_route[r].append(s)
    rng = random.Random(42)
    for route in range(6):
        if not by_route[route]: continue
        s = rng.choice(by_route[route])
        img = _find_image(s.get("img_id", ""), CFG["image_dir"])
        if not img:
            print(f"  Route {route}: image not found"); continue
        result = predictor.predict(img, s["question"], verbose=True)
        print(f"   GT Sentence: {s.get('answer', '(no GT)')[:140]}\n")


def interactive_mode(predictor, image_path, question):
    print(f"\n{'='*72}\n  Interactive Mode\n{'='*72}\n")
    if not os.path.exists(image_path):
        print(f"❌  Image not found"); return
    predictor.predict(image_path, question, verbose=True)


def bulk_mode(predictor, n_samples=200):
    print(f"\n{'='*72}\n  Bulk Test ({n_samples} samples)\n{'='*72}\n")
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]
    by_route = defaultdict(list)
    print(f"  Routing test samples ...")
    for s in test_split:
        q = s.get("question", "")
        if not q: continue
        try: r = infer_route(q)
        except Exception: continue
        by_route[r].append(s)
    per_route = max(1, n_samples // 6)
    selected = []
    rng = random.Random(42)
    for r in range(6):
        cands = by_route[r]
        if not cands: continue
        picked = rng.sample(cands, min(per_route, len(cands)))
        for p in picked:
            p["_route"] = r
            selected.append(p)
    print(f"  Selected {len(selected):,} samples\n")

    rows, latencies = [], []
    for sample in tqdm(selected, desc="Pipeline"):
        img = _find_image(sample.get("img_id", ""), CFG["image_dir"])
        if not img: continue
        try:
            r = predictor.predict(img, sample["question"])
        except Exception as e:
            print(f"  ⚠️   Error: {e}"); continue
        gt = sample.get("answer", "").strip()
        pred = r["s5_sentence"]
        m = compute_all_metrics(pred, gt, r["s4_answer"])
        rows.append({
            "route": r["route"], "route_name": r["route_name"],
            "img_id": sample.get("img_id", ""), "question": sample["question"],
            "s4_model": r["s4_model"], "s4_answer": str(r["s4_answer"]),
            "s5_sentence": pred, "gt_sentence": gt, **m,
            "s4_latency_ms": r["timings"]["s4_total_ms"],
            "s5_latency_ms": r["timings"]["s5_ms"],
        })
        latencies.append({"route": r["route"],
                          "s4_latency_ms": r["timings"]["s4_total_ms"],
                          "s5_latency_ms": r["timings"]["s5_ms"]})

    if not rows:
        print(f"\n❌  No samples evaluated"); return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CFG["out_dir"], "pipeline_test_results.csv"),
              index=False)

    # ── PRIMARY METRICS ──────────────────────────────────────────────────
    print(f"\n{'='*72}\n  PRIMARY METRICS (thesis-recommended)\n{'='*72}\n")
    hdr = f"  {'Route':<22} {'Soft':>8} {'Clin':>8} {'BLEU1':>8} {'BLEU2':>8} {'Fuzzy':>8} {'N':>5}"
    print(hdr)
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    summary = []
    for r in range(6):
        sub = df[df["route"] == r]
        if len(sub) == 0: continue
        row = {
            "route": r, "route_name": ROUTE_NAMES[r], "n": len(sub),
            "soft_match":        sub["soft_match"].mean()*100,
            "clinical_adequacy": sub["clinical_adequacy"].mean()*100,
            "bleu_1":            sub["bleu_1"].mean()*100,
            "bleu_2":            sub["bleu_2"].mean()*100,
            "fuzzy_match":       sub["fuzzy_match"].mean()*100,
            "token_f1":          sub["token_f1"].mean()*100,
            "exact":             sub["exact"].mean()*100,
        }
        summary.append(row)
        print(f"  R{r}: {ROUTE_NAMES[r]:<18} "
              f"{row['soft_match']:>7.2f}% {row['clinical_adequacy']:>7.2f}% "
              f"{row['bleu_1']:>7.2f}% {row['bleu_2']:>7.2f}% "
              f"{row['fuzzy_match']:>7.2f}% {row['n']:>5}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    print(f"  {'OVERALL':<22} "
          f"{df['soft_match'].mean()*100:>7.2f}% "
          f"{df['clinical_adequacy'].mean()*100:>7.2f}% "
          f"{df['bleu_1'].mean()*100:>7.2f}% "
          f"{df['bleu_2'].mean()*100:>7.2f}% "
          f"{df['fuzzy_match'].mean()*100:>7.2f}% "
          f"{len(df):>5}")

    print(f"\n{'='*72}\n  SECONDARY METRICS (strict baselines)\n{'='*72}\n")
    print(f"  {'Route':<22} {'Tok F1':>8} {'Exact':>8} {'N':>5}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*5}")
    for r in range(6):
        sub = df[df["route"] == r]
        if len(sub) == 0: continue
        print(f"  R{r}: {ROUTE_NAMES[r]:<18} "
              f"{sub['token_f1'].mean()*100:>7.2f}% "
              f"{sub['exact'].mean()*100:>7.2f}% {len(sub):>5}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*5}")
    print(f"  {'OVERALL':<22} "
          f"{df['token_f1'].mean()*100:>7.2f}% "
          f"{df['exact'].mean()*100:>7.2f}% {len(df):>5}")

    pd.DataFrame(summary).to_csv(
        os.path.join(CFG["out_dir"], "pipeline_metrics_summary.csv"),
        index=False)

    print(f"\n{'='*72}\n  GENERATION QUALITY CATEGORIES\n{'='*72}\n")
    cat = df["category"].value_counts()
    total = len(df)
    for c in ["PERFECT", "FLUENT_CORRECT", "FLUENT_INCORRECT", "POOR"]:
        n = cat.get(c, 0); pct = n/total*100
        print(f"  {c:<18} {n:>4} ({pct:5.2f}%)  {'█'*int(pct/2)}")

    print(f"\n{'='*72}\n  LATENCY BREAKDOWN\n{'='*72}\n")
    lat_df = pd.DataFrame(latencies)
    lat_df.to_csv(os.path.join(CFG["out_dir"], "pipeline_latency.csv"),
                  index=False)
    print(f"  Stages 1-4 (mean):  {lat_df['s4_latency_ms'].mean():>8.1f} ms")
    print(f"  Stage 5    (mean):  {lat_df['s5_latency_ms'].mean():>8.1f} ms")
    total_lat = (lat_df['s4_latency_ms'].mean()
                  + lat_df['s5_latency_ms'].mean())
    print(f"  Full pipeline    :  {total_lat:>8.1f} ms")

    # Sample outputs
    print(f"\n{'='*72}\n  SAMPLE OUTPUTS (best + worst per route)\n{'='*72}\n")
    sl = []
    for r in range(6):
        sub = df[df["route"] == r]
        if len(sub) == 0: continue
        for label, row in [("BEST ", sub.nlargest(1, "soft_match").iloc[0]),
                            ("WORST", sub.nsmallest(1, "soft_match").iloc[0])]:
            print(f"  [R{r} {ROUTE_NAMES[r]}] {label}  "
                  f"(Soft: {row['soft_match']*100:.0f}%, "
                  f"Clin: {row['clinical_adequacy']*100:.0f}%, "
                  f"BLEU1: {row['bleu_1']*100:.0f}%)")
            print(f"    Q  : {row['question'][:90]}")
            print(f"    S4 : {row['s4_answer'][:90]}")
            print(f"    S5 : {row['s5_sentence'][:140]}")
            print(f"    GT : {row['gt_sentence'][:140]}\n")
            sl.append({"route": r, "label": label,
                       "question": row["question"],
                       "s4": row["s4_answer"], "s5": row["s5_sentence"],
                       "gt": row["gt_sentence"],
                       "soft_match": row["soft_match"],
                       "clinical_adequacy": row["clinical_adequacy"]})

    with open(os.path.join(CFG["out_dir"], "pipeline_examples.txt"), "w") as f:
        f.write("Stage 5 Pipeline — Sample Outputs\n" + "="*72 + "\n\n")
        for s in sl:
            f.write(f"[R{s['route']} {ROUTE_NAMES[s['route']]}] {s['label']}\n")
            f.write(f"  Q : {s['question']}\n  S4: {s['s4']}\n")
            f.write(f"  S5: {s['s5']}\n  GT: {s['gt']}\n\n")

    print(f"\n{'='*72}\n  FINAL SCORECARD\n{'='*72}\n")
    print(f"  Samples evaluated   : {len(df):,}\n")
    print(f"  ┌───────────────────────────────────────────┐")
    print(f"  │ PRIMARY (thesis-recommended):             │")
    print(f"  │  Soft Match         : {df['soft_match'].mean()*100:6.2f}%        │")
    print(f"  │  Clinical Adequacy  : {df['clinical_adequacy'].mean()*100:6.2f}%        │")
    print(f"  │  BLEU-1             : {df['bleu_1'].mean()*100:6.2f}%        │")
    print(f"  │  BLEU-2             : {df['bleu_2'].mean()*100:6.2f}%        │")
    if HAVE_RAPIDFUZZ:
        print(f"  │  Fuzzy Match        : {df['fuzzy_match'].mean()*100:6.2f}%        │")
    print(f"  │                                           │")
    print(f"  │ SECONDARY (strict baselines):             │")
    print(f"  │  Token F1           : {df['token_f1'].mean()*100:6.2f}%        │")
    print(f"  │  Exact Match        : {df['exact'].mean()*100:6.2f}%        │")
    print(f"  └───────────────────────────────────────────┘\n")
    fluent = (cat.get('PERFECT', 0) + cat.get('FLUENT_CORRECT', 0)
              + cat.get('FLUENT_INCORRECT', 0))
    print(f"  Fluent sentences    : {fluent}/{total} ({fluent/total*100:.1f}%)")
    print(f"  Mean total latency  : {total_lat:.1f} ms\n")
    print(f"  📊  HEADLINE: {df['soft_match'].mean()*100:.1f}% Soft Match  |  "
          f"{df['clinical_adequacy'].mean()*100:.1f}% Clinical Adequacy\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                         choices=["demo", "interactive", "bulk", "all"])
    parser.add_argument("--image", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    predictor = FullPipelinePredictor()
    if args.mode in ["demo", "all"]:    demo_mode(predictor)
    if args.mode == "interactive":      interactive_mode(predictor, args.image,
                                                          args.question)
    if args.mode in ["bulk", "all"]:    bulk_mode(predictor, args.n_samples)


if __name__ == "__main__":
    main()
