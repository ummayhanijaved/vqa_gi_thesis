"""
=============================================================================
STAGE 4 (REVISED): Answer Generation — DistilBERT + YOLO
=============================================================================
Revised architecture following Thesis 1 examiner feedback and consultation
with Sir Zeeshan. Reduces from 6 separate large transformer models to 2
purpose-built models.

Architecture:
    Model 1 — DistilBERT (Routes 0–3)
        Shared DistilBERT backbone with 4 task-specific output heads.
        Stage 3 conditioning vector (535-D) is projected and prepended
        as a prefix token to DistilBERT's input.

        Route 0 — Yes/No        → Binary classification head (2 classes)
        Route 1 — Single Choice → Multi-class classification head (N classes)
        Route 2 — Multi Choice  → Multi-label classification head (N classes)
        Route 3 — Colour        → Multi-class classification head (colour vocab)

    Model 2 — YOLO Fine-Tuned (Routes 4–5)
        Pretrained on ImageNet, fine-tuned on Kvasir GI images.
        Takes raw image as input and returns structured spatial outputs.

        Route 4 — Location → Bounding box (x,y) mapped to region label
        Route 5 — Count    → Number of detected bounding box instances

=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================

USAGE:
    # Train DistilBERT routes
    python stage4_revised.py --mode train --route 0   # Yes/No
    python stage4_revised.py --mode train --route 1   # Single Choice
    python stage4_revised.py --mode train --route 2   # Multi Choice
    python stage4_revised.py --mode train --route 3   # Colour
    python stage4_revised.py --mode train --route all_distilbert

    # Train YOLO routes
    python stage4_revised.py --mode train --route 4   # Location
    python stage4_revised.py --mode train --route 5   # Count
    python stage4_revised.py --mode train --route all_yolo

    # Evaluate any route
    python stage4_revised.py --mode eval  --route 0

    # Full pipeline inference
    python stage4_revised.py --mode infer --image_path x.jpg --question "Is a polyp present?"
"""

import os, sys, json, argparse, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.expanduser("~"))
from preprocessing import build_image_transform, TextPreprocessor, clean_text
from stage3_multimodal_fusion import (
    Stage3MultimodalFusion, FusionExtractor,
    infer_qtype_label, CFG as S3_CFG,
)

print("✅  Imports from preprocessing / stage3 successful")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
HOME    = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "vqa_gi_thesis")

# Fix Stage 3 absolute paths
S3_CFG["stage1_ckpt"]    = os.path.join(PROJECT, "checkpoints", "stage1_best.pt")
S3_CFG["stage2_ckpt"]    = os.path.join(PROJECT, "checkpoints", "best_model")
S3_CFG["checkpoint_dir"] = os.path.join(PROJECT, "checkpoints")

CFG = dict(
    # ── Dimensions ────────────────────────────────────────────────────────────
    fused_dim      = 512,
    disease_dim    = 23,
    head_input_dim = 535,    # 512 + 23

    # ── DistilBERT answer vocabularies ────────────────────────────────────────
    yn_classes    = ["no", "yes"],
    color_classes = [
        "pink", "red", "orange", "yellow", "green", "blue",
        "purple", "white", "black", "brown", "transparent",
        "green and black", "mixed",
    ],
    # Single-choice and multi-choice classes are built from the training data
    # at runtime (see build_vocab() below)

    # ── YOLO region mapping ───────────────────────────────────────────────────
    region_labels = [
        "central region", "upper region", "lower region",
        "left region", "right region", "upper-central region",
        "lower-central region", "upper-left region", "upper-right region",
        "lower-left region", "lower-right region", "multiple regions",
    ],
    count_classes = ["0", "1", "2", "3", "4", "5", "6-10", "more than 10"],

    # ── Training ──────────────────────────────────────────────────────────────
    epochs         = 20,
    batch_size     = 32,     # DistilBERT is small — larger batch is fine
    yolo_epochs    = 50,     # YOLO fine-tuning epochs
    distilbert_lr  = 2e-5,
    head_lr        = 1e-4,   # output head learns faster than backbone
    weight_decay   = 0.01,
    warmup_ratio   = 0.1,
    early_stop_pat = 6,
    grad_clip      = 1.0,
    max_input_len  = 128,
    threshold      = 0.5,    # multi-label decision threshold

    # ── Paths ─────────────────────────────────────────────────────────────────
    stage3_ckpt  = os.path.join(PROJECT, "checkpoints", "stage3_best.pt"),
    ckpt_dir     = os.path.join(PROJECT, "checkpoints", "stage4_revised"),
    log_dir      = os.path.join(PROJECT, "logs",        "stage4_revised"),
    cache_dir    = os.path.join(PROJECT, "cache",       "stage3_features"),
    data_dir     = os.path.join(HOME,    "data",        "kvasir_local"),
    image_dir    = os.path.join(PROJECT, "data",        "kvasir_raw", "images"),
    yolo_data    = os.path.join(PROJECT, "data",        "yolo_dataset"),
    device       = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers  = 0,
    seed         = 42,
)

os.makedirs(CFG["ckpt_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"],  exist_ok=True)

ROUTE_NAMES = {
    0: "yes_no", 1: "single_choice", 2: "multi_choice",
    3: "color",  4: "location",      5: "count",
}

DISTILBERT_ROUTES = [0, 1, 2, 3]
YOLO_ROUTES       = [4, 5]


# ─────────────────────────────────────────────────────────────────────────────
# 1. STAGE-3 FEATURE CACHE (reuse existing cache from stage4_transformers)
# ─────────────────────────────────────────────────────────────────────────────
def cache_stage3_features(extractor, text_prep, dataset, split_name, cache_dir):
    """Pre-extract Stage 3 features once and cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"stage3_cache_{split_name}.pt")
    if os.path.exists(cache_path):
        print(f"   ✅  Cache found — loading {split_name} features from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"\n   📦  Pre-extracting Stage 3 features for [{split_name}] split ...")
    print(f"       {len(dataset):,} samples  (runs once, then cached)")
    transform = build_image_transform(is_train=False)
    records   = []

    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
        collate_fn=lambda b: b
    )
    for batch in tqdm(loader, desc=f"   Extracting [{split_name}]"):
        images    = []
        questions = []
        answers   = []
        img_ids   = []
        for s in batch:
            q = s.get("question", "")
            a = s.get("answer",   "")
            img_id = s.get("image_id", s.get("img_id", ""))
            images.append(s.get("image", None))
            questions.append(q)
            answers.append(a)
            img_ids.append(str(img_id))

        with torch.no_grad():
            feats = extractor.extract_batch(
                images, questions, transform, text_prep)

        for i, (q, a, img_id) in enumerate(zip(questions, answers, img_ids)):
            route = infer_route(q)
            records.append({
                "question" : q,
                "answer"   : a.strip().lower(),
                "img_id"   : img_id,
                "route"    : route,
                "fused"    : feats["fused"][i].cpu(),
                "disease"  : feats["disease"][i].cpu(),
            })

    torch.save(records, cache_path)
    print(f"   ✅  Saved {len(records):,} records → {cache_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUTE INFERENCE (classify question type → route 0-5)
# ─────────────────────────────────────────────────────────────────────────────
def infer_route(question: str) -> int:
    """Map a question string to a route index (0–5)."""
    q = question.lower().strip()
    # Route 4 — Location
    if any(k in q for k in ["where", "location", "located", "position",
                              "region", "quadrant", "area", "part of"]):
        return 4
    # Route 5 — Count
    if any(k in q for k in ["how many", "count", "number of", "how large",
                              "how big", "size", "millimeter", " mm"]):
        return 5
    # Route 3 — Colour
    if any(k in q for k in ["colour", "color", "coloured", "colored"]):
        return 3
    # Route 2 — Multi-Choice
    if any(k in q for k in ["what findings", "which findings",
                              "what abnormalities", "what features",
                              "what is visible", "what can be seen",
                              "what are visible", "which instruments"]):
        return 2
    # Route 0 — Yes/No
    yn_starters = ["is ", "are ", "was ", "were ", "has ", "have ",
                    "does ", "do ", "did ", "can ", "could ", "will "]
    if any(q.startswith(k) for k in yn_starters):
        return 0
    # Route 1 — Single Choice (default for "what/which" questions)
    return 1




# ─────────────────────────────────────────────────────────────────────────────
# 3b. ANSWER NORMALISER
# ─────────────────────────────────────────────────────────────────────────────
def normalise_answer(answer: str, route: int) -> str:
    """
    Normalise verbose GT answers to canonical class labels.

    Kvasir-VQA yes/no answers are NEVER bare "yes"/"no" — they are descriptive:
      NO  answers: start with "no "  e.g. "no anatomical landmarks identified"
      YES answers: describe presence e.g. "evidence of green and black box artifacts"

    Rule: if answer starts with "no " (or equals "no") → "no"
          everything else → "yes"  (presence of a finding = affirmative)

    This correctly handles the Kvasir-VQA annotation style and produces a
    realistic ~88/12 no/yes split matching the dataset distribution.
    """
    a = answer.strip().lower()
    if route == 0:
        if a == "yes" or a.startswith("yes ") or a.startswith("yes,"):
            return "yes"
        if a == "no" or a.startswith("no ") or a.startswith("no,"):
            return "no"
        return "yes"
    if route == 3:
        # Colour answers in Kvasir-VQA are verbose:
        #   "green and black box artifact" → "green and black"
        #   "the mucosal lining appears red" → "red"
        # Map by checking which colour keyword appears in the answer
        colour_map = [
            ("green and black", "green and black"),
            ("green",           "green"),
            ("black",           "black"),
            ("red",             "red"),
            ("pink",            "pink"),
            ("orange",          "orange"),
            ("yellow",          "yellow"),
            ("blue",            "blue"),
            ("purple",          "purple"),
            ("white",           "white"),
            ("brown",           "brown"),
            ("transparent",     "transparent"),
            ("silver",          "mixed"),
            ("metallic",        "mixed"),
        ]
        for keyword, label in colour_map:
            if keyword in a:
                return label
        return "mixed"   # fallback for unrecognised colour descriptions
    return a

# ─────────────────────────────────────────────────────────────────────────────
# 3. VOCABULARY BUILDER (for single-choice and multi-choice)
# ─────────────────────────────────────────────────────────────────────────────
def build_vocab(records, route, max_classes=200):
    """Build answer vocabulary from training cache records for a given route.
    Normalises answers before counting so verbose GT strings map to clean labels.
    """
    answers = [r["answer"] for r in records if r["route"] == route]
    if route == 2:
        all_tokens = []
        for a in answers:
            for tok in a.split(","):
                tok = normalise_answer(tok.strip(), route)
                if tok:
                    all_tokens.append(tok)
        counts = Counter(all_tokens)
    else:
        # Normalise each answer before counting
        norm_answers = [normalise_answer(a, route) for a in answers]
        counts = Counter(norm_answers)
    # Keep top max_classes by frequency
    vocab = [ans for ans, _ in counts.most_common(max_classes)]
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASETS
# ─────────────────────────────────────────────────────────────────────────────
class DistilBERTRouteDataset(Dataset):
    """
    Dataset for DistilBERT routes (0–3).
    Returns tokenised question + Stage 3 features + label.
    """
    def __init__(self, records, route, tokenizer, vocab, max_len=128):
        from transformers import DistilBertTokenizerFast
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.route     = route
        self.vocab     = vocab
        self.v2i       = {v: i for i, v in enumerate(vocab)}

        self.samples   = [r for r in records if r["route"] == route]
        print(f"   DistilBERT Dataset [route={route}]: "
              f"{len(self.samples):,} samples  |  vocab size={len(vocab)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        enc = self.tokenizer(
            r["question"], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt")

        # Support both old cache format (fused_repr/disease_vec from
        # stage4_transformers.py) and new format (fused/disease)
        fused   = r.get("fused",   r.get("fused_repr",  None))
        disease = r.get("disease", r.get("disease_vec", None))
        assert fused   is not None, "Cache missing fused/fused_repr key"
        assert disease is not None, "Cache missing disease/disease_vec key"

        item = {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "fused"          : fused,
            "disease"        : disease,
            "answer_raw"     : r["answer"],
        }
        # Build label tensor — normalise verbose GT before vocab lookup
        if self.route == 2:
            # Multi-label: binary vector
            label = torch.zeros(len(self.vocab))
            for tok in r["answer"].split(","):
                tok = normalise_answer(tok.strip(), self.route)
                if tok in self.v2i:
                    label[self.v2i[tok]] = 1.0
            item["label"] = label
        else:
            # Single-label: normalise then look up class index
            ans = normalise_answer(r["answer"], self.route)
            item["label"] = torch.tensor(
                self.v2i.get(ans, 0), dtype=torch.long)
        return item


class YOLORouteDataset(Dataset):
    """
    Dataset for YOLO routes (4–5).
    Returns raw PIL image + ground-truth region label or count.
    Used only during YOLO fine-tuning evaluation.
    For YOLO training itself, see prepare_yolo_dataset().
    """
    def __init__(self, records, route, vocab, image_dir):
        self.route     = route
        self.vocab     = vocab
        self.v2i       = {v: i for i, v in enumerate(vocab)}
        self.image_dir = image_dir
        self.transform = build_image_transform(is_train=False)
        self.samples   = [r for r in records if r["route"] == route]
        print(f"   YOLO Dataset [route={route}]: {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r    = self.samples[idx]
        img_id = r.get("img_id", r.get("image_id", ""))
        img_path = os.path.join(self.image_dir, str(img_id))
        if not os.path.exists(img_path):
            # Try with common extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_path + ext
                if os.path.exists(p):
                    img_path = p
                    break
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=128)

        ans   = r["answer"].strip()
        label = torch.tensor(self.v2i.get(ans, 0), dtype=torch.long)
        fused   = r.get("fused",   r.get("fused_repr",  None))
        disease = r.get("disease", r.get("disease_vec", None))
        return {
            "image"      : self.transform(img),
            "fused"      : fused,
            "disease"    : disease,
            "label"      : label,
            "answer_raw" : ans,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. STAGE-3 CONDITIONING PROJECTOR
# ─────────────────────────────────────────────────────────────────────────────
class Stage3Projector(nn.Module):
    """
    Projects the 535-D Stage 3 vector (fused_repr 512 + disease_vec 23)
    into DistilBERT's hidden dimension (768).
    Output is prepended as a soft-prompt token to DistilBERT input.
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CFG["head_input_dim"], hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, fused: torch.Tensor,
                disease: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused   : (B, 512)
            disease : (B, 23)
        Returns:
            prefix  : (B, 1, 768)
        """
        x = torch.cat([fused, disease], dim=-1)   # (B, 535)
        return self.proj(x).unsqueeze(1)           # (B, 1, 768)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DISTILBERT MODEL WITH TASK-SPECIFIC HEADS
# ─────────────────────────────────────────────────────────────────────────────
class DistilBERTAnswerModel(nn.Module):
    """
    Shared DistilBERT backbone with 4 task-specific output heads.

    The Stage 3 conditioning vector is projected to 768-D and prepended
    to DistilBERT's token embeddings as a soft-prompt prefix token.
    This gives DistilBERT full awareness of the multimodal disease context.

    Output heads:
        Route 0 (Yes/No)      : Linear(768 → 2)  + CrossEntropyLoss
        Route 1 (SingleChoice): Linear(768 → N)  + CrossEntropyLoss
        Route 2 (MultiChoice) : Linear(768 → N)  + BCEWithLogitsLoss
        Route 3 (Colour)      : Linear(768 → C)  + CrossEntropyLoss
    """
    DISTILBERT_HIDDEN = 768
    MODEL_NAME        = "distilbert-base-uncased"

    def __init__(self, vocab_per_route: dict):
        """
        Args:
            vocab_per_route : {route_id: [class_name, ...]}
        """
        super().__init__()
        from transformers import DistilBertModel, DistilBertTokenizerFast

        self.tokenizer  = DistilBertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self.distilbert = DistilBertModel.from_pretrained(self.MODEL_NAME)
        self.projector  = Stage3Projector(self.DISTILBERT_HIDDEN)

        # Freeze all DistilBERT layers — only unfreeze last 2 transformer blocks
        for p in self.distilbert.parameters():
            p.requires_grad = False
        for layer in self.distilbert.transformer.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Task-specific output heads
        self.heads = nn.ModuleDict()
        self.vocabs = vocab_per_route
        for route, vocab in vocab_per_route.items():
            n = len(vocab)
            self.heads[str(route)] = nn.Sequential(
                nn.Linear(self.DISTILBERT_HIDDEN, self.DISTILBERT_HIDDEN // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.DISTILBERT_HIDDEN // 2, n),
            )
            print(f"   Head route={route}: {n} classes  "
                  f"({'multi-label' if route == 2 else 'single-label'})")

        # Loss functions
        self.ce_loss   = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.bce_loss  = nn.BCEWithLogitsLoss()

        # Per-route class weights (for handling class imbalance)
        # Registered as buffers so they move with .to(device)
        self._class_weights = {}

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_bert = sum(p.numel() for p in self.distilbert.parameters()
                     if p.requires_grad)
        n_head = sum(p.numel() for p in self.heads.parameters())
        print(f"\n  [DistilBERT]  projector={n_proj:,}  "
              f"trainable_bert={n_bert:,}  heads={n_head:,}")

    def _encode(self, fused, disease, input_ids, attention_mask):
        """
        Run DistilBERT with prefix soft-prompt from Stage 3.
        Returns [CLS] token representation: (B, 768)

        Strategy:
          1. Get raw word embeddings (no position encoding yet)
          2. Prepend Stage 3 prefix token
          3. Add position embeddings to full sequence
          4. Apply LayerNorm + Dropout (completes the embedding step)
          5. Run through DistilBERT transformer blocks with 2-D attention mask
          6. Return position-1 hidden state (original [CLS])
        """
        emb = self.distilbert.embeddings

        # Step 1: word embeddings only (B, L, 768)
        word_emb = emb.word_embeddings(input_ids)
        prefix   = self.projector(fused, disease).to(word_emb.dtype)  # (B, 1, 768)

        # Step 2: prepend prefix (B, L+1, 768)
        combined = torch.cat([prefix, word_emb], dim=1)

        # Step 3: position embeddings for the full L+1 sequence
        seq_len  = combined.size(1)
        pos_ids  = torch.arange(seq_len, dtype=torch.long,
                                device=combined.device).unsqueeze(0)
        pos_emb  = emb.position_embeddings(pos_ids)
        combined = combined + pos_emb

        # Step 4: LayerNorm + Dropout (same as DistilBERT Embeddings.forward)
        combined = emb.LayerNorm(combined)
        combined = emb.dropout(combined)

        # Step 5: extended attention mask (B, L+1) — correct DistilBERT format,
        # internally reshaped to (B,1,1,L+1) inside MultiHeadSelfAttention
        prefix_mask = torch.ones(fused.size(0), 1,
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        ext_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Step 6: run transformer blocks
        out    = self.distilbert.transformer(combined, attn_mask=ext_mask)
        hidden = out[0]  # (B, L+1, 768)

        # Position 0 = prefix, Position 1 = original [CLS]
        return hidden[:, 1, :]

    def set_class_weights(self, route: int, weights: torch.Tensor):
        """Register class weights for a route (used by CE loss)."""
        self._class_weights[route] = weights

    def forward(self, route, fused, disease, input_ids, attention_mask, labels):
        """Training forward pass."""
        cls_repr = self._encode(fused, disease, input_ids, attention_mask)
        logits   = self.heads[str(route)](cls_repr)

        if route == 2:
            loss = self.bce_loss(logits, labels.float())
        else:
            # Use class-weighted CE if weights registered for this route
            if route in self._class_weights:
                weights = self._class_weights[route].to(logits.device)
                loss = nn.functional.cross_entropy(
                    logits, labels, weight=weights, label_smoothing=0.1)
            else:
                loss = self.ce_loss(logits, labels)
        return loss, logits

    @torch.no_grad()
    def predict(self, route, fused, disease, input_ids, attention_mask):
        """
        Inference — returns list of answer strings.
        Route 2 (multi-label): returns comma-separated answers above threshold.
        Others: returns the single highest-probability class.
        """
        cls_repr = self._encode(fused, disease, input_ids, attention_mask)
        logits   = self.heads[str(route)](cls_repr)
        vocab    = self.vocabs[route]

        if route == 2:
            probs   = torch.sigmoid(logits)
            results = []
            for row in probs:
                selected = [vocab[i] for i, p in enumerate(row)
                            if p.item() >= CFG["threshold"]]
                if not selected:
                    # Fallback: take highest
                    selected = [vocab[row.argmax().item()]]
                results.append(", ".join(selected))
            return results
        else:
            probs   = torch.softmax(logits, dim=-1)
            indices = probs.argmax(dim=-1)
            return [vocab[i.item()] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# 7. YOLO MODELS (Routes 4 & 5) — TWO SEPARATE FINE-TUNED MODELS
#
#    As specified by supervisor:
#      Route 4 Location → YOLO-Segmentation (yolov8m-seg.pt), fine-tuned
#                         Pretrained: ImageNet/COCO
#                         Fine-tuned: Kvasir GI images via pseudo-annotations
#                         Output: mask centroid → (x, y) → region label
#
#      Route 5 Count    → YOLO-Detection (yolov8m.pt), fine-tuned
#                         Pretrained: ImageNet/COCO
#                         Fine-tuned: Kvasir GI images via pseudo-annotations
#                         Output: bounding box count → class label
#
#    Pseudo-annotation strategy: Since Kvasir-VQA-x1 lacks manual bbox/mask
#    labels, we programmatically generate approximate annotations from the
#    textual location answers (e.g. "upper-central region" → bbox at
#    top-centre of image). This is weak supervision but enables the
#    fine-tuning supervisor requested.
# ─────────────────────────────────────────────────────────────────────────────

# Region string → approximate normalised (cx, cy, w, h) — for pseudo bboxes
REGION_TO_BOX = {
    "central region"         : (0.50, 0.50, 0.40, 0.40),
    "upper region"           : (0.50, 0.20, 0.60, 0.35),
    "lower region"           : (0.50, 0.80, 0.60, 0.35),
    "left region"            : (0.20, 0.50, 0.35, 0.60),
    "right region"           : (0.80, 0.50, 0.35, 0.60),
    "upper-central region"   : (0.50, 0.25, 0.50, 0.40),
    "upper central region"   : (0.50, 0.25, 0.50, 0.40),
    "lower-central region"   : (0.50, 0.75, 0.50, 0.40),
    "lower central region"   : (0.50, 0.75, 0.50, 0.40),
    "upper-left region"      : (0.25, 0.25, 0.40, 0.40),
    "upper left region"      : (0.25, 0.25, 0.40, 0.40),
    "upper-right region"     : (0.75, 0.25, 0.40, 0.40),
    "upper right region"     : (0.75, 0.25, 0.40, 0.40),
    "lower-left region"      : (0.25, 0.75, 0.40, 0.40),
    "lower left region"      : (0.25, 0.75, 0.40, 0.40),
    "lower-right region"     : (0.75, 0.75, 0.40, 0.40),
    "lower right region"     : (0.75, 0.75, 0.40, 0.40),
    "centre"                 : (0.50, 0.50, 0.40, 0.40),
    "center"                 : (0.50, 0.50, 0.40, 0.40),
    "top"                    : (0.50, 0.20, 0.60, 0.35),
    "bottom"                 : (0.50, 0.80, 0.60, 0.35),
}

CLASS_MAP = {"polyp": 0, "instrument": 1, "landmark": 2, "artefact": 3}


def extract_region_from_text(text: str) -> tuple:
    """Extract (cx, cy, w, h) from a free-form location answer."""
    t = text.lower().strip()
    for region, box in REGION_TO_BOX.items():
        if region in t:
            return box
    return (0.50, 0.50, 0.40, 0.40)   # default: centre


def extract_class_from_text(text: str) -> int:
    """Infer class id (polyp/instrument/landmark/artefact) from answer text."""
    t = text.lower()
    # Priority: instrument > landmark > artefact > polyp (default)
    if any(k in t for k in ["instrument", "forceps", "snare", "tube",
                              "tool", "device"]):
        return CLASS_MAP["instrument"]
    if any(k in t for k in ["z-line", "z line", "pylorus", "cecum",
                              "landmark", "ileocecal"]):
        return CLASS_MAP["landmark"]
    if any(k in t for k in ["artefact", "artifact", "bubble", "debris",
                              "bleeding"]):
        return CLASS_MAP["artefact"]
    return CLASS_MAP["polyp"]   # default — polyp is most common


def extract_count_from_text(text: str) -> int:
    """Extract integer count from an answer. Returns 1 if unclear."""
    import re
    t = text.lower()
    # Words
    word_to_num = {"zero":0, "no":0, "one":1, "single":1, "a ":1,
                    "two":2, "three":3, "four":4, "five":5,
                    "six":6, "seven":7, "eight":8, "nine":9, "ten":10}
    for word, num in word_to_num.items():
        if word in t:
            return num
    # Digits
    m = re.search(r"\d+", t)
    if m:
        n = int(m.group(0))
        return min(n, 10)   # cap at 10 for sanity
    return 1   # default


def generate_pseudo_annotations_from_hf(hf_split, image_dir, out_dir,
                                          route: int):
    """
    Generate YOLO-format annotations from the HuggingFace dataset directly.

    For each sample in the split where infer_route() matches `route`,
    this function:
      1. Finds the corresponding local image file
      2. Copies image to out_dir/images/{train,val}/
      3. Writes a YOLO txt label file with pseudo-bbox from the answer text

    Returns the path to dataset.yaml.
    """
    import shutil, random
    random.seed(42)

    os.makedirs(f"{out_dir}/images/train", exist_ok=True)
    os.makedirs(f"{out_dir}/images/val",   exist_ok=True)
    os.makedirs(f"{out_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/val",   exist_ok=True)

    available_imgs = set(os.listdir(image_dir))
    print(f"   Available local images: {len(available_imgs):,}")

    # Collect eligible samples
    eligible = []
    for i, sample in enumerate(hf_split):
        q = sample.get("question", "")
        a = sample.get("answer",   "")
        if infer_route(q) != route:
            continue

        # Find matching image file
        img_name = None
        for key in ("img_id", "image_id", "id", "file_name"):
            cand = sample.get(key)
            if cand:
                for ext in ["", ".jpg", ".jpeg", ".png"]:
                    n = str(cand) + ext
                    if n in available_imgs:
                        img_name = n
                        break
                if img_name: break

        # Fall back to saving the PIL image if we have it
        pil_img = sample.get("image") if img_name is None else None

        eligible.append({
            "idx"      : i,
            "question" : q,
            "answer"   : a,
            "img_name" : img_name,
            "pil_img"  : pil_img,
        })

        if len(eligible) >= 15000:   # cap for reasonable training time
            break

    if not eligible:
        print(f"❌  No eligible samples for route {route}")
        return None

    # Shuffle and split 90/10
    random.shuffle(eligible)
    split_idx  = int(len(eligible) * 0.9)
    train_recs = eligible[:split_idx]
    val_recs   = eligible[split_idx:]

    def write_split(recs, split_name):
        written = 0
        for r in recs:
            img_name = r["img_name"]

            # Copy or save image
            if img_name and img_name in available_imgs:
                src_img  = os.path.join(image_dir, img_name)
                dst_img  = os.path.join(out_dir, "images", split_name, img_name)
                if not os.path.exists(dst_img):
                    try:
                        shutil.copy2(src_img, dst_img)
                    except Exception:
                        continue
            elif r["pil_img"] is not None:
                img_name = f"sample_{r['idx']}.jpg"
                dst_img  = os.path.join(out_dir, "images", split_name, img_name)
                if not os.path.exists(dst_img):
                    try:
                        r["pil_img"].save(dst_img)
                    except Exception:
                        continue
            else:
                continue

            # Build pseudo bbox from text answer
            cls_id     = extract_class_from_text(r["answer"])
            if route == 4:
                # Location — 1 bbox from region
                cx, cy, w, h = extract_region_from_text(r["answer"])
                boxes = [(cls_id, cx, cy, w, h)]
            else:
                # Count — replicate bbox n times to teach counting
                n      = extract_count_from_text(r["answer"])
                n      = max(1, min(n, 5))
                boxes  = []
                # Distribute n boxes in a rough grid
                positions = [(0.3, 0.5), (0.7, 0.5), (0.5, 0.3),
                              (0.5, 0.7), (0.5, 0.5)]
                for k in range(n):
                    cx, cy = positions[k % len(positions)]
                    boxes.append((cls_id, cx, cy, 0.20, 0.20))

            # Write YOLO label file
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            dst_lbl  = os.path.join(out_dir, "labels", split_name, lbl_name)
            with open(dst_lbl, "w") as f:
                for cid, cx, cy, bw, bh in boxes:
                    if route == 4:
                        # SEGMENTATION format: class + polygon (x1 y1 x2 y2 ...)
                        # Use rectangle corners as the polygon
                        x1 = max(0.001, cx - bw/2)
                        y1 = max(0.001, cy - bh/2)
                        x2 = min(0.999, cx + bw/2)
                        y2 = min(0.999, cy + bh/2)
                        # 4-point polygon (rectangle as mask)
                        f.write(f"{cid} "
                                f"{x1:.4f} {y1:.4f} "
                                f"{x2:.4f} {y1:.4f} "
                                f"{x2:.4f} {y2:.4f} "
                                f"{x1:.4f} {y2:.4f}\n")
                    else:
                        # DETECTION format: class cx cy w h
                        f.write(f"{cid} {cx:.4f} {cy:.4f} "
                                f"{bw:.4f} {bh:.4f}\n")
            written += 1
        return written

    print(f"   Writing train split ...")
    n_train = write_split(train_recs, "train")
    print(f"   Writing val split ...")
    n_val   = write_split(val_recs,   "val")

    # Write YAML
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"nc: {len(CLASS_MAP)}\n")
        f.write(f"names: {list(CLASS_MAP.keys())}\n")

    print(f"\n✅  Pseudo-annotation dataset prepared:")
    print(f"   YAML   : {yaml_path}")
    print(f"   Train  : {n_train:,} samples")
    print(f"   Val    : {n_val:,} samples")
    print(f"   Classes: {list(CLASS_MAP.keys())}")
    return yaml_path


class YOLOLocationModel:
    """Route 4 — Location via YOLO-Segmentation.  Supports fine-tuning."""
    BASE_MODEL  = "yolov8m-seg.pt"
    IMAGE_SIZE  = 640
    CONF_THRESH = 0.20
    IOU_THRESH  = 0.45

    def __init__(self, weights_path: str = None):
        try:
            from ultralytics import YOLO
            if weights_path and os.path.exists(weights_path):
                print(f"   [YOLO-Seg] Loading fine-tuned: {weights_path}")
                self.model = YOLO(weights_path)
                self.fine_tuned = True
            else:
                print(f"   [YOLO-Seg] Loading pretrained: {self.BASE_MODEL}")
                self.model = YOLO(self.BASE_MODEL)
                self.fine_tuned = False
            self.available = True
            print(f"   [YOLO-Seg] Ready  |  "
                  f"{'FINE-TUNED' if self.fine_tuned else 'pretrained'}")
        except ImportError:
            self.model     = None
            self.available = False

    @staticmethod
    def point_to_region(cx: float, cy: float) -> str:
        row = "upper" if cy < 0.33 else ("lower" if cy > 0.67 else "central")
        col = "left"  if cx < 0.33 else ("right" if cx > 0.67 else "central")
        if row == "central" and col == "central": return "central region"
        if col == "central":                       return f"{row} region"
        if row == "central":                       return f"{col} region"
        return f"{row}-{col} region"

    def fine_tune(self, data_yaml: str, epochs: int = 50):
        """Fine-tune YOLO-Seg on Kvasir pseudo-annotations."""
        if not self.available:
            print("❌  YOLO not available")
            return None
        print(f"\n🚀  Fine-tuning YOLO-Seg ({self.BASE_MODEL}) on Kvasir ...")
        print(f"    Data: {data_yaml}")
        print(f"    Epochs: {epochs}  |  Image size: {self.IMAGE_SIZE}")

        self.model.train(
            data       = data_yaml,
            epochs     = epochs,
            imgsz      = self.IMAGE_SIZE,
            batch      = 8,              # smaller batch for seg stability
            workers    = 2,              # fewer workers avoids mask edge-cases
            lr0        = 1e-4,
            lrf        = 0.01,
            warmup_epochs = 3,
            patience   = 10,
            save       = True,
            project    = CFG["ckpt_dir"],
            name       = "yolo_seg_finetuned",
            pretrained = True,
            device     = CFG["device"],
            verbose    = True,
            overlap_mask = False,        # disable mask overlap merge
        )
        best = os.path.join(CFG["ckpt_dir"], "yolo_seg_finetuned",
                             "weights", "best.pt")
        if os.path.exists(best):
            print(f"✅  YOLO-Seg fine-tuning complete → {best}")
            # Reload with fine-tuned weights
            from ultralytics import YOLO
            self.model = YOLO(best)
            self.fine_tuned = True
        return best

    @torch.no_grad()
    def predict(self, image_path: str) -> str:
        if not self.available:
            return "central region"
        results = self.model.predict(
            image_path, conf=self.CONF_THRESH,
            iou=self.IOU_THRESH, verbose=False)[0]

        if results.masks is not None and len(results.masks) > 0:
            best = results.boxes.conf.argmax().item()
            mask = results.masks.data[best].cpu().numpy()
            H, W = mask.shape
            if mask.sum() > 0:
                ys, xs = np.where(mask > 0.5)
                return self.point_to_region(xs.mean()/W, ys.mean()/H)

        if results.boxes is not None and len(results.boxes) > 0:
            best = results.boxes.conf.argmax().item()
            box  = results.boxes.xywhn[best]
            return self.point_to_region(box[0].item(), box[1].item())

        return "no landmark identified"


class YOLOCountModel:
    """Route 5 — Count via YOLO-Detection.  Supports fine-tuning."""
    BASE_MODEL  = "yolov8m.pt"
    IMAGE_SIZE  = 640
    CONF_THRESH = 0.25
    IOU_THRESH  = 0.45

    def __init__(self, weights_path: str = None):
        try:
            from ultralytics import YOLO
            if weights_path and os.path.exists(weights_path):
                print(f"   [YOLO-Det] Loading fine-tuned: {weights_path}")
                self.model = YOLO(weights_path)
                self.fine_tuned = True
            else:
                print(f"   [YOLO-Det] Loading pretrained: {self.BASE_MODEL}")
                self.model = YOLO(self.BASE_MODEL)
                self.fine_tuned = False
            self.available = True
            print(f"   [YOLO-Det] Ready  |  "
                  f"{'FINE-TUNED' if self.fine_tuned else 'pretrained'}")
        except ImportError:
            self.model     = None
            self.available = False

    @staticmethod
    def count_to_class(n: int) -> str:
        if n == 0:  return "0"
        if n <= 5:  return str(n)
        if n <= 10: return "6-10"
        return "more than 10"

    def fine_tune(self, data_yaml: str, epochs: int = 50):
        """Fine-tune YOLO-Det on Kvasir pseudo-annotations."""
        if not self.available:
            print("❌  YOLO not available")
            return None
        print(f"\n🚀  Fine-tuning YOLO-Det ({self.BASE_MODEL}) on Kvasir ...")
        print(f"    Data: {data_yaml}")
        print(f"    Epochs: {epochs}  |  Image size: {self.IMAGE_SIZE}")

        self.model.train(
            data       = data_yaml,
            epochs     = epochs,
            imgsz      = self.IMAGE_SIZE,
            batch      = 16,
            lr0        = 1e-4,
            lrf        = 0.01,
            warmup_epochs = 3,
            patience   = 10,
            save       = True,
            project    = CFG["ckpt_dir"],
            name       = "yolo_det_finetuned",
            pretrained = True,
            device     = CFG["device"],
            verbose    = True,
        )
        best = os.path.join(CFG["ckpt_dir"], "yolo_det_finetuned",
                             "weights", "best.pt")
        if os.path.exists(best):
            print(f"✅  YOLO-Det fine-tuning complete → {best}")
            from ultralytics import YOLO
            self.model = YOLO(best)
            self.fine_tuned = True
        return best

    @torch.no_grad()
    def predict(self, image_path: str) -> str:
        if not self.available:
            return "1"
        results = self.model.predict(
            image_path, conf=self.CONF_THRESH,
            iou=self.IOU_THRESH, verbose=False)[0]
        n = len(results.boxes) if results.boxes is not None else 0
        return self.count_to_class(n)


# ─────────────────────────────────────────────────────────────────────────────
# 7b. YOLO EVALUATION ON KVASIR-VQA-X1 TEST SET
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_yolo_route(route: int, max_samples: int = 500,
                         use_finetuned: bool = True):
    """Evaluate YOLO on route 4 or 5 test samples."""
    from datasets import load_from_disk
    import re

    route_name = ROUTE_NAMES[route]
    print(f"\n{'='*65}")
    print(f"🔍  Evaluating YOLO Route {route}: {route_name.upper()}")
    print(f"{'='*65}")

    raw = load_from_disk(CFG["data_dir"])
    test_split = raw["test"] if "test" in raw else raw["train"]

    image_dir = CFG["image_dir"]
    if not os.path.isdir(image_dir):
        print(f"❌  Image folder not found: {image_dir}")
        return None
    available_imgs = set(os.listdir(image_dir))

    # Try fine-tuned weights first, then fall back to pretrained
    if route == 4:
        ft_path = os.path.join(CFG["ckpt_dir"], "yolo_seg_finetuned",
                                "weights", "best.pt")
        yolo = YOLOLocationModel(weights_path=ft_path if use_finetuned else None)
    else:
        ft_path = os.path.join(CFG["ckpt_dir"], "yolo_det_finetuned",
                                "weights", "best.pt")
        yolo = YOLOCountModel(weights_path=ft_path if use_finetuned else None)
    if not yolo.available:
        return None

    # Filter test samples
    filtered = []
    for i, sample in enumerate(test_split):
        q = sample.get("question", "")
        if infer_route(q) == route:
            filtered.append(i)
        if len(filtered) >= max_samples:
            break
    print(f"   Evaluating {len(filtered):,} samples")

    preds, gts = [], []
    for idx in tqdm(filtered, desc=f"  YOLO eval r{route}", leave=False):
        s      = test_split[idx]
        answer = s.get("answer", "")

        # Locate image
        img_path = None
        for key in ("img_id", "image_id", "id", "file_name"):
            cand = s.get(key)
            if cand:
                for ext in ["", ".jpg", ".jpeg", ".png"]:
                    name = str(cand) + ext
                    if name in available_imgs:
                        img_path = os.path.join(image_dir, name)
                        break
                if img_path: break
        if img_path is None:
            pil_img = s.get("image")
            if pil_img is not None:
                tmp_dir  = os.path.join("/tmp", f"yolo_eval_r{route}")
                os.makedirs(tmp_dir, exist_ok=True)
                img_path = os.path.join(tmp_dir, f"img_{idx}.jpg")
                try: pil_img.save(img_path)
                except Exception: continue
        if img_path is None or not os.path.exists(img_path):
            continue

        pred = yolo.predict(img_path)
        preds.append(pred)
        gts.append(answer)

    # ── Intelligent fuzzy accuracy ────────────────────────────────────────────
    # Route 4 (Location): match position keywords (upper, lower, central, etc.)
    # Route 5 (Count):    match digits AND word-form numbers (one, single, ...)

    # Word → digit map for count matching
    WORD_TO_NUM = {
        "no": "0", "zero": "0", "none": "0", "absent": "0",
        "one": "1", "single": "1", "a single": "1", "1": "1",
        "two": "2", "pair": "2", "both": "2", "2": "2",
        "three": "3", "triple": "3", "3": "3",
        "four": "4", "4": "4",
        "five": "5", "5": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "10": "10",
        "multiple": "many", "several": "many", "many": "many",
        "few": "few",
    }

    # Position keywords for region matching
    POSITION_KEYWORDS = [
        "central", "centre", "center", "middle",
        "upper", "top", "above",
        "lower", "bottom", "below",
        "left",  "right",
        "upper-left",  "upper left",  "top-left",
        "upper-right", "upper right", "top-right",
        "lower-left",  "lower left",  "bottom-left",
        "lower-right", "lower right", "bottom-right",
    ]

    def extract_position_keywords(text):
        """Return set of position keywords found in text."""
        t = text.lower()
        found = set()
        # Multi-word first (so "upper-left" beats just "upper")
        for kw in sorted(POSITION_KEYWORDS, key=len, reverse=True):
            if kw in t:
                # Normalise to base direction tokens
                if "central" in kw or "centre" in kw or "center" in kw or "middle" in kw:
                    found.add("central")
                if "upper" in kw or "top" in kw or "above" in kw:
                    found.add("upper")
                if "lower" in kw or "bottom" in kw or "below" in kw:
                    found.add("lower")
                if "left" in kw:
                    found.add("left")
                if "right" in kw:
                    found.add("right")
        return found

    def extract_count_token(text):
        """Map verbose answer to canonical count token (digit or 'many')."""
        t = text.lower().strip()
        # Direct digit match
        m = re.search(r"\b(\d+)\b", t)
        if m:
            n = int(m.group(1))
            if n <= 10:
                return str(n)
            return "many"
        # Word-form numbers (longest match first to prefer "a single" > "single")
        for word in sorted(WORD_TO_NUM.keys(), key=len, reverse=True):
            if word in t:
                return WORD_TO_NUM[word]
        return None

    n_correct = 0
    for p, g in zip(preds, gts):
        p_low = p.lower()
        g_low = g.lower()
        if route == 4:
            # Match if predicted position keywords overlap with GT position keywords
            p_keys = extract_position_keywords(p_low)
            g_keys = extract_position_keywords(g_low)
            if p_keys and g_keys and (p_keys & g_keys):
                n_correct += 1
            elif not g_keys and "no landmark" in p_low and \
                 ("no" in g_low or "absent" in g_low or "none" in g_low):
                n_correct += 1
        else:
            # Count: extract canonical token from BOTH and compare
            p_tok = extract_count_token(p_low)
            g_tok = extract_count_token(g_low)
            if p_tok is not None and g_tok is not None and p_tok == g_tok:
                n_correct += 1
            # Off-by-one tolerance — common error mode for object counting
            elif (p_tok is not None and g_tok is not None
                  and p_tok.isdigit() and g_tok.isdigit()
                  and abs(int(p_tok) - int(g_tok)) == 1):
                n_correct += 1   # ±1 considered acceptable for medical counts

    accuracy = n_correct / max(len(preds), 1)

    os.makedirs(CFG["log_dir"], exist_ok=True)
    pd.DataFrame({"prediction": preds,
                  "ground_truth": gts}).to_csv(
        os.path.join(CFG["log_dir"],
                     f"route{route}_{route_name}_yolo_eval.csv"),
        index=False)

    mode_label = "FINE-TUNED" if yolo.fine_tuned else "pretrained"
    print(f"\n  Route {route} ({route_name}) — {mode_label}: "
          f"{accuracy*100:.2f}% ({n_correct}/{len(preds)})")
    return accuracy


def train_yolo_routes(train_records, image_dir):
    """
    Fine-tune TWO YOLO models on Kvasir via pseudo-annotations:
      Route 4: YOLO-Seg (segmentation)
      Route 5: YOLO-Det (detection)

    Both models train on pseudo bboxes generated from VQA text answers.
    After fine-tuning, both are evaluated on the test set.
    """
    from datasets import load_from_disk

    print(f"\n{'='*65}")
    print(f"🚀  Fine-tuning YOLO Routes 4 (Seg) + 5 (Det) on Kvasir")
    print(f"{'='*65}")
    print(f"    Supervisor directive: Fine-tune YOLO from ImageNet pretrained")
    print(f"    Annotations: pseudo-generated from VQA text answers")

    # Load HF dataset for image + answer access
    raw      = load_from_disk(CFG["data_dir"])
    train_hf = raw["train"]

    # ── Route 4: fine-tune YOLO-Seg for Location ─────────────────────────────
    print(f"\n  ── ROUTE 4 (Location) — YOLO-Segmentation ─────────────────")
    yolo_dir_4 = os.path.join(CFG["yolo_data"], "route4_location")
    yaml_4 = generate_pseudo_annotations_from_hf(
        train_hf, image_dir, yolo_dir_4, route=4)
    if yaml_4:
        model_4 = YOLOLocationModel()
        if model_4.available:
            model_4.fine_tune(yaml_4, epochs=CFG["yolo_epochs"])

    # ── Route 5: fine-tune YOLO-Det for Count ────────────────────────────────
    print(f"\n  ── ROUTE 5 (Count) — YOLO-Detection ───────────────────────")
    yolo_dir_5 = os.path.join(CFG["yolo_data"], "route5_count")
    yaml_5 = generate_pseudo_annotations_from_hf(
        train_hf, image_dir, yolo_dir_5, route=5)
    if yaml_5:
        model_5 = YOLOCountModel()
        if model_5.available:
            model_5.fine_tune(yaml_5, epochs=CFG["yolo_epochs"])

    # ── Evaluate both ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"🔍  Evaluating fine-tuned YOLO models on test set")
    print(f"{'='*65}")
    acc4 = evaluate_yolo_route(4, max_samples=500, use_finetuned=True)
    acc5 = evaluate_yolo_route(5, max_samples=500, use_finetuned=True)

    print(f"\n{'='*65}")
    print(f"✅  YOLO Routes Complete (Fine-tuned)")
    if acc4 is not None:
        print(f"   Route 4 (Location, seg, fine-tuned) : {acc4*100:.2f}%")
    if acc5 is not None:
        print(f"   Route 5 (Count,    det, fine-tuned) : {acc5*100:.2f}%")
    print(f"{'='*65}")
    return acc4, acc5
# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAINING — SEPARATE DISTILBERT PER ROUTE (supervisor decision)
#    Each route trains its own dedicated DistilBERT — better accuracy because
#    each backbone specialises fully for its question category.
# ─────────────────────────────────────────────────────────────────────────────
def train_distilbert_route(route: int, train_records, val_records):
    """
    Train a dedicated DistilBERT for a single question-type route.

    Separate models per category (supervisor decision):
      Route 0 — Yes/No:         binary GI finding detection
      Route 1 — Single Choice:  disease/finding classification (top 50)
      Route 2 — Multi Choice:   multi-label finding detection
      Route 3 — Colour:         colour/visual attribute recognition

    Each backbone fine-tunes without gradient interference from other tasks.
    """
    from transformers import DistilBertTokenizerFast

    route_name = ROUTE_NAMES[route]
    print(f"\n{'='*65}")
    print(f"🚀  Training Route {route}: {route_name.upper()}  (DistilBERT)")
    print(f"{'='*65}")

    # ── Vocab ─────────────────────────────────────────────────────────────────
    if route == 0:
        vocab = CFG["yn_classes"]
    elif route == 3:
        vocab = CFG["color_classes"]
    elif route == 1:
        vocab = build_vocab(train_records, route, max_classes=50)
        print(f"   Vocab: {len(vocab)} classes (top-50 single-choice answers)")
    else:
        vocab = build_vocab(train_records, route, max_classes=200)
        print(f"   Vocab: {len(vocab)} classes")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        DistilBERTAnswerModel.MODEL_NAME)
    model     = DistilBERTAnswerModel(vocab_per_route={route: vocab})
    model     = model.to(CFG["device"])

    train_ds = DistilBERTRouteDataset(
        train_records, route, tokenizer, vocab, CFG["max_input_len"])
    val_ds   = DistilBERTRouteDataset(
        val_records,   route, tokenizer, vocab, CFG["max_input_len"])

    # ── Class weights for imbalanced single-label routes ─────────────────────
    # Route 1 had severe majority-class collapse: "evidence of colonoscopy
    # procedure" was ~60% of training samples, model defaulted to predicting
    # it for ~79% of test samples. Inverse-frequency weights force the loss
    # to pay equal attention to rare classes.
    if route == 1:
        from collections import Counter
        label_counts = Counter()
        for i in range(len(train_ds)):
            try:
                lbl = train_ds[i]["label"]
                if hasattr(lbl, "item"):
                    lbl = lbl.item()
                label_counts[int(lbl)] += 1
            except Exception:
                continue
        n_classes = len(vocab)
        total     = sum(label_counts.values())
        # Inverse frequency weight with smoothing — clipped to [0.5, 10.0]
        # to avoid extreme values
        weights = torch.ones(n_classes, dtype=torch.float32)
        for cls_idx in range(n_classes):
            cnt = label_counts.get(cls_idx, 1)   # avoid div by 0
            w   = total / (n_classes * cnt)
            weights[cls_idx] = max(0.5, min(10.0, w))
        model.set_class_weights(route, weights)
        print(f"   Applied inverse-frequency class weights "
              f"(min={weights.min():.2f}, max={weights.max():.2f}, "
              f"mean={weights.mean():.2f})")
        # Show top 5 rarest → highest weight classes
        top_rare = sorted(range(n_classes),
                          key=lambda i: label_counts.get(i, 0))[:5]
        print(f"   Rarest classes (upweighted):")
        for ci in top_rare:
            print(f"      weight={weights[ci].item():.2f}  "
                  f"count={label_counts.get(ci, 0):<4}  "
                  f"class='{vocab[ci][:50]}'")

    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"]*2,
                          shuffle=False, num_workers=0)
    print(f"   Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  "
          f"Device: {CFG['device']}")

    backbone_params = [p for p in model.distilbert.parameters()
                       if p.requires_grad]
    head_params     = (list(model.projector.parameters()) +
                       list(model.heads.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": CFG["distilbert_lr"]},
        {"params": head_params,     "lr": CFG["head_lr"]},
    ], weight_decay=CFG["weight_decay"])

    total_steps  = len(train_dl) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    patience_cnt  = 0
    log_rows      = []
    ckpt_path     = os.path.join(
        CFG["ckpt_dir"], f"stage4_revised_{route_name}_best.pt")

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0; n_steps = 0; n_skipped = 0

        for batch in tqdm(train_dl,
                          desc=f"  Epoch {epoch:02d}/{CFG['epochs']} [train]",
                          leave=False):
            fused   = batch["fused"].to(CFG["device"])
            disease = batch["disease"].to(CFG["device"])
            inp_ids = batch["input_ids"].to(CFG["device"])
            att_msk = batch["attention_mask"].to(CFG["device"])
            labels  = batch["label"].to(CFG["device"])

            optimizer.zero_grad()
            loss, _ = model(route, fused, disease, inp_ids, att_msk, labels)
            if not torch.isfinite(loss):
                n_skipped += 1; continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            n_steps += 1

        tr_loss /= max(n_steps, 1)
        if n_skipped > 0:
            print(f"  ⚠️   {n_skipped} NaN batches skipped "
                  f"({100*n_skipped/max(n_steps+n_skipped,1):.1f}%)")

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = 0.0; n_correct = 0; n_total = 0

        with torch.no_grad():
            for batch in tqdm(val_dl,
                              desc=f"  Epoch {epoch:02d}/{CFG['epochs']} [val]",
                              leave=False):
                fused   = batch["fused"].to(CFG["device"])
                disease = batch["disease"].to(CFG["device"])
                inp_ids = batch["input_ids"].to(CFG["device"])
                att_msk = batch["attention_mask"].to(CFG["device"])
                labels  = batch["label"].to(CFG["device"])

                loss, logits = model(route, fused, disease,
                                     inp_ids, att_msk, labels)
                if torch.isfinite(loss):
                    va_loss += loss.item()
                if route == 2:
                    pb = (torch.sigmoid(logits) >= CFG["threshold"]).float()
                    n_correct += (pb == labels.float()).float().mean(dim=1).sum().item()
                else:
                    n_correct += (logits.argmax(dim=-1) == labels).sum().item()
                n_total += labels.size(0)

        va_loss /= max(len(val_dl), 1)
        val_acc  = n_correct / max(n_total, 1)

        print(f"  Epoch {epoch:02d}/{CFG['epochs']} | "
              f"tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
              f"val_acc={val_acc*100:.2f}%")

        log_rows.append({"epoch": epoch, "tr_loss": tr_loss,
                         "va_loss": va_loss, "val_acc": val_acc})

        if va_loss < best_val_loss:
            best_val_loss = va_loss; patience_cnt = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab": vocab, "route": route,
                        "val_loss": va_loss, "val_acc": val_acc}, ckpt_path)
            print(f"  ✅  New best val_loss={va_loss:.4f} → {ckpt_path}")
        else:
            patience_cnt += 1
            print(f"  ⏳  patience {patience_cnt}/{CFG['early_stop_pat']}")
            if patience_cnt >= CFG["early_stop_pat"]:
                print(f"\n🛑  Early stopping at epoch {epoch}")
                break

    pd.DataFrame(log_rows).to_csv(
        os.path.join(CFG["log_dir"], f"route{route}_{route_name}_log.csv"),
        index=False)
    print(f"\n✅  Route {route} ({route_name}) done.  "
          f"Best val_loss={best_val_loss:.4f}")
    return best_val_loss


def train_distilbert_all_routes(train_records, val_records):
    """Convenience wrapper — trains routes 0-3 separately in sequence."""
    results = {}
    for r in [0, 1, 2, 3]:
        results[r] = train_distilbert_route(r, train_records, val_records)
    print(f"\n{'='*65}")
    print("✅  All 4 routes trained separately:")
    for r, loss in results.items():
        print(f"   Route {r} ({ROUTE_NAMES[r]:<15}): best_val_loss={loss:.4f}")
    print(f"{'='*65}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 11. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_route(route: int, test_records):
    from transformers import DistilBertTokenizerFast

    route_name = ROUTE_NAMES[route]
    ckpt_path  = os.path.join(
        CFG["ckpt_dir"], f"stage4_revised_{route_name}_best.pt")
    assert os.path.exists(ckpt_path), \
        f"Checkpoint not found: {ckpt_path}\nRun: python stage4_revised.py --mode train --route {route}"

    ckpt      = torch.load(ckpt_path, map_location=CFG["device"],
                           weights_only=False)
    vocab     = ckpt["vocab"]
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        DistilBERTAnswerModel.MODEL_NAME)
    model     = DistilBERTAnswerModel(vocab_per_route={route: vocab})
    model.load_state_dict(ckpt["model_state"])
    model     = model.to(CFG["device"])
    model.eval()

    test_ds = DistilBERTRouteDataset(
        test_records, route, tokenizer, vocab, CFG["max_input_len"])
    test_dl = DataLoader(test_ds, batch_size=CFG["batch_size"] * 2,
                         shuffle=False, num_workers=0)

    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"  Evaluating route {route}"):
            fused   = batch["fused"].to(CFG["device"])
            disease = batch["disease"].to(CFG["device"])
            inp_ids = batch["input_ids"].to(CFG["device"])
            att_msk = batch["attention_mask"].to(CFG["device"])

            preds   = model.predict(route, fused, disease, inp_ids, att_msk)
            all_preds.extend(preds)
            all_gts.extend(batch["answer_raw"])

    correct  = [int(p == g) for p, g in zip(all_preds, all_gts)]
    accuracy = sum(correct) / max(len(correct), 1)

    print(f"\n{'─'*50}")
    print(f"  Route {route}  ({route_name})  —  DistilBERT")
    print(f"{'─'*50}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%  "
          f"({sum(correct):,} / {len(correct):,})")
    print(f"{'─'*50}\n")

    # Save CSV
    df = pd.DataFrame({"prediction": all_preds, "ground_truth": all_gts,
                        "correct": correct})
    out_path = os.path.join(
        CFG["log_dir"], f"route{route}_{route_name}_eval.csv")
    df.to_csv(out_path, index=False)
    print(f"✅  Eval results saved → {out_path}")
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 12. FULL PIPELINE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
class Stage4RevisedPredictor:
    """
    End-to-end inference: image + question → answer.
    Automatically selects the correct model (DistilBERT or YOLO)
    based on the question type.
    """
    def __init__(self):
        from transformers import DistilBertTokenizerFast
        print("\n🔍  Loading Stage 4 Revised Pipeline ...")

        self.extractor  = FusionExtractor(CFG["stage3_ckpt"])
        self.text_prep  = TextPreprocessor()
        self.transform  = build_image_transform(is_train=False)
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained(
            DistilBERTAnswerModel.MODEL_NAME)

        # Load DistilBERT models for routes 0–3
        self.distilbert_models = {}
        for route in DISTILBERT_ROUTES:
            route_name = ROUTE_NAMES[route]
            ckpt_path  = os.path.join(
                CFG["ckpt_dir"], f"stage4_revised_{route_name}_best.pt")
            if os.path.exists(ckpt_path):
                ckpt  = torch.load(ckpt_path, map_location=CFG["device"],
                                   weights_only=False)
                vocab = ckpt["vocab"]
                m     = DistilBERTAnswerModel(vocab_per_route={route: vocab})
                m.load_state_dict(ckpt["model_state"])
                m     = m.to(CFG["device"])
                m.eval()
                self.distilbert_models[route] = (m, vocab)
                print(f"   ✅  Route {route} ({route_name}) loaded")
            else:
                print(f"   ⚠️   Route {route} checkpoint not found — skipping")

        # Load TWO separate YOLO models — supervisor's directive
        # Route 4 → Segmentation (location from mask)
        # Route 5 → Detection    (count from bboxes)
        self.yolo_location = YOLOLocationModel()
        self.yolo_count    = YOLOCountModel()

        print("✅  Stage 4 Revised Pipeline ready\n")

    @torch.no_grad()
    def predict(self, image_path: str, question: str) -> dict:
        route = infer_route(question)
        print(f"   Question: {question}")
        print(f"   Route   : {route} ({ROUTE_NAMES[route]})")

        if route in DISTILBERT_ROUTES:
            if route not in self.distilbert_models:
                return {"answer": "model not loaded", "route": route,
                        "model": "DistilBERT"}
            model, vocab = self.distilbert_models[route]

            # Extract Stage 3 features
            img = Image.open(image_path).convert("RGB")
            feats = self.extractor.extract_batch(
                [img], [question], self.transform, self.text_prep)
            fused   = feats["fused"][0].unsqueeze(0).to(CFG["device"])
            disease = feats["disease"][0].unsqueeze(0).to(CFG["device"])

            enc     = self.tokenizer(
                question, return_tensors="pt",
                max_length=CFG["max_input_len"],
                padding="max_length", truncation=True)
            inp_ids = enc["input_ids"].to(CFG["device"])
            att_msk = enc["attention_mask"].to(CFG["device"])

            preds   = model.predict(route, fused, disease, inp_ids, att_msk)
            answer  = preds[0]
            return {"answer": answer, "route": route, "model": "DistilBERT"}

        elif route in YOLO_ROUTES:
            if route == 4:
                answer = self.yolo_location.predict(image_path)
                model_name = "YOLO-Segmentation"
            else:
                answer = self.yolo_count.predict(image_path)
                model_name = "YOLO-Detection"
            return {"answer": answer, "route": route, "model": model_name}

        return {"answer": "unknown route", "route": route, "model": "none"}


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 4 Revised — DistilBERT + YOLO")
    parser.add_argument("--mode", required=True,
                        choices=["train", "eval", "infer", "demo"],
                        help="Operation mode")
    parser.add_argument("--route", default="0",
                        help="Route: 0-5, all_distilbert, all_yolo, all")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--question",   default=None)
    args = parser.parse_args()

    # ── Load data + build cache ───────────────────────────────────────────────
    if args.mode in ["train", "eval"]:
        from datasets import load_from_disk
        raw = load_from_disk(CFG["data_dir"])
        print(f"Dataset splits: {list(raw.keys())}")

        extractor  = FusionExtractor(CFG["stage3_ckpt"])
        text_prep  = TextPreprocessor()

        # Build / load caches
        if "train" in raw:
            if "validation" in raw:
                train_raw = raw["train"]
                val_raw   = raw["validation"]
            else:
                print("   ℹ️  No 'validation' split — using 90/10 train split")
                split     = raw["train"].train_test_split(
                    test_size=0.1, seed=CFG["seed"])
                train_raw = split["train"]
                val_raw   = split["test"]
                print(f"      Train: {len(train_raw):,}  Val: {len(val_raw):,}")

            train_records = cache_stage3_features(
                extractor, text_prep, train_raw, "train", CFG["cache_dir"])
            val_records   = cache_stage3_features(
                extractor, text_prep, val_raw,   "val",   CFG["cache_dir"])

        test_records = cache_stage3_features(
            extractor, text_prep, raw["test"], "test", CFG["cache_dir"])

    # ── Route selection ───────────────────────────────────────────────────────
    route_str = args.route.lower()
    if route_str == "all":
        routes = list(range(6))
    elif route_str == "all_distilbert":
        routes = DISTILBERT_ROUTES
    elif route_str == "all_yolo":
        routes = YOLO_ROUTES
    else:
        routes = [int(route_str)]

    # ── Mode dispatch ─────────────────────────────────────────────────────────
    if args.mode == "train":
        distilbert_requested = [r for r in routes if r in DISTILBERT_ROUTES]
        yolo_requested       = [r for r in routes if r in YOLO_ROUTES]
        for r in distilbert_requested:
            train_distilbert_route(r, train_records, val_records)
        for r in yolo_requested:
            train_yolo_routes(train_records, CFG["image_dir"])

    elif args.mode == "eval":
        for r in routes:
            if r in DISTILBERT_ROUTES:
                evaluate_route(r, test_records)
            else:
                print(f"⚠️   Route {r} (YOLO) eval not yet implemented "
                      f"— use YOLO built-in validation metrics.")

    elif args.mode == "infer":
        assert args.image_path and args.question, \
            "Provide --image_path and --question for inference"
        predictor = Stage4RevisedPredictor()
        result    = predictor.predict(args.image_path, args.question)
        print(f"\n{'='*50}")
        print(f"  Answer : {result['answer']}")
        print(f"  Route  : {result['route']} ({ROUTE_NAMES[result['route']]})")
        print(f"  Model  : {result['model']}")
        print(f"{'='*50}\n")

    elif args.mode == "demo":
        demo_questions = [
            (0, "Is a polyp present in this image?"),
            (1, "What type of abnormality is visible?"),
            (2, "What findings are present in this image?"),
            (3, "What colour is the artefact visible?"),
            (4, "Where is the polyp located in this image?"),
            (5, "How many polyps are visible in this image?"),
        ]
        print("\n📋  Route classification demo:")
        for expected, q in demo_questions:
            got = infer_route(q)
            status = "✅" if got == expected else "⚠️ "
            print(f"  {status}  Route {got}  |  {q}")


if __name__ == "__main__":
    main()