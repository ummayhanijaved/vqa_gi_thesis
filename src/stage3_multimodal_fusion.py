"""
=============================================================================
STAGE 3: Multimodal Fusion Module
Co-attention mechanism combining:
    • Visual features      (2048-D) ← frozen ResNet50 backbone (Stage 1)
    • Disease vector       (23-D)   ← TreeNet MLP head (Stage 1)
    • Question embeddings  (768-D)  ← DistilBERT [CLS] token (Stage 2)
→ Fused representation (512-D) routed to Stage 4 specialised models
=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================

ARCHITECTURE:
    Image (3×224×224) ──→ [FROZEN ResNet50] ──→ V ∈ R^2048
                                                      ↓
    Question (text) ───→ [FROZEN DistilBERT] ──→ Q ∈ R^768
                                                      ↓
    V + Q ─────────────→ [Cross-Attention] ──→ A ∈ R^512
                                                      ↓
    Disease vector d ──→ [Disease Gate] ──────→ G ∈ R^256
                                                      ↓
    A + G ─────────────→ [Fusion MLP] ───────→ F ∈ R^512  ← fused repr.
                                                      ↓
                         [Question Router] ──→ route to Stage 4 model

USAGE:
    python stage3_multimodal_fusion.py --mode train
    python stage3_multimodal_fusion.py --mode eval  --checkpoint ./checkpoints/stage3_best.pt
    python stage3_multimodal_fusion.py --mode demo
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED IMPORTS FROM PREPROCESSING MODULE
# All image transforms and text processing come from one source of truth
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~"))

from preprocessing import (
    build_image_transform,      # image augmentation pipeline
    TextPreprocessor,           # DistilBERT tokenizer wrapper
    clean_text,                 # text cleaning function
    CFG as PREP_CFG,            # preprocessing config
)

from stage1_disease_classifier import (
    TreeNetDiseaseClassifier,   # frozen ResNet50 + MLP head
    DISEASE_LABELS,             # 23 disease names
)

from stage2_question_categorizer import (
    QuestionTypePredictor,      # DistilBERT question classifier
    CLASS_NAMES as QTYPE_NAMES, # 6 question type names
)

print("✅  Imports from preprocessing / stage1 / stage2 successful")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Dimensions
    visual_dim      = 2048,     # ResNet50 avgpool output
    text_dim        = 768,      # DistilBERT hidden size
    disease_dim     = 23,       # TreeNet output
    attn_dim        = 512,      # cross-attention projection
    disease_gate_dim= 256,      # disease gating MLP
    fusion_dim      = 512,      # final fused representation
    num_heads       = 8,        # multi-head attention heads
    num_qtypes      = 6,        # question type classes

    # Training
    epochs          = 15,
    batch_size      = 32,
    learning_rate   = 2e-4,
    weight_decay    = 0.01,
    warmup_steps    = 300,
    dropout         = 0.2,
    early_stop_pat  = 4,
    grad_clip       = 1.0,
    fp16            = torch.cuda.is_available(),

    # Checkpoints
    stage1_ckpt     = "./checkpoints/stage1_best.pt",
    stage2_ckpt     = "./checkpoints/best_model",
    stage3_ckpt_dir = "./checkpoints",

    # Data
    data_dir        = "./data/kvasir_local",
    log_dir         = "./logs",
    seed            = 42,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers     = 0,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LABEL INFERENCE (same as Stage 2 — reused from preprocessing logic)
# ─────────────────────────────────────────────────────────────────────────────
def infer_qtype_label(question: str, answer: str) -> int:
    """
    Question type label — matches Stage 2 training logic exactly.
    Order of checks matters: most specific first.
    """
    q, a = question.lower().strip(), answer.lower().strip()
    
    # Numerical count — very specific keywords
    if any(w in q for w in ("how many", "count", "number of", "total number")):
        return 5
    # Color — color/colour anywhere in question
    if any(w in q for w in ("color", "colour", "what color", "what colour")):
        return 3
    # Location — spatial keywords
    if any(w in q for w in ("where", "location", "located", "position",
                             "which part", "which area", "which region",
                             "whereabouts")):
        return 4
    # Yes/No — answer starts with yes or no
    if a.startswith("yes") or a.startswith("no"):
        return 0
    # Multiple choice — answer contains comma-separated items
    if "," in a and len(a.split(",")) >= 2 and len(a) < 300:
        return 2
    # Single choice — default
    return 1

# ─────────────────────────────────────────────────────────────────────────────
# 3.  UNIFIED DATASET  (uses preprocessing.py transforms — single source)
# ─────────────────────────────────────────────────────────────────────────────
class VQAFusionDataset(Dataset):
    """
    Unified dataset for Stage 3 fusion training.

    Uses build_image_transform() and TextPreprocessor from preprocessing.py
    — no duplicated transform code anywhere.

    Returns per sample:
        image           : (3, 224, 224) float32
        q_input_ids     : (128,) int64
        q_attention_mask: (128,) int64
        qtype_label     : int  (0-5, for routing loss)
        answer_raw      : str  (ground truth for Stage 4)
        question_raw    : str
    """
    def __init__(self, hf_split, split: str,
                 text_preprocessor: TextPreprocessor,
                 url_map: dict = {}):
        self.data       = hf_split
        self.split      = split
        self.img_tfm    = build_image_transform(split)   # ← from preprocessing.py
        self.text_prep  = text_preprocessor              # ← from preprocessing.py
        self.url_map    = url_map                        # URL → local path
        print(f"   VQAFusionDataset [{split}]: {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # ── Image (local file → PIL) ──────────────────────────────────────
        raw_img = ex["image"]
        if isinstance(raw_img, str) and raw_img in self.url_map:
            from PIL import Image as PILImage
            img = PILImage.open(self.url_map[raw_img]).convert("RGB")
        elif isinstance(raw_img, str) and raw_img.startswith("http"):
            # Fallback: download on-the-fly (slow but won't crash)
            import requests, io
            from PIL import Image as PILImage
            r = requests.get(raw_img, timeout=30)
            img = PILImage.open(io.BytesIO(r.content)).convert("RGB")
        else:
            img = raw_img.convert("RGB")
        img_tensor = self.img_tfm(img)          # (3, 224, 224)

        # ── Question (preprocessing.py tokenizer) ──────────────────────────
        q_enc = self.text_prep.preprocess(ex["question"])

        # ── Labels ─────────────────────────────────────────────────────────
        qtype = infer_qtype_label(ex["question"], ex["answer"])

        return {
            "image"           : img_tensor,
            "q_input_ids"     : q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "qtype_label"     : torch.tensor(qtype, dtype=torch.long),
            "question_raw"    : ex["question"],
            "answer_raw"      : ex["answer"],
        }


def build_fusion_dataloaders(text_prep: TextPreprocessor):
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    import json
    print("📦  Loading dataset from local disk …")
    raw = load_from_disk(CFG["data_dir"])
    # Load local image path map (built by download_images.py)
    url_map_path = "./data/url_to_path.json"
    url_map = {}
    if os.path.exists(url_map_path):
        with open(url_map_path) as f:
            url_map = json.load(f)
        print(f"   Local image map: {len(url_map):,} entries")
    else:
        print("   ⚠️  No local image map found. Run download_images.py first!")

    indices   = list(range(len(raw["train"])))
    tr_idx, va_idx = train_test_split(indices, test_size=0.20,
                                      random_state=CFG["seed"])

    train_ds = VQAFusionDataset(raw["train"].select(tr_idx), "train", text_prep, url_map)
    val_ds   = VQAFusionDataset(raw["train"].select(va_idx), "val",   text_prep, url_map)
    test_ds  = VQAFusionDataset(raw["test"],                  "test",  text_prep, url_map)

    kw = dict(num_workers=CFG["num_workers"], pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, **kw),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FROZEN ENCODERS  (Stage 1 + Stage 2 — no gradient updates)
# ─────────────────────────────────────────────────────────────────────────────
class FrozenVisualEncoder(nn.Module):
    """
    Wraps Stage 1 TreeNet backbone.
    Extracts:
        visual_feat : (B, 2048)  — ResNet50 avgpool
        disease_vec : (B, 23)    — sigmoid probability vector d
    Both are produced with torch.no_grad() — frozen throughout.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.treenet = TreeNetDiseaseClassifier()
        ckpt = torch.load(checkpoint_path,
                          map_location=CFG["device"])
        self.treenet.load_state_dict(ckpt["model_state"])
        # Freeze everything
        for p in self.treenet.parameters():
            p.requires_grad = False
        self.treenet.eval()
        print(f"🔒  FrozenVisualEncoder loaded  "
              f"(stage1 best_f1={ckpt['best_f1']:.4f})")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.treenet(x)
        return out["features"], out["probs"]   # (B,2048), (B,23)


class FrozenTextEncoder(nn.Module):
    """
    Wraps Stage 2 DistilBERT.
    Extracts [CLS] token hidden state as question embedding (B, 768).
    Produced with torch.no_grad() — frozen throughout.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(checkpoint_path)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()
        print(f"🔒  FrozenTextEncoder loaded  (stage2 DistilBERT)")

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
        # [CLS] token = first token of last hidden state
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return cls

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CO-ATTENTION FUSION MODULE
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between visual features V and question embedding Q.

    Step 1: Project V and Q to common dimension d_attn=512
    Step 2: Multi-head cross-attention  A = Attention(Q, V, V)
            — question attends over visual features
    Step 3: Residual + LayerNorm
    Output : attended representation (B, 512)
    """
    def __init__(self, visual_dim=2048, text_dim=768,
                 attn_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        # Project to common dimension
        self.v_proj = nn.Sequential(
            nn.Linear(visual_dim, attn_dim),
            nn.ReLU(), nn.LayerNorm(attn_dim)
        )
        self.q_proj = nn.Sequential(
            nn.Linear(text_dim, attn_dim),
            nn.ReLU(), nn.LayerNorm(attn_dim)
        )
        # Multi-head cross-attention: Q attends over V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = attn_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm    = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feat: torch.Tensor,
                question_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat   : (B, 2048)
            question_feat : (B, 768)
        Returns:
            attended : (B, 512)
        """
        # Project to (B, 1, attn_dim) for attention API
        V = self.v_proj(visual_feat).unsqueeze(1)      # (B, 1, 512)
        Q = self.q_proj(question_feat).unsqueeze(1)    # (B, 1, 512)

        # Cross-attention: query=Q, key=V, value=V
        attn_out, attn_weights = self.cross_attn(
            query = Q, key = V, value = V
        )                                              # (B, 1, 512)

        # Residual + norm
        out = self.norm(Q + self.dropout(attn_out))    # (B, 1, 512)
        return out.squeeze(1), attn_weights             # (B, 512)


class DiseaseGate(nn.Module):
    """
    Disease-aware gating mechanism.
    Projects 23-D disease vector d through MLP to produce
    a gate vector G ∈ R^256 that modulates the fused representation.

    G = sigmoid(W₂ · ReLU(W₁ · d))

    This ensures disease context from Stage 1 explicitly influences
    the fusion output — the core novelty of the thesis architecture.
    """
    def __init__(self, disease_dim=23, gate_dim=256, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(disease_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, gate_dim),
            nn.Sigmoid(),           # gate values ∈ [0, 1]
        )

    def forward(self, disease_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:  disease_vec : (B, 23) probabilities from Stage 1
        Returns: gate      : (B, 256)
        """
        return self.gate(disease_vec)


class FusionMLP(nn.Module):
    """
    Final fusion MLP combining cross-attended features + disease gate.

    Input  : [attended (512) || disease_gate (256)] → 768-D concat
    Output : fused_repr (512-D) → fed to Stage 4 routing

    Also produces question type logits (6-class) for routing supervision.
    """
    def __init__(self, attn_dim=512, gate_dim=256,
                 fusion_dim=512, num_qtypes=6, dropout=0.2):
        super().__init__()
        in_dim = attn_dim + gate_dim   # 512 + 256 = 768

        self.fusion = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        # Routing head — predicts which Stage 4 model to use
        self.router = nn.Linear(fusion_dim, num_qtypes)

    def forward(self, attended: torch.Tensor,
                disease_gate: torch.Tensor) -> tuple:
        """
        Args:
            attended     : (B, 512) — from CrossAttentionFusion
            disease_gate : (B, 256) — from DiseaseGate
        Returns:
            fused_repr   : (B, 512) — fused multimodal representation
            routing_logits: (B, 6)  — question type routing scores
        """
        combined    = torch.cat([attended, disease_gate], dim=-1)  # (B, 768)
        fused_repr  = self.fusion(combined)                         # (B, 512)
        routing_logits = self.router(fused_repr)                    # (B, 6)
        return fused_repr, routing_logits


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL STAGE 3 MODEL
# ─────────────────────────────────────────────────────────────────────────────
class Stage3MultimodalFusion(nn.Module):
    """
    Complete Stage 3 pipeline:

        Image ──→ FrozenVisualEncoder ──→ visual_feat (2048)
                                       └→ disease_vec (23)
        Question → FrozenTextEncoder  ──→ question_feat (768)

        visual_feat + question_feat → CrossAttentionFusion → attended (512)
        disease_vec                 → DiseaseGate          → gate (256)
        attended + gate             → FusionMLP            → fused_repr (512)
                                                            → routing_logits (6)

    Only FusionMLP + DiseaseGate + CrossAttentionFusion are trainable.
    Both encoders remain completely frozen.
    """
    def __init__(self):
        super().__init__()

        # Frozen encoders (Stage 1 + Stage 2)
        self.visual_enc = FrozenVisualEncoder(CFG["stage1_ckpt"])
        self.text_enc   = FrozenTextEncoder(CFG["stage2_ckpt"])

        # Trainable fusion modules
        self.cross_attn = CrossAttentionFusion(
            visual_dim = CFG["visual_dim"],
            text_dim   = CFG["text_dim"],
            attn_dim   = CFG["attn_dim"],
            num_heads  = CFG["num_heads"],
            dropout    = CFG["dropout"],
        )
        self.disease_gate = DiseaseGate(
            disease_dim = CFG["disease_dim"],
            gate_dim    = CFG["disease_gate_dim"],
            dropout     = CFG["dropout"],
        )
        self.fusion_mlp = FusionMLP(
            attn_dim   = CFG["attn_dim"],
            gate_dim   = CFG["disease_gate_dim"],
            fusion_dim = CFG["fusion_dim"],
            num_qtypes = CFG["num_qtypes"],
            dropout    = CFG["dropout"],
        )

        # Parameter summary
        n_total     = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters()
                         if p.requires_grad)
        n_frozen    = n_total - n_trainable
        print(f"\n🧠  Stage3MultimodalFusion")
        print(f"    Total params     : {n_total:,}")
        print(f"    Trainable        : {n_trainable:,}  "
              f"(CrossAttn + DiseaseGate + FusionMLP)")
        print(f"    Frozen           : {n_frozen:,}  "
              f"(ResNet50 + DistilBERT)")

    def forward(self, image: torch.Tensor,
                q_input_ids: torch.Tensor,
                q_attention_mask: torch.Tensor) -> dict:
        """
        Args:
            image            : (B, 3, 224, 224)
            q_input_ids      : (B, 128)
            q_attention_mask : (B, 128)
        Returns dict with:
            fused_repr       : (B, 512)  ← input to Stage 4
            routing_logits   : (B, 6)    ← question type routing
            disease_vec      : (B, 23)   ← passthrough from Stage 1
            attn_weights     : (B, 1, 1) ← for attention visualisation
        """
        # Stage 1 — visual + disease
        visual_feat, disease_vec = self.visual_enc(image)

        # Stage 2 — question embedding
        question_feat = self.text_enc(q_input_ids, q_attention_mask)

        # Cross-attention fusion
        attended, attn_weights = self.cross_attn(visual_feat, question_feat)

        # Disease gating
        disease_gate = self.disease_gate(disease_vec)

        # Final fusion + routing
        fused_repr, routing_logits = self.fusion_mlp(attended, disease_gate)

        return {
            "fused_repr"    : fused_repr,
            "routing_logits": routing_logits,
            "disease_vec"   : disease_vec,
            "attn_weights"  : attn_weights,
            "visual_feat"   : visual_feat,
            "question_feat" : question_feat,
        }

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler,
                    scaler, loss_fn, epoch):
    model.train()
    total_loss, correct, total = 0., 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)

    for batch in pbar:
        imgs   = batch["image"].to(CFG["device"])
        ids    = batch["q_input_ids"].to(CFG["device"])
        mask   = batch["q_attention_mask"].to(CFG["device"])
        labels = batch["qtype_label"].to(CFG["device"])

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
            out  = model(imgs, ids, mask)
            loss = loss_fn(out["routing_logits"].float(), labels)
        
        # Skip NaN batches
        if torch.isnan(loss):
            print(f"   ⚠️  NaN loss at step, skipping batch")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            CFG["grad_clip"]
        )
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = out["routing_logits"].argmax(-1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, desc="val"):
    from sklearn.metrics import f1_score, classification_report
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc=f"  [{desc}]", leave=False):
        imgs   = batch["image"].to(CFG["device"])
        ids    = batch["q_input_ids"].to(CFG["device"])
        mask   = batch["q_attention_mask"].to(CFG["device"])
        labels = batch["qtype_label"].to(CFG["device"])

        with torch.cuda.amp.autocast(enabled=False):  # FP32 for stable eval
            out  = model(imgs.float(), ids, mask)
            loss = loss_fn(out["routing_logits"].float(), labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = out["routing_logits"].argmax(-1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds .extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        zero_division=0)
    return dict(
        loss     = total_loss / total,
        acc      = correct / total,
        macro_f1 = macro_f1,
        preds    = all_preds,
        labels   = all_labels,
    )


def train():
    from transformers import get_cosine_schedule_with_warmup
    os.makedirs(CFG["stage3_ckpt_dir"], exist_ok=True)
    os.makedirs(CFG["log_dir"],         exist_ok=True)

    # Single shared TextPreprocessor — from preprocessing.py
    text_prep = TextPreprocessor()

    train_loader, val_loader, test_loader = build_fusion_dataloaders(text_prep)

    model   = Stage3MultimodalFusion().to(CFG["device"])
    loss_fn = nn.CrossEntropyLoss()

    # Only train fusion modules — encoders are frozen
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )
    total_steps = len(train_loader) * CFG["epochs"]
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = CFG["warmup_steps"],
        num_training_steps = total_steps,
    )
    scaler   = torch.cuda.amp.GradScaler(enabled=CFG["fp16"])
    best_acc = 0.; patience = 0
    best_ckpt = os.path.join(CFG["stage3_ckpt_dir"], "stage3_best.pt")
    history  = []

    print(f"\n🚀  Training Stage 3 on {CFG['device'].upper()}"
          f"  |  FP16={CFG['fp16']}\n" + "="*70)

    for epoch in range(CFG["epochs"]):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, loss_fn, epoch)
        va = evaluate(model, val_loader, loss_fn, "val")

        row = dict(epoch=epoch+1,
                   tr_loss=round(tr_loss,4), tr_acc=round(tr_acc,4),
                   va_loss=round(va["loss"],4), va_acc=round(va["acc"],4),
                   va_f1=round(va["macro_f1"],4))
        history.append(row)

        print(f"Ep {epoch+1:02d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}"
              f" | va_loss={va['loss']:.4f}  va_acc={va['acc']:.4f}"
              f"  va_f1={va['macro_f1']:.4f}")

        if va["acc"] > best_acc:
            best_acc = va["acc"]; patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch"      : epoch+1,
                "best_acc"   : best_acc,
            }, best_ckpt)
            print(f"   ✅  New best val_acc={best_acc:.4f} → {best_ckpt}")
        else:
            patience += 1
            print(f"   ⏳  patience {patience}/{CFG['early_stop_pat']}")
            if patience >= CFG["early_stop_pat"]:
                print(f"\n🛑  Early stopping at epoch {epoch+1}"); break

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n" + "="*70 + "\n📊  Test evaluation …")
    ckpt = torch.load(best_ckpt, map_location=CFG["device"])
    model.load_state_dict(ckpt["model_state"])
    te = evaluate(model, test_loader, loss_fn, "test")

    from sklearn.metrics import classification_report
    print(f"   Test Accuracy  : {te['acc']:.4f}")
    print(f"   Test Macro-F1  : {te['macro_f1']:.4f}")
    print()
    print(classification_report(te["labels"], te["preds"],
                                 target_names=QTYPE_NAMES, zero_division=0))

    pd.DataFrame(history).to_csv(
        f"{CFG['log_dir']}/stage3_epoch_log.csv", index=False)
    save_plots(history)
    print(f"\n✅  Stage 3 done. Best val_acc = {best_acc:.4f}")


def save_plots(history):
    df  = pd.DataFrame(history)
    eps = df["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Stage 3: Multimodal Fusion — Training Dynamics",
                 fontsize=13, fontweight="bold")
    axes[0].plot(eps, df["tr_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(eps, df["va_loss"], "r-o", ms=4, label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(eps, df["tr_acc"],  "b-o", ms=4, label="Train")
    axes[1].plot(eps, df["va_acc"],  "r-o", ms=4, label="Val")
    axes[1].set_title("Routing Accuracy"); axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(eps, df["va_f1"],   "g-o", ms=4)
    axes[2].set_title("Val Macro-F1"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    path = f"{CFG['log_dir']}/stage3_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📈  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  STAGE 4 INTERFACE  (FusionExtractor)
# ─────────────────────────────────────────────────────────────────────────────
class FusionExtractor:
    """
    Stage 4 interface for Stage 3.
    Loads trained fusion model and extracts:
        fused_repr     : (B, 512)  → input to Stage 4 answer models
        routing_label  : (B,)      → which Stage 4 model to use
        disease_vec    : (B, 23)   → disease context passthrough

    Usage in Stage 4:
        extractor = FusionExtractor("./checkpoints/stage3_best.pt")
        out = extractor.extract(image_tensor, input_ids, attention_mask)
        fused  = out["fused_repr"]      # (B, 512)
        route  = out["routing_label"]   # (B,)  0-5
        d_vec  = out["disease_vec"]     # (B, 23)
    """
    def __init__(self, checkpoint_path: str):
        self.device = CFG["device"]
        self.model  = Stage3MultimodalFusion().to(self.device)
        ckpt        = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"✅  FusionExtractor loaded  "
              f"(best_acc={ckpt['best_acc']:.4f})")

    @torch.no_grad()
    def extract(self, image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        image        = image.to(self.device)
        input_ids    = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        out = self.model(image, input_ids, attention_mask)
        return {
            "fused_repr"   : out["fused_repr"],
            "routing_label": out["routing_logits"].argmax(-1),
            "routing_probs": F.softmax(out["routing_logits"], dim=-1),
            "disease_vec"  : out["disease_vec"],
            "attn_weights" : out["attn_weights"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","eval","demo"],
                        default="demo")
    parser.add_argument("--checkpoint",
                        default="./checkpoints/stage3_best.pt")
    args = parser.parse_args()

    if args.mode == "demo":
        print("\n🧠  Stage 3 model summary (no data needed) …\n")
        model = Stage3MultimodalFusion().to(CFG["device"])
        dummy_img  = torch.randn(2, 3, 224, 224).to(CFG["device"])
        dummy_ids  = torch.randint(0, 30522, (2, 128)).to(CFG["device"])
        dummy_mask = torch.ones(2, 128, dtype=torch.long).to(CFG["device"])
        out = model(dummy_img, dummy_ids, dummy_mask)
        print(f"\n   Input image shape    : {dummy_img.shape}")
        print(f"   Input ids shape      : {dummy_ids.shape}")
        print(f"   → fused_repr shape   : {out['fused_repr'].shape}"
              f"  ← Stage 4 input")
        print(f"   → routing_logits     : {out['routing_logits'].shape}"
              f"  ← 6 question types")
        print(f"   → disease_vec shape  : {out['disease_vec'].shape}"
              f"  ← 23-D vector")
        print(f"   → attn_weights shape : {out['attn_weights'].shape}")
        print(f"\n   Routing probs (random init): "
              f"{F.softmax(out['routing_logits'][0], dim=-1).tolist()}")
        print(f"\n   Question type routing:")
        for i, name in enumerate(QTYPE_NAMES):
            print(f"   [{i}] {name}")
        print(f"\n✅  Stage 3 architecture verified.")
        print(f"    Next: python stage3_multimodal_fusion.py --mode train")

    elif args.mode == "train":
        train()

    elif args.mode == "eval":
        extractor = FusionExtractor(args.checkpoint)
        print("✅  FusionExtractor ready for Stage 4 integration.")
