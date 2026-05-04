"""
=============================================================================
STAGE 4: Specialised Answer Generation — Transformer Models
Routes each sample to one of 6 specialised transformer-based heads based on
Stage 2/3 routing decision, then generates the final answer.

Architecture:
    Stage 3 output:
        fused_repr   (512-D)  → projected into each transformer's hidden space
        routing_label (0-5)   → selects which transformer head to use
        disease_vec   (23-D)  → appended to form 535-D conditioning vector

    6 Specialised Transformer Heads:
        [0] Yes/No          → FLAN-T5       (google/flan-t5-base)
        [1] Single-Choice   → BART-Large    (facebook/bart-large)
        [2] Multi-Choice    → DeBERTa-v3    (microsoft/deberta-v3-base)
        [3] Color-Related   → ViT-GPT2      (nlpconnect/vit-gpt2-image-captioning)
        [4] Location-Related→ BLIP2-FLAN    (Salesforce/blip2-flan-t5-xl)
        [5] Numerical Count → T5-Large      (google/t5-large)

    Conditioning mechanism (shared across all heads):
        Stage3Projector: Linear(535 → model_hidden_dim) → LayerNorm
        Projected vector is prepended as a soft-prompt token to the
        encoder input, giving each transformer access to the full
        fused multimodal + disease context from Stages 1-3.

=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================

USAGE:
    python stage4_transformers.py --mode demo
    python stage4_transformers.py --mode train   --route all
    python stage4_transformers.py --mode train   --route 0
    python stage4_transformers.py --mode eval    --route all
    python stage4_transformers.py --mode infer   --image_path x.jpg --question "..."
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

# ── Transformers — imported lazily inside each head class to avoid
#    version conflicts. Only the scheduler is imported at module level.
# ─────────────────────────────────────────────────────────────────────────────
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.expanduser("~"))
from preprocessing import build_image_transform, TextPreprocessor, clean_text
from stage1_disease_classifier import TreeNetDiseaseClassifier
from stage3_multimodal_fusion import (
    Stage3MultimodalFusion, FusionExtractor,
    infer_qtype_label, CFG as S3_CFG,
)

print("✅  All imports successful")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
HOME    = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "vqa_gi_thesis")

CFG = dict(
    # ── Dimensions ────────────────────────────────────────────────────────────
    fused_dim      = 512,
    disease_dim    = 23,
    head_input_dim = 535,          # 512 + 23 — input to all projectors

    # ── Model names ───────────────────────────────────────────────────────────
    # Each route uses a different pretrained transformer
    model_names = {
        0: "google/flan-t5-base",                           # Yes/No
        1: "facebook/bart-large",                           # Single-Choice
        2: "microsoft/deberta-v3-base",                     # Multi-Choice
        3: "nlpconnect/vit-gpt2-image-captioning",          # Color
        4: "Salesforce/blip2-flan-t5-xl",                   # Location
        5: "google/t5-large",                               # Count
    },

    # ── Answer vocabularies (for classification heads) ─────────────────────
    yn_classes       = ["no", "yes"],
    color_classes    = [
        "pink", "red", "orange", "yellow", "green", "blue",
        "purple", "white", "black", "brown", "transparent", "mixed",
    ],
    location_classes = [
        "esophagus", "stomach", "duodenum", "small-bowel", "colon",
        "rectum", "cecum", "pylorus", "z-line", "retroflex-rectum",
        "retroflex-stomach", "ileocecal-valve", "upper-gi", "lower-gi",
        "unknown",
    ],
    count_classes    = ["0", "1", "2", "3", "4", "5", "6-10", "more than 10"],

    # ── DeBERTa multi-label vocab ──────────────────────────────────────────
    vocab_file        = os.path.join(HOME, "data", "stage4_vocab.json"),
    max_vocab_multi   = 150,

    # ── Generation parameters ──────────────────────────────────────────────
    max_new_tokens    = 32,
    num_beams         = 4,
    max_input_length  = 128,

    # ── Training ───────────────────────────────────────────────────────────
    epochs            = 15,
    batch_size        = 8,              # small — these are large models
    learning_rate     = 1e-5,           # lower LR for more stable convergence
    projector_lr      = 5e-5,           # projector learns faster than transformer
    weight_decay      = 0.01,
    warmup_ratio      = 0.1,
    early_stop_pat    = 5,              # more patience — loss can plateau before improving
    grad_clip         = 1.0,
    fp16              = False,   # T5/BART are NaN-prone in fp16; use fp32 for stability

    # ── Paths ──────────────────────────────────────────────────────────────
    stage3_ckpt       = os.path.join(PROJECT, "checkpoints", "stage3_best.pt"),
    ckpt_dir          = os.path.join(PROJECT, "checkpoints", "stage4_transformers"),
    log_dir           = os.path.join(PROJECT, "logs",        "stage4_transformers"),
    data_dir          = os.path.join(HOME,    "data",        "kvasir_local"),
    device            = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers       = 0,
    seed              = 42,
)

os.makedirs(CFG["ckpt_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"],  exist_ok=True)

# Route → head key (consistent throughout)
ROUTE_TO_KEY = {0: "yn", 1: "single", 2: "multi",
                3: "color", 4: "loc",  5: "count"}

ROUTE_NAMES = {
    0: "yes_no", 1: "single_choice", 2: "multi_choice",
    3: "color",  4: "location",      5: "count",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. STAGE-3 CONDITIONING PROJECTOR (shared design, one per head)
# ─────────────────────────────────────────────────────────────────────────────
class Stage3Projector(nn.Module):
    """
    Projects the 535-D Stage 3 conditioning vector (fused_repr + disease_vec)
    into the hidden dimension of a downstream transformer model.

    The projected vector is used as a prefix soft-prompt token prepended to
    the transformer's encoder input embeddings, giving it full access to the
    multimodal disease-aware context produced by Stages 1–3.

    Args:
        hidden_dim : target transformer's hidden dimension
    """
    def __init__(self, hidden_dim: int):
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
            prefix  : (B, 1, hidden_dim)  — one soft-prompt token
        """
        x = torch.cat([fused, disease], dim=-1)   # (B, 535)
        return self.proj(x).unsqueeze(1)           # (B, 1, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HEAD IMPLEMENTATIONS
# Each head wraps a pretrained transformer + Stage3Projector.
# All follow the same interface: forward() for training, generate() for inference.
# ─────────────────────────────────────────────────────────────────────────────

class FlanT5YesNoHead(nn.Module):
    """
    Route 0 — Yes/No questions.
    Model   : FLAN-T5-base (google/flan-t5-base)
    Strategy: Constrained binary decoding.
              The decoder's first-token logits are masked to ONLY "yes"/"no"
              tokens, converting seq2seq into effective binary classification.
              This eliminates free-generation errors and targets 90%+ accuracy.

    Key upgrades vs original:
      - Constrained decoding (yes/no token IDs only) — biggest accuracy gain
      - 4 unfrozen layers (was 2) — more capacity
      - Label smoothing 0.1 — better generalisation
      - forward() uses direct CE loss over yes/no logits — cleaner signal
    """
    HIDDEN = 768
    MODEL  = "google/flan-t5-base"

    def __init__(self):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer  = T5Tokenizer.from_pretrained(self.MODEL)
        self.model      = T5ForConditionalGeneration.from_pretrained(self.MODEL)
        self.projector  = Stage3Projector(self.HIDDEN)

        # Get token IDs for "yes" and "no" — used for constrained decoding
        self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id  = self.tokenizer.encode("no",  add_special_tokens=False)[0]

        # Unfreeze last 4 encoder + decoder blocks (was 2) — more capacity
        self._freeze_except_last_n(n=4)

        # Label smoothing loss for better generalisation
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_xfmr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [FLAN-T5 YN]   projector={n_proj:,}  trainable_xfmr={n_xfmr:,}")
        print(f"  [FLAN-T5 YN]   yes_id={self.yes_id}  no_id={self.no_id}  "
              f"(constrained decoding enabled)")

    def _freeze_except_last_n(self, n: int):
        """Freeze all encoder/decoder blocks except last n layers each."""
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in self.model.encoder.block[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for layer in self.model.decoder.block[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.lm_head.parameters():
            p.requires_grad = True

    def _prepend_prefix(self, input_ids, attention_mask, prefix_embed):
        token_embeds  = self.model.encoder.embed_tokens(input_ids)  # (B, L, H)
        prefix_embed  = prefix_embed.to(token_embeds.dtype)         # match dtype — prevents NaN from fp16/fp32 mismatch
        inputs_embeds = torch.cat([prefix_embed, token_embeds], dim=1)
        prefix_mask   = torch.ones(
            prefix_embed.size(0), 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_mask

    def _get_yn_logits(self, fused, disease, input_ids, attention_mask):
        """
        Run one decoder step and return logits for yes/no tokens only.
        Returns (B, 2) tensor — [no_logit, yes_logit]
        """
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)

        # Decoder start token (T5 uses pad_token_id as decoder_start_token_id)
        B = fused.size(0)
        decoder_input_ids = torch.full(
            (B, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long, device=fused.device,
        )

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            decoder_input_ids=decoder_input_ids,
        )
        # out.logits: (B, 1, vocab_size) — take first (only) decoder step
        first_token_logits = out.logits[:, 0, :]   # (B, vocab_size)

        # Mask to yes/no only
        yn_logits = first_token_logits[
            :, [self.no_id, self.yes_id]]            # (B, 2): [no, yes]
        return yn_logits

    def forward(self, fused, disease, input_ids, attention_mask, labels):
        """
        Training forward pass using constrained yes/no logits.
        labels: (B, seq_len) token IDs — first token is yes_id or no_id.
        We convert to binary (0=no, 1=yes) and use CrossEntropyLoss.
        """
        yn_logits = self._get_yn_logits(fused, disease, input_ids, attention_mask)

        # Convert token-ID labels → binary labels (0=no, 1=yes)
        # labels shape: (B, L) — first valid token (non -100) is yes or no
        first_tokens = labels[:, 0]                    # (B,)
        # Replace -100 padding with no_id as fallback
        first_tokens = torch.where(
            first_tokens == -100,
            torch.tensor(self.no_id, device=labels.device),
            first_tokens,
        )
        binary_labels = (first_tokens == self.yes_id).long()   # 0=no, 1=yes

        loss = self.ce_loss(yn_logits, binary_labels)

        # Return object with .loss attribute to match HF API
        class _Out:
            pass
        o = _Out(); o.loss = loss; o.logits = yn_logits
        return o

    @torch.no_grad()
    def generate(self, fused, disease, input_ids, attention_mask):
        """
        Constrained inference: returns 'yes' or 'no' based on which
        token gets the higher logit. Never generates other tokens.
        """
        yn_logits = self._get_yn_logits(fused, disease, input_ids, attention_mask)
        preds     = yn_logits.argmax(dim=-1)   # 0=no, 1=yes
        return ["yes" if p.item() == 1 else "no" for p in preds]



class BartLargeSingleChoiceHead(nn.Module):
    """
    Route 1 — Single-Choice questions.
    Model   : BART-Large (facebook/bart-large)
    Strategy: Seq2Seq. Stage 3 vector projected into BART's hidden dim (1024)
              and prepended to encoder embeddings. Decoder generates the answer
              as free text (e.g. "colonic polyp", "esophagitis").
    """
    HIDDEN = 1024
    MODEL  = "facebook/bart-large"

    def __init__(self):
        super().__init__()
        from transformers import BartForConditionalGeneration, BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained(self.MODEL)
        self.model     = BartForConditionalGeneration.from_pretrained(self.MODEL)
        self.projector = Stage3Projector(self.HIDDEN)

        self._freeze_except_last_n(n=4)   # was 2 — more capacity

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_xfmr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [BART-L Single] projector={n_proj:,}  trainable_xfmr={n_xfmr:,}")

    def _freeze_except_last_n(self, n: int):
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in self.model.model.encoder.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for layer in self.model.model.decoder.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.lm_head.parameters():
            p.requires_grad = True

    def _prepend_prefix(self, input_ids, attention_mask, prefix_embed):
        token_embeds  = self.model.model.shared(input_ids)
        prefix_embed  = prefix_embed.to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embed, token_embeds], dim=1)
        prefix_mask   = torch.ones(
            prefix_embed.size(0), 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_mask

    def forward(self, fused, disease, input_ids, attention_mask, labels):
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, fused, disease, input_ids, attention_mask):
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)
        out_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            max_new_tokens=CFG["max_new_tokens"],
            num_beams=CFG["num_beams"],
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)


class DeBERTaMultiChoiceHead(nn.Module):
    """
    Route 2 — Multi-Choice questions.
    Model   : DeBERTa-v3-base (microsoft/deberta-v3-base)
    Strategy: Encoder-only multi-label classifier. The Stage 3 conditioning
              token is prepended to the input sequence. [CLS] representation
              is passed to a linear classifier with BCE loss.
              Outputs a multi-hot vector over the answer vocabulary.
    """
    HIDDEN = 768
    MODEL  = "microsoft/deberta-v3-base"

    def __init__(self, n_classes: int):
        super().__init__()
        from transformers import (
            DebertaV2ForSequenceClassification, DebertaV2Tokenizer)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.MODEL)
        self.model     = DebertaV2ForSequenceClassification.from_pretrained(
            self.MODEL,
            num_labels=n_classes,
            problem_type="multi_label_classification",
        )
        self.projector = Stage3Projector(self.HIDDEN)
        self.n_classes = n_classes

        # Freeze embeddings + first 8 layers, train last 4
        self._freeze_except_last_n(n=4)

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_xfmr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [DeBERTa Multi] projector={n_proj:,}  trainable_xfmr={n_xfmr:,}")

    def _freeze_except_last_n(self, n: int):
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in self.model.deberta.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        for p in self.model.pooler.parameters():
            p.requires_grad = True

    def _prepend_prefix(self, input_ids, attention_mask, prefix_embed):
        token_embeds  = self.model.deberta.get_input_embeddings()(input_ids)
        prefix_embed  = prefix_embed.to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embed, token_embeds], dim=1)
        prefix_mask   = torch.ones(
            prefix_embed.size(0), 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_mask

    def forward(self, fused, disease, input_ids, attention_mask, labels):
        """
        labels : (B, n_classes) float multi-hot tensor
        """
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )

    @torch.no_grad()
    def predict(self, fused, disease, input_ids, attention_mask,
                vocab_multi: list, threshold: float = 0.5):
        """Returns list of predicted answer string lists."""
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)
        out    = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
        )
        probs  = torch.sigmoid(out.logits)
        answers = []
        for row in probs:
            active = [vocab_multi[i]
                      for i, p in enumerate(row) if p > threshold]
            answers.append(", ".join(active) if active else "<none>")
        return answers


class ViTGPT2ColorHead(nn.Module):
    """
    Route 3 — Color-Related questions.
    Model   : ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)
    Strategy: The model's vision encoder is replaced by our Stage 3
              conditioning vector projected to GPT2's hidden dim (768),
              reshaped as a sequence of prefix tokens. The GPT2 decoder
              then generates the color answer autoregressively.

    Note: We do NOT re-run ViT on the raw image (image already encoded
          by ResNet50 in Stage 1 + fused in Stage 3). Instead we use
          the Stage 3 vector as a surrogate visual representation.
    """
    HIDDEN = 768
    MODEL  = "nlpconnect/vit-gpt2-image-captioning"
    N_PREFIX = 4    # number of prefix tokens to simulate ViT patch tokens

    def __init__(self):
        super().__init__()
        # ViTFeatureExtractor was removed in transformers>=4.37
        # Use ViTImageProcessor instead (backward-compatible alias)
        from transformers import (
            VisionEncoderDecoderModel, GPT2Tokenizer,
        )
        try:
            from transformers import ViTImageProcessor as ViTProcessor
        except ImportError:
            from transformers import ViTFeatureExtractor as ViTProcessor

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model     = VisionEncoderDecoderModel.from_pretrained(self.MODEL)

        # Replace the linear visual projection with our Stage3Projector
        # then expand to N_PREFIX tokens
        self.projector = Stage3Projector(self.HIDDEN)
        self.expand    = nn.Linear(self.HIDDEN, self.HIDDEN * self.N_PREFIX)

        # Freeze ViT encoder (we bypass it), fine-tune last 2 GPT2 layers
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.decoder.parameters():
            p.requires_grad = False
        for layer in self.model.decoder.transformer.h[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.decoder.lm_head.parameters():
            p.requires_grad = True

        n_proj = sum(p.numel() for p in self.projector.parameters()) + \
                 sum(p.numel() for p in self.expand.parameters())
        n_xfmr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [ViT-GPT2 Color] projector={n_proj:,}  trainable_xfmr={n_xfmr:,}")

    def _get_encoder_hidden(self, fused, disease):
        """
        Produce fake encoder hidden states from Stage 3 vector.
        Shape: (B, N_PREFIX, hidden_dim)
        """
        prefix = self.projector(fused, disease).squeeze(1)  # (B, H)
        expanded = self.expand(prefix)                       # (B, H*N_PREFIX)
        B = fused.size(0)
        return expanded.view(B, self.N_PREFIX, self.HIDDEN)  # (B, N_PREFIX, H)

    def forward(self, fused, disease, labels):
        """
        labels : (B, L) token ids of the target color answer
        """
        encoder_hidden = self._get_encoder_hidden(fused, disease)
        # Feed encoder_hidden directly as encoder_outputs
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        return self.model(
            encoder_outputs=encoder_outputs,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, fused, disease):
        """Returns list of color answer strings."""
        encoder_hidden = self._get_encoder_hidden(fused, disease)
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        out_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=CFG["max_new_tokens"],
            num_beams=CFG["num_beams"],
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)


class BLIP2FLANLocationHead(nn.Module):
    """
    Route 4 — Location-Related questions.
    Model   : BLIP2-FLAN-T5-XL (Salesforce/blip2-flan-t5-xl)
    Strategy: BLIP2 uses a Query Transformer (Q-Former) to bridge vision
              and language. We inject the Stage 3 conditioning vector into
              the Q-Former's cross-attention by replacing the visual encoder
              hidden states with our Stage 3 representation.
              FLAN-T5-XL then generates the anatomical location.

    Note: BLIP2-FLAN-T5-XL requires ~16GB VRAM. If GPU memory is limited,
          replace with "Salesforce/blip2-flan-t5-base" (~8GB VRAM).
    """
    HIDDEN       = 768    # Q-Former hidden dim
    T5_HIDDEN    = 2048   # FLAN-T5-XL hidden dim
    MODEL        = "Salesforce/blip2-flan-t5-xl"
    MODEL_LIGHT  = "Salesforce/blip2-flan-t5-base"    # fallback

    def __init__(self):
        super().__init__()
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        # Choose model based on available VRAM
        model_id = self.MODEL
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 14:
                model_id = self.MODEL_LIGHT
                print(f"  ⚠️  Low VRAM ({vram_gb:.1f}GB) — using {self.MODEL_LIGHT}")

        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model     = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Project Stage 3 vector → Q-Former hidden dim
        # Expanded to match number of BLIP2 query tokens (32)
        self.projector  = Stage3Projector(self.HIDDEN)
        self.n_queries  = 32
        self.expand     = nn.Linear(self.HIDDEN, self.HIDDEN * self.n_queries)

        # Freeze all BLIP2 layers, only fine-tune last 2 layers of language model
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.language_model.model.decoder.layers[-2:].parameters():
            p.requires_grad = True
        for p in self.model.language_model.lm_head.parameters():
            p.requires_grad = True

        n_proj = sum(p.numel() for p in self.projector.parameters()) + \
                 sum(p.numel() for p in self.expand.parameters())
        n_xfmr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [BLIP2 Loc]    projector={n_proj:,}  trainable_xfmr={n_xfmr:,}")

    def _get_query_embeds(self, fused, disease):
        """
        Build BLIP2 query embeddings from Stage 3 vector.
        Replaces the pixel_values → vision encoder → Q-Former pipeline.
        Shape: (B, n_queries, HIDDEN)
        """
        prefix   = self.projector(fused, disease).squeeze(1)  # (B, H)
        expanded = self.expand(prefix)                         # (B, H*n_queries)
        B = fused.size(0)
        return expanded.view(B, self.n_queries, self.HIDDEN)   # (B, 32, H)

    def forward(self, fused, disease, input_ids, attention_mask, labels):
        """
        Uses query embeddings as the image representation for BLIP2.
        """
        query_embeds = self._get_query_embeds(fused, disease)

        # BLIP2 forward with pre-computed query embeddings
        return self.model(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            query_embeds=query_embeds,
        )

    @torch.no_grad()
    def generate(self, fused, disease, input_ids, attention_mask):
        """Returns list of location answer strings."""
        query_embeds = self._get_query_embeds(fused, disease)
        out_ids = self.model.generate(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_embeds=query_embeds,
            max_new_tokens=CFG["max_new_tokens"],
            num_beams=CFG["num_beams"],
        )
        return self.processor.tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)


class T5LargeCountHead(nn.Module):
    """
    Route 5 — Numerical Count questions.
    Model   : T5-Large (google/t5-large)
    Strategy: Seq2Seq. Stage 3 conditioning projected into T5-Large's
              hidden dim (1024) and prepended as a soft prompt. Decoder
              generates the count answer (e.g. "3", "more than 10").
              T5-Large used over T5-base to better handle numerical reasoning.
    """
    HIDDEN = 1024
    MODEL  = "google/t5-large"

    def __init__(self):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.MODEL)
        self.projector = Stage3Projector(self.HIDDEN)

        self._freeze_except_last_n(n=2)

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_xfmr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [T5-L Count]   projector={n_proj:,}  trainable_xfmr={n_xfmr:,}")

    def _freeze_except_last_n(self, n: int):
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in self.model.encoder.block[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for layer in self.model.decoder.block[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.lm_head.parameters():
            p.requires_grad = True

    def _prepend_prefix(self, input_ids, attention_mask, prefix_embed):
        token_embeds  = self.model.encoder.embed_tokens(input_ids)
        prefix_embed  = prefix_embed.to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embed, token_embeds], dim=1)
        prefix_mask   = torch.ones(
            prefix_embed.size(0), 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_mask

    def forward(self, fused, disease, input_ids, attention_mask, labels):
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, fused, disease, input_ids, attention_mask):
        prefix        = self.projector(fused, disease)
        inputs_embeds, extended_mask = self._prepend_prefix(
            input_ids, attention_mask, prefix)
        out_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            max_new_tokens=CFG["max_new_tokens"],
            num_beams=CFG["num_beams"],
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. STAGE 4 ANSWER GENERATOR — assembles all 6 heads
# ─────────────────────────────────────────────────────────────────────────────
class Stage4TransformerGenerator(nn.Module):
    """
    Complete Stage 4 — 6 specialised transformer heads.
    At training : each head is fine-tuned independently on its route subset.
    At inference: Stage 3 routing selects which head to call.

    Memory note: All 6 models loaded together requires ~30-40GB VRAM.
    For single-GPU training, use --route argument to train one head at a time.
    """
    def __init__(self, vocab: dict, routes: list = None):
        """
        Args:
            vocab  : vocabulary dict (must contain 'multi' key for DeBERTa)
            routes : list of route ints to load (default: all 6).
                     Use [0] to load only FLAN-T5, etc.
        """
        super().__init__()
        routes = routes or list(range(6))
        n_multi = len(vocab.get("multi", ["<unk>"]))

        print(f"\n🧠  Stage4TransformerGenerator — loading routes {routes}")

        self.heads     = nn.ModuleDict()
        self.tokenizers = {}

        if 0 in routes:
            h = FlanT5YesNoHead()
            self.heads["yn"]     = h
            self.tokenizers["yn"] = h.tokenizer

        if 1 in routes:
            h = BartLargeSingleChoiceHead()
            self.heads["single"]     = h
            self.tokenizers["single"] = h.tokenizer

        if 2 in routes:
            h = DeBERTaMultiChoiceHead(n_multi)
            self.heads["multi"]     = h
            self.tokenizers["multi"] = h.tokenizer

        if 3 in routes:
            h = ViTGPT2ColorHead()
            self.heads["color"]     = h
            self.tokenizers["color"] = h.tokenizer

        if 4 in routes:
            h = BLIP2FLANLocationHead()
            self.heads["loc"]     = h
            self.tokenizers["loc"] = h.processor.tokenizer

        if 5 in routes:
            h = T5LargeCountHead()
            self.heads["count"]     = h
            self.tokenizers["count"] = h.tokenizer

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total params    : {n_total:,}")
        print(f"  Trainable params: {n_train:,}")
        print(f"  Frozen params   : {n_total - n_train:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4a. FEATURE CACHE  (run once, reuse every epoch)
# ─────────────────────────────────────────────────────────────────────────────
def cache_stage3_features(extractor: FusionExtractor,
                           text_prep: TextPreprocessor,
                           hf_split, split: str,
                           cache_dir: str,
                           batch_size: int = 64) -> str:
    """
    Pre-extract Stage 3 features for every sample and save to disk.
    This runs ONCE — subsequent calls load from cache instantly.

    Saved file: {cache_dir}/stage3_cache_{split}.pt
    Contains a list of dicts:
        fused_repr  : (512,) float32 tensor
        disease_vec : (23,)  float32 tensor
        route       : int
        question    : str
        answer      : str
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"stage3_cache_{split}.pt")

    if os.path.exists(cache_path):
        print(f"   ✅  Cache found — loading {split} features from {cache_path}")
        return cache_path

    print(f"\n   📦  Pre-extracting Stage 3 features for [{split}] split ...")
    print(f"       {len(hf_split):,} samples  (runs once, then cached)")

    img_tfm = build_image_transform(split)
    device  = CFG["device"]
    records = []

    # Process in batches for speed
    indices = list(range(len(hf_split)))
    for start in tqdm(range(0, len(indices), batch_size),
                      desc=f"   Extracting [{split}]"):
        batch_idx = indices[start : start + batch_size]
        batch_ex  = [hf_split[i] for i in batch_idx]

        imgs  = torch.stack([
            img_tfm(ex["image"].convert("RGB")) for ex in batch_ex
        ]).to(device)

        questions = [ex["question"] for ex in batch_ex]
        q_encs = [text_prep.preprocess(q) for q in questions]
        ids  = torch.stack([e["input_ids"].squeeze(0)       for e in q_encs]).to(device)
        mask = torch.stack([e["attention_mask"].squeeze(0)  for e in q_encs]).to(device)

        with torch.no_grad():
            out = extractor.extract(imgs, ids, mask)

        fused   = out["fused_repr"].cpu()    # (B, 512)
        disease = out["disease_vec"].cpu()   # (B, 23)
        routes  = out["routing_label"].cpu() # (B,)

        for i, ex in enumerate(batch_ex):
            records.append({
                "fused_repr"  : fused[i],
                "disease_vec" : disease[i],
                "route"       : routes[i].item(),
                "question"    : ex["question"],
                "answer"      : ex["answer"].lower().strip(),
            })

    torch.save(records, cache_path)
    print(f"   ✅  Saved {len(records):,} records → {cache_path}")
    return cache_path


# ─────────────────────────────────────────────────────────────────────────────
# 4b. DATASET  (loads from pre-extracted cache — fast)
# ─────────────────────────────────────────────────────────────────────────────
class Stage4TransformerDataset(Dataset):
    """
    Fast dataset for Stage 4 training.
    Loads pre-extracted Stage 3 features from disk cache.
    No GPU inference at all during training — just tensor lookups.

    Returns per sample:
        fused_repr   : (512,)
        disease_vec  : (23,)
        route        : int (0–5)
        question_raw : str
        answer_raw   : str
    """
    def __init__(self, cache_path: str, split: str,
                 vocab: dict, target_route: int = None):
        self.split        = split
        self.vocab        = vocab
        self.target_route = target_route
        self.multi_to_idx = {w: i for i, w in enumerate(vocab.get("multi", []))}

        print(f"   Loading cache [{split}]: {cache_path}")
        all_records = torch.load(cache_path, weights_only=False)

        # Filter to target route if specified
        if target_route is not None:
            self.records = [r for r in all_records
                            if r["route"] == target_route]
            print(f"   Stage4TransformerDataset [{split}]: "
                  f"{len(self.records):,} samples  (route={target_route})"
                  f"  from {len(all_records):,} total")
        else:
            self.records = all_records
            print(f"   Stage4TransformerDataset [{split}]: "
                  f"{len(self.records):,} samples  (all routes)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "fused_repr"  : r["fused_repr"],
            "disease_vec" : r["disease_vec"],
            "route"       : torch.tensor(r["route"], dtype=torch.long),
            "question_raw": r["question"],
            "answer_raw"  : r["answer"],
        }


def collate_fn_for_route(batch, route: int, tokenizer,
                          vocab: dict, device: str):
    """
    Route-specific collation.
    Tokenises questions and answers for the target route's model.
    Returns tensors ready for that model's forward().
    """
    # Filter to this route only
    samples = [s for s in batch if s["route"].item() == route]
    if not samples:
        return None

    fused   = torch.stack([s["fused_repr"]  for s in samples]).float()
    disease = torch.stack([s["disease_vec"] for s in samples]).float()

    # Sanitize any NaN/Inf that Stage 3 may have produced
    fused   = torch.nan_to_num(fused,   nan=0.0, posinf=1.0, neginf=-1.0)
    disease = torch.nan_to_num(disease, nan=0.0, posinf=1.0, neginf=-1.0)

    fused   = fused.to(device)
    disease = disease.to(device)
    questions = [s["question_raw"] for s in samples]
    answers   = [s["answer_raw"]   for s in samples]

    # Tokenise questions (encoder input)
    q_enc = tokenizer(
        questions, padding=True, truncation=True,
        max_length=CFG["max_input_length"], return_tensors="pt",
    ).to(device)

    # Tokenise answers (decoder target labels)
    if route == 2:
        # DeBERTa: multi-hot labels
        n_cls = len(vocab.get("multi", []))
        multi_to_idx = {w: i for i, w in enumerate(vocab.get("multi", []))}
        labels = torch.zeros(len(samples), n_cls)
        for i, a in enumerate(answers):
            for tok in a.split(","):
                tok = tok.strip()
                if tok in multi_to_idx:
                    labels[i, multi_to_idx[tok]] = 1.0
        labels = labels.to(device)
    else:
        # Seq2Seq: tokenise answer text
        a_enc  = tokenizer(
            answers, padding=True, truncation=True,
            max_length=CFG["max_new_tokens"], return_tensors="pt",
        ).to(device)
        labels = a_enc["input_ids"]
        # Replace padding token id with -100 (ignored in loss)
        labels[labels == tokenizer.pad_token_id] = -100

    return {
        "fused"          : fused,
        "disease"        : disease,
        "input_ids"      : q_enc["input_ids"],
        "attention_mask" : q_enc["attention_mask"],
        "labels"         : labels,
        "answers_raw"    : answers,
        "questions_raw"  : questions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING — one route at a time
# ─────────────────────────────────────────────────────────────────────────────
def train_route(route: int, vocab: dict):
    """
    Fine-tune a single transformer head on its dedicated route subset.

    Args:
        route : int 0-5
        vocab : vocabulary dict
    """
    torch.manual_seed(CFG["seed"])
    route_name = ROUTE_NAMES[route]
    head_key   = ROUTE_TO_KEY[route]

    print(f"\n{'='*70}")
    print(f"🚀  Training Route {route}: {route_name.upper()}")
    print(f"{'='*70}")

    # ── Fix relative paths in S3_CFG so FusionExtractor finds checkpoints ──
    # stage3_multimodal_fusion.py uses relative paths (./checkpoints/...)
    # which break when running from src/. Patch them with absolute paths.
    S3_CFG["stage1_ckpt"]   = os.path.join(PROJECT, "checkpoints", "stage1_best.pt")
    S3_CFG["stage2_ckpt"]   = os.path.join(PROJECT, "checkpoints", "best_model")
    S3_CFG["checkpoint_dir"]= os.path.join(PROJECT, "checkpoints")
    S3_CFG["log_dir"]       = os.path.join(PROJECT, "logs")
    S3_CFG["data_dir"]      = os.path.join(HOME,    "data", "kvasir_local")

    # ── Load Stage 3 extractor (frozen) ────────────────────────────────────
    extractor  = FusionExtractor(CFG["stage3_ckpt"])
    text_prep  = TextPreprocessor()

    # ── Dataset ────────────────────────────────────────────────────────────
    from datasets import load_from_disk
    raw = load_from_disk(CFG["data_dir"])

    # Kvasir-VQA-x1 has 'train' and 'test' splits only — no 'validation'.
    # Carve 10% of train as validation set.
    if "validation" in raw:
        train_split = raw["train"]
        val_split   = raw["validation"]
    else:
        split       = raw["train"].train_test_split(test_size=0.10, seed=CFG["seed"])
        train_split = split["train"]
        val_split   = split["test"]
        print(f"   ℹ️  No 'validation' split found — using 90/10 train split")
        print(f"      Train: {len(train_split):,}  Val: {len(val_split):,}")

    # Pre-extract Stage 3 features once and cache to disk
    cache_dir   = os.path.join(PROJECT, "cache", "stage3_features")
    tr_cache    = cache_stage3_features(
        extractor, text_prep, train_split, "train", cache_dir)
    va_cache    = cache_stage3_features(
        extractor, text_prep, val_split, "val", cache_dir)

    tr_set = Stage4TransformerDataset(tr_cache, "train", vocab, target_route=route)
    va_set = Stage4TransformerDataset(va_cache, "val",   vocab, target_route=route)

    tr_loader = DataLoader(tr_set, batch_size=CFG["batch_size"],
                           shuffle=True,  num_workers=CFG["num_workers"],
                           collate_fn=lambda b: b)
    va_loader = DataLoader(va_set, batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=CFG["num_workers"],
                           collate_fn=lambda b: b)

    # ── Build only this route's model ──────────────────────────────────────
    model = Stage4TransformerGenerator(vocab, routes=[route])
    model = model.to(CFG["device"])

    head      = model.heads[head_key]
    tokenizer = model.tokenizers[head_key]

    # ── Optimiser — different LR for projector vs transformer ──────────────
    projector_params = list(head.projector.parameters())
    if hasattr(head, "expand"):
        projector_params += list(head.expand.parameters())
    xfmr_params = [p for n, p in head.named_parameters()
                   if p.requires_grad and "projector" not in n
                   and "expand" not in n]

    # ── Route-specific LR overrides ────────────────────────────────────────
    # BART-Large is very sensitive — needs 5× lower LR than FLAN-T5/T5
    route_lr_override = {
        0: (1e-5, 5e-5),   # FLAN-T5:  (xfmr_lr, proj_lr)
        1: (2e-6, 1e-5),   # BART-Large: must be low or gradients explode
        2: (1e-5, 5e-5),   # DeBERTa
        3: (1e-5, 5e-5),   # ViT-GPT2
        4: (2e-6, 1e-5),   # BLIP2 (large — same as BART)
        5: (1e-5, 5e-5),   # T5-Large
    }
    xfmr_lr, proj_lr = route_lr_override[route]

    optimizer = torch.optim.AdamW([
        {"params": projector_params, "lr": proj_lr},
        {"params": xfmr_params,      "lr": xfmr_lr},
    ], weight_decay=CFG["weight_decay"])

    total_steps    = len(tr_loader) * CFG["epochs"]
    warmup_steps   = int(total_steps * CFG["warmup_ratio"])
    scheduler      = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    scaler   = GradScaler(enabled=CFG["fp16"])
    best_val = float("inf")
    patience = 0
    history  = []
    best_ckpt = os.path.join(
        CFG["ckpt_dir"], f"stage4_{route_name}_best.pt")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(CFG["epochs"]):
        head.train()
        tr_loss   = 0.0
        n_steps   = 0
        n_skipped = 0   # count NaN-skipped batches

        for raw_batch in tqdm(tr_loader,
                              desc=f"  Epoch {epoch+1} [train]",
                              leave=False):
            batch = collate_fn_for_route(
                raw_batch, route, tokenizer, vocab, CFG["device"])
            if batch is None:
                continue

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=CFG["fp16"]):
                if route == 2:
                    out  = head(batch["fused"], batch["disease"],
                                batch["input_ids"], batch["attention_mask"],
                                batch["labels"])
                    loss = out.loss
                elif route == 3:
                    out  = head(batch["fused"], batch["disease"],
                                batch["labels"])
                    loss = out.loss
                else:
                    out  = head(batch["fused"], batch["disease"],
                                batch["input_ids"], batch["attention_mask"],
                                batch["labels"])
                    loss = out.loss

            if not torch.isfinite(loss):
                n_skipped += 1
                continue   # skip NaN/Inf batches — do NOT backprop

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(head.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_loss += loss.item()
            n_steps += 1

        tr_loss /= max(n_steps, 1)
        if n_skipped > 0:
            pct = 100 * n_skipped / max(n_steps + n_skipped, 1)
            print(f"  ⚠️   {n_skipped} NaN batches skipped ({pct:.1f}%) — "
                  f"LR may be too high or labels malformed")

        # ── Validation ─────────────────────────────────────────────────────
        head.eval()
        va_loss = 0.0
        n_va    = 0

        with torch.no_grad():
            for raw_batch in tqdm(va_loader,
                                  desc=f"  Epoch {epoch+1} [val]",
                                  leave=False):
                batch = collate_fn_for_route(
                    raw_batch, route, tokenizer, vocab, CFG["device"])
                if batch is None:
                    continue

                with torch.amp.autocast(device_type="cuda", enabled=CFG["fp16"]):
                    if route == 2:
                        out = head(batch["fused"], batch["disease"],
                                   batch["input_ids"], batch["attention_mask"],
                                   batch["labels"])
                    elif route == 3:
                        out = head(batch["fused"], batch["disease"],
                                   batch["labels"])
                    else:
                        out = head(batch["fused"], batch["disease"],
                                   batch["input_ids"], batch["attention_mask"],
                                   batch["labels"])
                    if torch.isfinite(out.loss):
                        va_loss += out.loss.item()
                        n_va    += 1

        va_loss /= max(n_va, 1)

        print(f"  Epoch {epoch+1:02d}/{CFG['epochs']:02d} "
              f"| tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}")

        history.append({"epoch": epoch+1,
                        "tr_loss": tr_loss, "va_loss": va_loss})

        if va_loss < best_val:
            best_val = va_loss
            patience = 0
            torch.save({
                "model_state" : head.state_dict(),
                "epoch"       : epoch + 1,
                "best_val"    : best_val,
                "route"       : route,
                "route_name"  : route_name,
                "vocab"       : vocab,
            }, best_ckpt)
            print(f"  ✅  New best val_loss={best_val:.4f} → {best_ckpt}")
        else:
            patience += 1
            print(f"  ⏳  patience {patience}/{CFG['early_stop_pat']}")
            if patience >= CFG["early_stop_pat"]:
                print(f"\n🛑  Early stopping at epoch {epoch+1}")
                break

    # ── Save history plot ───────────────────────────────────────────────────
    df  = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["tr_loss"], "b-o", ms=4, label="Train loss")
    ax.plot(df["epoch"], df["va_loss"], "r-o", ms=4, label="Val loss")
    ax.set_title(f"Stage 4 Route {route} ({route_name}) — Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(CFG["log_dir"], f"route{route}_{route_name}_loss.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close()

    df.to_csv(os.path.join(
        CFG["log_dir"], f"route{route}_{route_name}_log.csv"), index=False)
    print(f"\n✅  Route {route} training done. Best val_loss={best_val:.4f}")
    return best_ckpt


def train_all(vocab: dict):
    """Train all 6 routes sequentially."""
    print("\n🚀  Training all 6 transformer heads sequentially ...\n")
    results = {}
    for route in range(6):
        ckpt = train_route(route, vocab)
        results[route] = ckpt
    print("\n✅  All routes trained.")
    for r, ck in results.items():
        print(f"   Route {r} ({ROUTE_NAMES[r]}): {ck}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_route(route: int, vocab: dict, ckpt_path: str = None):
    """
    Evaluate a single trained transformer head.
    Computes exact-match accuracy and sample predictions.
    """
    from datasets import load_from_disk
    route_name = ROUTE_NAMES[route]
    head_key   = ROUTE_TO_KEY[route]

    if ckpt_path is None:
        ckpt_path = os.path.join(
            CFG["ckpt_dir"], f"stage4_{route_name}_best.pt")

    print(f"\n📊  Evaluating Route {route}: {route_name}")

    # Patch S3_CFG absolute paths (same fix as train_route)
    S3_CFG["stage1_ckpt"]    = os.path.join(PROJECT, "checkpoints", "stage1_best.pt")
    S3_CFG["stage2_ckpt"]    = os.path.join(PROJECT, "checkpoints", "best_model")
    S3_CFG["checkpoint_dir"] = os.path.join(PROJECT, "checkpoints")
    S3_CFG["log_dir"]        = os.path.join(PROJECT, "logs")
    S3_CFG["data_dir"]       = os.path.join(HOME,    "data", "kvasir_local")

    extractor = FusionExtractor(CFG["stage3_ckpt"])
    text_prep = TextPreprocessor()
    raw       = load_from_disk(CFG["data_dir"])

    cache_dir = os.path.join(PROJECT, "cache", "stage3_features")
    te_cache  = cache_stage3_features(
        extractor, text_prep, raw["test"], "test", cache_dir)

    te_set = Stage4TransformerDataset(te_cache, "test", vocab, target_route=route)
    te_loader = DataLoader(te_set, batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=CFG["num_workers"],
                           collate_fn=lambda b: b)

    model     = Stage4TransformerGenerator(vocab, routes=[route])
    ckpt      = torch.load(ckpt_path, map_location=CFG["device"])
    model.heads[head_key].load_state_dict(ckpt["model_state"])
    model     = model.to(CFG["device"])
    head      = model.heads[head_key]
    tokenizer = model.tokenizers[head_key]
    head.eval()

    exact_match = 0
    total       = 0
    samples     = []

    with torch.no_grad():
        for raw_batch in tqdm(te_loader, desc="  [eval]", leave=False):
            batch = collate_fn_for_route(
                raw_batch, route, tokenizer, vocab, CFG["device"])
            if batch is None:
                continue

            # Generate predictions
            if route == 2:
                preds = head.predict(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"],
                    vocab.get("multi", []))
            elif route == 3:
                preds = head.generate(batch["fused"], batch["disease"])
            elif route == 4:
                preds = head.generate(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"])
            else:
                preds = head.generate(
                    batch["fused"], batch["disease"],
                    batch["input_ids"], batch["attention_mask"])

            for pred, gt in zip(preds, batch["answers_raw"]):
                pred_clean = pred.strip().lower()
                gt_clean   = gt.strip().lower()
                match      = int(pred_clean == gt_clean)
                exact_match += match
                total       += 1
                if len(samples) < 10:
                    samples.append({
                        "question" : batch["questions_raw"][len(samples)],
                        "gt"       : gt_clean,
                        "pred"     : pred_clean,
                        "correct"  : bool(match),
                    })

    acc = exact_match / max(total, 1)
    print(f"\n  Route {route} ({route_name})")
    print(f"  Exact-match accuracy : {acc:.4f}  ({exact_match}/{total})")
    print(f"\n  Sample predictions:")
    for s in samples[:5]:
        tick = "✅" if s["correct"] else "❌"
        print(f"  {tick}  Q: {s['question'][:60]}")
        print(f"      GT  : {s['gt']}")
        print(f"      Pred: {s['pred']}\n")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL PIPELINE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
class FullPipelinePredictor:
    """
    End-to-end predictor: Image + Question → Answer (transformer-based)
    Loads Stage 3 extractor + all 6 Stage 4 transformer heads.
    """
    def __init__(self, stage3_ckpt: str, stage4_ckpt_dir: str, vocab: dict,
                 routes: list = None):
        self.device = CFG["device"]
        # Patch S3_CFG absolute paths before FusionExtractor loads stage1 ckpt
        S3_CFG["stage1_ckpt"]    = os.path.join(PROJECT, "checkpoints", "stage1_best.pt")
        S3_CFG["stage2_ckpt"]    = os.path.join(PROJECT, "checkpoints", "best_model")
        S3_CFG["checkpoint_dir"] = os.path.join(PROJECT, "checkpoints")
        self.extractor = FusionExtractor(stage3_ckpt)
        self.img_tfm   = build_image_transform("test")
        self.text_prep = TextPreprocessor()
        self.vocab     = vocab

        routes = routes or list(range(6))
        self.model = Stage4TransformerGenerator(vocab, routes=routes)

        # Load each trained head
        for route in routes:
            key       = ROUTE_TO_KEY[route]
            name      = ROUTE_NAMES[route]
            ckpt_path = os.path.join(stage4_ckpt_dir,
                                     f"stage4_{name}_best.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.heads[key].load_state_dict(ckpt["model_state"])
                print(f"  ✅  Loaded route {route} ({name})")
            else:
                print(f"  ⚠️  Checkpoint not found: {ckpt_path}")

        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅  FullPipelinePredictor ready")

    @torch.no_grad()
    def predict(self, image_path: str, question: str) -> dict:
        """
        Full pipeline: image + question → answer.

        Returns:
            answer        : str
            question_type : str
            routing_conf  : float
            disease_vec   : list[float]
            model_used    : str
        """
        from PIL import Image as PILImage
        from stage2_question_categorizer import CLASS_NAMES

        MODEL_USED = {
            "yn": "FLAN-T5", "single": "BART-Large",
            "multi": "DeBERTa-v3", "color": "ViT-GPT2",
            "loc": "BLIP2-FLAN", "count": "T5-Large",
        }

        img    = PILImage.open(image_path).convert("RGB")
        img_t  = self.img_tfm(img).unsqueeze(0).to(self.device)
        q_enc  = self.text_prep.preprocess(question)
        ids    = q_enc["input_ids"].unsqueeze(0).to(self.device)
        mask   = q_enc["attention_mask"].unsqueeze(0).to(self.device)

        fusion_out = self.extractor.extract(img_t, ids, mask)
        fused      = fusion_out["fused_repr"]
        disease    = fusion_out["disease_vec"]
        route      = fusion_out["routing_label"].item()
        route_conf = fusion_out["routing_probs"][0, route].item()
        head_key   = ROUTE_TO_KEY[route]

        head      = self.model.heads[head_key]
        tokenizer = self.model.tokenizers[head_key]

        # Re-tokenise question for this head's tokenizer
        q_tok  = tokenizer(
            [question], padding=True, truncation=True,
            max_length=CFG["max_input_length"], return_tensors="pt",
        ).to(self.device)

        if route == 2:
            answer_list = head.predict(
                fused, disease,
                q_tok["input_ids"], q_tok["attention_mask"],
                self.vocab.get("multi", []))
        elif route == 3:
            answer_list = head.generate(fused, disease)
        elif route == 4:
            answer_list = head.generate(
                fused, disease,
                q_tok["input_ids"], q_tok["attention_mask"])
        else:
            answer_list = head.generate(
                fused, disease,
                q_tok["input_ids"], q_tok["attention_mask"])

        return {
            "answer"        : answer_list[0],
            "question_type" : CLASS_NAMES[route],
            "routing_conf"  : round(route_conf, 4),
            "model_used"    : MODEL_USED[head_key],
            "disease_vec"   : disease[0].cpu().tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. VOCABULARY BUILDER (reused from original stage4)
# ─────────────────────────────────────────────────────────────────────────────
def build_vocabulary():
    from datasets import load_from_disk
    print("📦  Building vocabularies ...")
    raw = load_from_disk(CFG["data_dir"])

    multi_counter = Counter()
    for ex in tqdm(raw["train"], desc="  Scanning", leave=False):
        route = infer_qtype_label(ex["question"], ex["answer"])
        if route == 2:
            for tok in ex["answer"].lower().split(","):
                tok = tok.strip()
                if tok:
                    multi_counter[tok] += 1

    multi_vocab = ["<unk>"] + \
                  [w for w, _ in multi_counter.most_common(CFG["max_vocab_multi"]-1)]
    vocab = {"multi": multi_vocab,
             "multi_counts": dict(multi_counter.most_common(30))}

    with open(CFG["vocab_file"], "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"✅  Multi-choice vocab: {len(multi_vocab)} tokens → {CFG['vocab_file']}")
    return vocab


def load_vocabulary():
    if not os.path.exists(CFG["vocab_file"]):
        return build_vocabulary()
    with open(CFG["vocab_file"]) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4 — Transformer-based Answer Generation")
    parser.add_argument("--mode",
        choices=["demo", "build_vocab", "train", "eval", "infer"],
        default="demo",
        help="demo | build_vocab | train | eval | infer")
    parser.add_argument("--route",
        default="all",
        help="Route to train/eval: 0-5 or 'all'  (default: all)")
    parser.add_argument("--ckpt_dir",
        default=CFG["ckpt_dir"],
        help="Directory containing stage4_*_best.pt checkpoints")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--question",   default=None)
    args = parser.parse_args()

    routes = list(range(6)) if args.route == "all" \
             else [int(args.route)]

    # ── demo ──────────────────────────────────────────────────────────────
    if args.mode == "demo":
        print("\n🧠  Stage 4 Transformer — Architecture Demo\n")
        print("  Models per route:")
        model_map = {
            0: "FLAN-T5-base     (google/flan-t5-base)",
            1: "BART-Large       (facebook/bart-large)",
            2: "DeBERTa-v3-base  (microsoft/deberta-v3-base)",
            3: "ViT-GPT2         (nlpconnect/vit-gpt2-image-captioning)",
            4: "BLIP2-FLAN-T5-XL (Salesforce/blip2-flan-t5-xl)",
            5: "T5-Large         (google/t5-large)",
        }
        for r, m in model_map.items():
            print(f"    Route {r} [{ROUTE_NAMES[r]:15s}] → {m}")

        print("\n  Conditioning mechanism:")
        print("    535-D vector (512 fused + 23 disease)")
        print("    → Stage3Projector (Linear + GELU + LayerNorm + Dropout)")
        print("    → Soft-prompt token prepended to encoder input")
        print("    → Transformer generates answer in natural language")

        print("\n  VRAM requirements (approx.):")
        print("    Route 0 FLAN-T5-base   :  ~2 GB")
        print("    Route 1 BART-Large     :  ~6 GB")
        print("    Route 2 DeBERTa-v3     :  ~3 GB")
        print("    Route 3 ViT-GPT2       :  ~2 GB")
        print("    Route 4 BLIP2-FLAN-XL  : ~16 GB  (use base for <14GB)")
        print("    Route 5 T5-Large       :  ~6 GB")
        print("\n  Recommended: train one route at a time")
        print("    python stage4_transformers.py --mode train --route 0")

    # ── build_vocab ────────────────────────────────────────────────────────
    elif args.mode == "build_vocab":
        build_vocabulary()

    # ── train ─────────────────────────────────────────────────────────────
    elif args.mode == "train":
        vocab = load_vocabulary()
        if args.route == "all":
            train_all(vocab)
        else:
            train_route(int(args.route), vocab)

    # ── eval ──────────────────────────────────────────────────────────────
    elif args.mode == "eval":
        vocab = load_vocabulary()
        results = {}
        for r in routes:
            acc = evaluate_route(r, vocab)
            results[r] = acc
        print("\n📊  Summary:")
        for r, a in results.items():
            print(f"   Route {r} [{ROUTE_NAMES[r]:15s}]: {a:.4f}")

    # ── infer ─────────────────────────────────────────────────────────────
    elif args.mode == "infer":
        if not args.image_path or not args.question:
            print("❌  --image_path and --question required for infer mode")
            print("    Example:")
            print("    python stage4_transformers.py --mode infer \\")
            print("        --image_path ./sample.jpg \\")
            print('        --question "Is there a polyp visible?"')
        else:
            vocab     = load_vocabulary()
            predictor = FullPipelinePredictor(
                CFG["stage3_ckpt"], args.ckpt_dir, vocab, routes=routes)
            result = predictor.predict(args.image_path, args.question)
            print(f"\n{'='*55}")
            print(f"  Question     : {args.question}")
            print(f"  Answer       : {result['answer']}")
            print(f"  Q-type       : {result['question_type']}")
            print(f"  Model used   : {result['model_used']}")
            print(f"  Route conf   : {result['routing_conf']:.4f}")
            active = [i for i, p in enumerate(result['disease_vec']) if p > 0.5]
            print(f"  Diseases     : {active}")
            print(f"{'='*55}")
