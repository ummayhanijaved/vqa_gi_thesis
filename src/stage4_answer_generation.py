"""
=============================================================================
STAGE 4: Specialised Answer Generation
Routes each sample to one of 6 specialised answer heads based on
Stage 3 routing decision, then generates the final answer.

Architecture:
    Stage 3 output:
        fused_repr   (512-D)  → shared input to all heads
        routing_label (0-5)   → selects which head to use
        disease_vec   (23-D)  → appended for disease-aware answering

    6 Specialised Answer Heads:
        [0] yes/no          → Binary classifier    (2 classes)
        [1] single-choice   → Vocabulary classifier (top-K answers)
        [2] multiple-choice → Multi-label classifier
        [3] color           → Color classifier     (fixed palette)
        [4] location        → Location classifier  (anatomical sites)
        [5] numerical count → Ordinal classifier   (0-10+)

    Each head:
        Input  : concat(fused_repr[512], disease_vec[23]) = 535-D
        Hidden : Linear(535→256) → GELU → LayerNorm → Dropout
        Output : Linear(256→n_classes)

=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================

USAGE:
    python stage4_answer_generation.py --mode demo
    python stage4_answer_generation.py --mode build_vocab
    python stage4_answer_generation.py --mode train
    python stage4_answer_generation.py --mode eval   --checkpoint ./checkpoints/stage4_best.pt
    python stage4_answer_generation.py --mode infer  --checkpoint ./checkpoints/stage4_best.pt
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

sys.path.insert(0, os.path.expanduser("~"))
from preprocessing import build_image_transform, TextPreprocessor, clean_text
from stage1_disease_classifier import TreeNetDiseaseClassifier
from stage3_multimodal_fusion import (
    Stage3MultimodalFusion, FusionExtractor,
    infer_qtype_label, CFG as S3_CFG
)

print("✅  Imports from preprocessing / stage1 / stage3 successful")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Dimensions
    fused_dim       = 512,      # Stage 3 fused_repr
    disease_dim     = 23,       # Stage 3 disease_vec
    head_input_dim  = 535,      # 512 + 23
    head_hidden_dim = 256,
    dropout         = 0.2,

    # Fixed answer classes per route
    yn_classes      = ["no", "yes"],   # route 0
    color_classes   = [                # route 3
        "pink", "red", "orange", "yellow", "green",
        "blue", "purple", "white", "black", "brown",
        "transparent", "mixed"
    ],
    location_classes = [               # route 4
        "esophagus", "stomach", "duodenum", "small-bowel",
        "colon", "rectum", "cecum", "pylorus", "z-line",
        "retroflex-rectum", "retroflex-stomach", "ileocecal-valve",
        "upper-gi", "lower-gi", "unknown"
    ],
    count_classes   = [                # route 5
        "0", "1", "2", "3", "4", "5", "6-10", "more than 10"
    ],

    # Vocabulary (built from training data)
    vocab_file      = "./data/stage4_vocab.json",
    max_vocab_single= 200,    # top-200 single-choice answers
    max_vocab_multi = 150,    # top-150 multi-choice answer tokens

    # Training
    epochs          = 20,
    batch_size      = 64,
    learning_rate   = 3e-4,
    weight_decay    = 0.01,
    warmup_steps    = 200,
    early_stop_pat  = 5,
    grad_clip       = 1.0,
    fp16            = torch.cuda.is_available(),

    # Paths
    stage3_ckpt     = "./checkpoints/stage3_best.pt",
    stage4_ckpt_dir = "./checkpoints",
    data_dir        = "./data/kvasir_local",
    log_dir         = "./logs",
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers     = 0,
    seed            = 42,
)

os.makedirs(CFG["log_dir"],         exist_ok=True)
os.makedirs(CFG["stage4_ckpt_dir"], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. VOCABULARY BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_vocabulary():
    """
    Scan training answers for routes 1 (single) and 2 (multi)
    and build top-K answer vocabularies.
    """
    from datasets import load_from_disk, Image as HFImage
    print("📦  Building answer vocabularies from training data ...")
    raw = load_from_disk(CFG["data_dir"])

    single_counter = Counter()
    multi_counter  = Counter()

    for ex in tqdm(raw["train"], desc="   Scanning answers", leave=False):
        q, a = ex["question"], ex["answer"]
        route = infer_qtype_label(q, a)
        a_clean = a.lower().strip()

        if route == 1:   # single-choice
            single_counter[a_clean] += 1
        elif route == 2:  # multiple-choice
            # Each comma-separated token is a class
            for token in a_clean.split(","):
                token = token.strip()
                if token:
                    multi_counter[token] += 1

    # Build vocab lists
    single_vocab = ["<unk>"] + \
                   [w for w, _ in single_counter.most_common(CFG["max_vocab_single"]-1)]
    multi_vocab  = ["<unk>"] + \
                   [w for w, _ in multi_counter.most_common(CFG["max_vocab_multi"]-1)]

    vocab = {
        "single": single_vocab,
        "multi" : multi_vocab,
        "single_counts": dict(single_counter.most_common(50)),
        "multi_counts" : dict(multi_counter.most_common(50)),
    }

    with open(CFG["vocab_file"], "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"   Single-choice vocab : {len(single_vocab):,} classes")
    print(f"   Multi-choice vocab  : {len(multi_vocab):,} tokens")
    print(f"   Top single answers  : {list(single_counter.most_common(5))}")
    print(f"   Top multi tokens    : {list(multi_counter.most_common(5))}")
    print(f"✅  Vocab saved → {CFG['vocab_file']}")
    return vocab


def load_vocabulary():
    if not os.path.exists(CFG["vocab_file"]):
        return build_vocabulary()
    with open(CFG["vocab_file"]) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────
class Stage4Dataset(Dataset):
    """
    Dataset for Stage 4 training.
    Uses Stage 3 FusionExtractor to get fused_repr and disease_vec
    on-the-fly during training (both frozen, so fast).

    Returns per sample:
        fused_repr   : (512,)
        disease_vec  : (23,)
        route        : int (0-5)
        target_*     : answer label per route
        answer_raw   : str
        question_raw : str
    """
    def __init__(self, hf_split, split: str,
                 extractor: FusionExtractor,
                 text_prep: TextPreprocessor,
                 vocab: dict):
        self.data      = hf_split
        self.split     = split
        self.extractor = extractor
        self.text_prep = text_prep
        self.img_tfm   = build_image_transform(split)   # from preprocessing.py

        # Build answer → index maps
        self.single_to_idx = {w: i for i, w in enumerate(vocab["single"])}
        self.multi_to_idx  = {w: i for i, w in enumerate(vocab["multi"])}
        self.yn_to_idx     = {w: i for i, w in enumerate(CFG["yn_classes"])}
        self.color_to_idx  = {w: i for i, w in enumerate(CFG["color_classes"])}
        self.loc_to_idx    = {w: i for i, w in enumerate(CFG["location_classes"])}
        self.count_to_idx  = {w: i for i, w in enumerate(CFG["count_classes"])}

        print(f"   Stage4Dataset [{split}]: {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # ── Image ──────────────────────────────────────────────────────────
        img = ex["image"].convert("RGB")
        img_tensor = self.img_tfm(img)

        # ── Text ───────────────────────────────────────────────────────────
        q_enc = self.text_prep.preprocess(ex["question"])

        # ── Stage 3 features (no_grad, frozen) ─────────────────────────────
        with torch.no_grad():
            out = self.extractor.extract(
                img_tensor.unsqueeze(0).to(CFG["device"]),
                q_enc["input_ids"].unsqueeze(0).to(CFG["device"]),
                q_enc["attention_mask"].unsqueeze(0).to(CFG["device"]),
            )
        fused_repr  = out["fused_repr"][0].cpu()      # (512,)
        disease_vec = out["disease_vec"][0].cpu()      # (23,)
        route       = out["routing_label"][0].item()   # int 0-5

        # ── Answer labels per route ─────────────────────────────────────────
        a = ex["answer"].lower().strip()

        # Route 0: yes/no
        yn_label = self.yn_to_idx.get(
            "yes" if a.startswith("yes") else "no", 0)

        # Route 1: single-choice
        single_label = self.single_to_idx.get(a, 0)  # 0 = <unk>

        # Route 2: multiple-choice (multi-hot)
        multi_label = torch.zeros(len(self.multi_to_idx))
        for token in a.split(","):
            token = token.strip()
            if token in self.multi_to_idx:
                multi_label[self.multi_to_idx[token]] = 1.0

        # Route 3: color
        color_label = 0
        for i, c in enumerate(CFG["color_classes"]):
            if c in a:
                color_label = i; break

        # Route 4: location
        loc_label = 0
        for i, loc in enumerate(CFG["location_classes"]):
            if loc.replace("-", " ") in a or loc in a:
                loc_label = i; break

        # Route 5: count
        count_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
                     "5": 5, "none": 0, "one": 1, "two": 2,
                     "three": 3, "four": 4, "five": 5}
        count_label = 0
        for word, idx_c in count_map.items():
            if word in a:
                count_label = idx_c; break
        # Check for large counts
        if any(str(n) in a for n in range(6, 11)):
            count_label = 6
        if "more than 10" in a or "many" in a:
            count_label = 7

        return {
            "fused_repr"   : fused_repr,
            "disease_vec"  : disease_vec,
            "route"        : torch.tensor(route, dtype=torch.long),
            "yn_label"     : torch.tensor(yn_label, dtype=torch.long),
            "single_label" : torch.tensor(single_label, dtype=torch.long),
            "multi_label"  : multi_label,
            "color_label"  : torch.tensor(color_label, dtype=torch.long),
            "loc_label"    : torch.tensor(loc_label, dtype=torch.long),
            "count_label"  : torch.tensor(count_label, dtype=torch.long),
            "question_raw" : ex["question"],
            "answer_raw"   : ex["answer"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANSWER HEAD (shared architecture for all routes)
# ─────────────────────────────────────────────────────────────────────────────
class AnswerHead(nn.Module):
    """
    Single specialised answer head.
    Input  : concat(fused_repr[512], disease_vec[23]) = 535-D
    Output : logits over n_classes

    All 6 heads share this architecture — only n_classes differs.
    """
    def __init__(self, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(CFG["head_input_dim"], CFG["head_hidden_dim"]),
            nn.GELU(),
            nn.LayerNorm(CFG["head_hidden_dim"]),
            nn.Dropout(dropout),
            nn.Linear(CFG["head_hidden_dim"], n_classes),
        )

    def forward(self, fused: torch.Tensor,
                disease: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fused, disease], dim=-1)  # (B, 535)
        return self.head(x)                       # (B, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FULL STAGE 4 MODEL
# ─────────────────────────────────────────────────────────────────────────────
class Stage4AnswerGenerator(nn.Module):
    """
    Complete Stage 4:
        6 specialised AnswerHead instances, one per question type.
        At inference: stage 3 routing selects which head to use.
        At training : all heads trained simultaneously on their
                      respective subsets.
    """
    def __init__(self, vocab: dict):
        super().__init__()

        n_single = len(vocab["single"])
        n_multi  = len(vocab["multi"])

        self.heads = nn.ModuleDict({
            "yn"    : AnswerHead(len(CFG["yn_classes"])),       # 2
            "single": AnswerHead(n_single),                      # 200
            "multi" : AnswerHead(n_multi),                       # 150
            "color" : AnswerHead(len(CFG["color_classes"])),     # 12
            "loc"   : AnswerHead(len(CFG["location_classes"])),  # 15
            "count" : AnswerHead(len(CFG["count_classes"])),     # 8
        })

        self.route_to_head = {
            0: "yn", 1: "single", 2: "multi",
            3: "color", 4: "loc", 5: "count"
        }

        n_total = sum(p.numel() for p in self.parameters())
        print(f"\n🧠  Stage4AnswerGenerator")
        print(f"    6 specialised heads  |  Total params: {n_total:,}")
        for name, h in self.heads.items():
            n = sum(p.numel() for p in h.parameters())
            out_dim = list(h.head.children())[-1].out_features
            print(f"    [{name:8s}] → {out_dim:4d} classes  |  {n:,} params")

    def forward(self, fused: torch.Tensor,
                disease: torch.Tensor,
                route: int) -> torch.Tensor:
        """
        Args:
            fused   : (B, 512)
            disease : (B, 23)
            route   : int 0-5
        Returns:
            logits  : (B, n_classes) for this route
        """
        head_name = self.route_to_head[route]
        return self.heads[head_name](fused, disease)

    def predict(self, fused: torch.Tensor,
                disease: torch.Tensor,
                route: int,
                vocab: dict,
                threshold: float = 0.5) -> list:
        """
        Full prediction with answer string decoding.
        Returns list of answer strings.
        """
        logits = self.forward(fused, disease, route)

        if route == 0:   # yes/no
            preds = logits.argmax(-1)
            return [CFG["yn_classes"][p] for p in preds.tolist()]

        elif route == 1:  # single-choice
            preds = logits.argmax(-1)
            return [vocab["single"][p] for p in preds.tolist()]

        elif route == 2:  # multiple-choice (multi-label)
            probs = torch.sigmoid(logits)
            answers = []
            for row in probs:
                active = [vocab["multi"][i]
                          for i, p in enumerate(row) if p > threshold]
                answers.append(", ".join(active) if active else "<none>")
            return answers

        elif route == 3:  # color
            preds = logits.argmax(-1)
            return [CFG["color_classes"][p] for p in preds.tolist()]

        elif route == 4:  # location
            preds = logits.argmax(-1)
            return [CFG["location_classes"][p] for p in preds.tolist()]

        elif route == 5:  # count
            preds = logits.argmax(-1)
            return [CFG["count_classes"][p] for p in preds.tolist()]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def get_loss_for_route(model, batch, route_idx, device):
    """Compute loss for a specific route subset of a batch."""
    mask = (batch["route"] == route_idx)
    if mask.sum() == 0:
        return None

    fused   = batch["fused_repr"][mask].to(device)
    disease = batch["disease_vec"][mask].to(device)

    head_map = {
        0: ("yn_label",     nn.CrossEntropyLoss()),
        1: ("single_label", nn.CrossEntropyLoss()),
        2: ("multi_label",  nn.BCEWithLogitsLoss()),
        3: ("color_label",  nn.CrossEntropyLoss()),
        4: ("loc_label",    nn.CrossEntropyLoss()),
        5: ("count_label",  nn.CrossEntropyLoss()),
    }
    label_key, loss_fn = head_map[route_idx]
    labels  = batch[label_key][mask].to(device)
    logits  = model(fused, disease, route_idx)

    if route_idx == 2:
        labels = labels.float()

    return loss_fn(logits, labels)


def compute_accuracy(model, batch, route_idx, device):
    """Compute accuracy for a route subset of a batch."""
    mask = (batch["route"] == route_idx)
    if mask.sum() == 0:
        return None, 0

    fused   = batch["fused_repr"][mask].to(device)
    disease = batch["disease_vec"][mask].to(device)

    label_keys = {0:"yn_label", 1:"single_label", 2:"multi_label",
                  3:"color_label", 4:"loc_label", 5:"count_label"}
    labels = batch[label_keys[route_idx]][mask].to(device)
    logits = model(fused, disease, route_idx)

    if route_idx == 2:
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc   = (preds == labels.float()).all(dim=1).float().mean().item()
    else:
        preds = logits.argmax(-1)
        acc   = (preds == labels).float().mean().item()

    return acc, mask.sum().item()


def train():
    from transformers import get_cosine_schedule_with_warmup
    from datasets import load_from_disk, Image as HFImage
    from sklearn.model_selection import train_test_split

    torch.manual_seed(CFG["seed"])

    vocab = load_vocabulary()
    text_prep = TextPreprocessor()

    print("📦  Loading Stage 3 extractor ...")
    extractor = FusionExtractor(CFG["stage3_ckpt"])

    print("📦  Loading dataset ...")
    raw = load_from_disk(CFG["data_dir"])
    raw = raw.cast_column("image", HFImage())

    # Split train → train/val
    indices = list(range(len(raw["train"])))
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(indices, test_size=0.20,
                                      random_state=CFG["seed"])

    train_ds = Stage4Dataset(raw["train"].select(tr_idx),
                             "train", extractor, text_prep, vocab)
    val_ds   = Stage4Dataset(raw["train"].select(va_idx),
                             "val",   extractor, text_prep, vocab)
    test_ds  = Stage4Dataset(raw["test"],
                             "test",  extractor, text_prep, vocab)

    kw = dict(num_workers=CFG["num_workers"], pin_memory=False)
    tr_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                           shuffle=True, **kw)
    va_loader = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                           shuffle=False, **kw)
    te_loader = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                           shuffle=False, **kw)

    model = Stage4AnswerGenerator(vocab).to(CFG["device"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["learning_rate"],
                                  weight_decay=CFG["weight_decay"])
    total_steps = len(tr_loader) * CFG["epochs"]
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = CFG["warmup_steps"],
        num_training_steps = total_steps,
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG["fp16"])
    best_acc  = 0.; patience = 0
    best_ckpt = os.path.join(CFG["stage4_ckpt_dir"], "stage4_best.pt")
    history   = []

    print(f"\n🚀  Training Stage 4 on {CFG['device'].upper()}"
          f"  |  FP16={CFG['fp16']}\n" + "="*70)

    for epoch in range(CFG["epochs"]):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.; n_batches = 0
        route_correct = {r: 0 for r in range(6)}
        route_total   = {r: 0 for r in range(6)}

        for batch in tqdm(tr_loader,
                          desc=f"Ep {epoch+1:02d} [train]",
                          leave=False):
            optimizer.zero_grad()
            loss = torch.tensor(0., device=CFG["device"])

            for route_idx in range(6):
                with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
                    route_loss = get_loss_for_route(
                        model, batch, route_idx, CFG["device"])
                if route_loss is not None and not torch.isnan(route_loss):
                    loss = loss + route_loss

            if loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(optimizer); scaler.update()

            scheduler.step()
            total_loss += loss.item()
            n_batches  += 1

            # Track per-route accuracy
            with torch.no_grad():
                for r in range(6):
                    acc, cnt = compute_accuracy(
                        model, batch, r, CFG["device"])
                    if acc is not None:
                        route_correct[r] += acc * cnt
                        route_total[r]   += cnt

        tr_loss = total_loss / max(n_batches, 1)
        tr_acc  = (sum(route_correct.values()) /
                   max(sum(route_total.values()), 1))

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_correct = {r: 0 for r in range(6)}
        val_total   = {r: 0 for r in range(6)}

        with torch.no_grad():
            for batch in tqdm(va_loader, desc="  [val]", leave=False):
                for r in range(6):
                    acc, cnt = compute_accuracy(
                        model, batch, r, CFG["device"])
                    if acc is not None:
                        val_correct[r] += acc * cnt
                        val_total[r]   += cnt

        va_acc = (sum(val_correct.values()) /
                  max(sum(val_total.values()), 1))

        # Per-route val accuracies
        route_accs = {r: (val_correct[r] / max(val_total[r], 1))
                      for r in range(6)}

        row = dict(epoch=epoch+1,
                   tr_loss=round(tr_loss,4), tr_acc=round(tr_acc,4),
                   va_acc=round(va_acc,4),
                   **{f"va_{r}": round(route_accs[r], 4)
                      for r in range(6)})
        history.append(row)

        print(f"Ep {epoch+1:02d}  tr_loss={tr_loss:.4f}"
              f"  tr_acc={tr_acc:.4f}"
              f" | va_acc={va_acc:.4f}")
        for r in range(6):
            from stage2_question_categorizer import CLASS_NAMES
            print(f"         [{CLASS_NAMES[r]:16s}]"
                  f"  val_acc={route_accs[r]:.4f}"
                  f"  (n={val_total[r]:,})")

        if va_acc > best_acc:
            best_acc = va_acc; patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch"      : epoch + 1,
                "best_acc"   : best_acc,
                "vocab"      : vocab,
            }, best_ckpt)
            print(f"   ✅  New best val_acc={best_acc:.4f} → {best_ckpt}")
        else:
            patience += 1
            print(f"   ⏳  patience {patience}/{CFG['early_stop_pat']}")
            if patience >= CFG["early_stop_pat"]:
                print(f"\n🛑  Early stopping at epoch {epoch+1}")
                break

    # ── Test ─────────────────────────────────────────────────────────────
    print("\n" + "="*70 + "\n📊  Test evaluation …")
    ckpt = torch.load(best_ckpt, map_location=CFG["device"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    te_correct = {r: 0 for r in range(6)}
    te_total   = {r: 0 for r in range(6)}
    with torch.no_grad():
        for batch in tqdm(te_loader, desc="  [test]", leave=False):
            for r in range(6):
                acc, cnt = compute_accuracy(
                    model, batch, r, CFG["device"])
                if acc is not None:
                    te_correct[r] += acc * cnt
                    te_total[r]   += cnt

    te_acc = (sum(te_correct.values()) /
              max(sum(te_total.values()), 1))

    print(f"\n   Overall Test Accuracy : {te_acc:.4f}")
    from stage2_question_categorizer import CLASS_NAMES
    for r in range(6):
        r_acc = te_correct[r] / max(te_total[r], 1)
        print(f"   [{CLASS_NAMES[r]:16s}] acc={r_acc:.4f}"
              f"  (n={te_total[r]:,})")

    pd.DataFrame(history).to_csv(
        f"{CFG['log_dir']}/stage4_epoch_log.csv", index=False)
    _save_plots(history)
    print(f"\n✅  Stage 4 done. Best val_acc = {best_acc:.4f}")


def _save_plots(history):
    df  = pd.DataFrame(history)
    eps = df["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Stage 4: Specialised Answer Generation — Training",
                 fontsize=12, fontweight="bold")
    axes[0].plot(eps, df["tr_loss"], "b-o", ms=4, label="Train loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(eps, df["tr_acc"]*100, "b-o", ms=4, label="Train acc")
    axes[1].plot(eps, df["va_acc"]*100, "r-o", ms=4, label="Val acc")
    axes[1].set_title("Overall Accuracy (%)"); axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = f"{CFG['log_dir']}/stage4_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📈  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE INFERENCE  (Image + Question → Answer)
# ─────────────────────────────────────────────────────────────────────────────
class FullPipelinePredictor:
    """
    End-to-end predictor: Image + Question → Answer
    Loads Stage 3 extractor + Stage 4 answer heads.
    This is the complete thesis pipeline inference interface.
    """
    def __init__(self, stage3_ckpt: str, stage4_ckpt: str):
        self.device    = CFG["device"]
        self.extractor = FusionExtractor(stage3_ckpt)
        self.img_tfm   = build_image_transform("test")
        self.text_prep = TextPreprocessor()

        ckpt = torch.load(stage4_ckpt, map_location=self.device)
        self.vocab = ckpt["vocab"]
        self.model = Stage4AnswerGenerator(self.vocab).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"✅  FullPipelinePredictor ready")

    @torch.no_grad()
    def predict(self, image_path: str, question: str) -> dict:
        """
        Full pipeline: image path + question → answer

        Returns dict with:
            answer        : str   — predicted answer
            question_type : str   — detected question type
            routing_conf  : float — routing confidence
            disease_vec   : list  — 23-D disease probabilities
            fused_repr    : list  — 512-D fused representation
        """
        from PIL import Image as PILImage

        # Load and preprocess image
        img    = PILImage.open(image_path).convert("RGB")
        img_t  = self.img_tfm(img).unsqueeze(0).to(self.device)

        # Tokenize question
        q_enc = self.text_prep.preprocess(question)
        ids   = q_enc["input_ids"].unsqueeze(0).to(self.device)
        mask  = q_enc["attention_mask"].unsqueeze(0).to(self.device)

        # Stage 3: fused representation + routing
        fusion_out = self.extractor.extract(img_t, ids, mask)
        fused      = fusion_out["fused_repr"]       # (1, 512)
        disease    = fusion_out["disease_vec"]       # (1, 23)
        route      = fusion_out["routing_label"].item()
        route_conf = fusion_out["routing_probs"][0, route].item()

        # Stage 4: answer generation
        answers = self.model.predict(
            fused, disease, route, self.vocab)

        from stage2_question_categorizer import CLASS_NAMES
        return {
            "answer"       : answers[0],
            "question_type": CLASS_NAMES[route],
            "routing_conf" : round(route_conf, 4),
            "disease_vec"  : disease[0].cpu().tolist(),
            "fused_repr"   : fused[0].cpu().tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["demo", "build_vocab", "train", "eval", "infer"],
        default="demo")
    parser.add_argument("--checkpoint",
        default="./checkpoints/stage4_best.pt")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--question",   default=None)
    args = parser.parse_args()

    if args.mode == "demo":
        print("\n🧠  Stage 4 model summary (no data needed) …\n")
        dummy_vocab = {
            "single": ["<unk>"] + [f"ans_{i}" for i in range(199)],
            "multi" : ["<unk>"] + [f"tok_{i}" for i in range(149)],
        }
        model = Stage4AnswerGenerator(dummy_vocab)

        # Simulate Stage 3 outputs
        B = 4
        dummy_fused   = torch.randn(B, 512)
        dummy_disease = torch.rand(B, 23)

        print("\n   Simulated inference per route:")
        from stage2_question_categorizer import CLASS_NAMES
        for r in range(6):
            logits = model(dummy_fused, dummy_disease, r)
            print(f"   Route {r} [{CLASS_NAMES[r]:16s}]"
                  f" → logits shape: {logits.shape}")

        print(f"\n✅  Stage 4 architecture verified.")
        print(f"    Next: python stage4_answer_generation.py --mode build_vocab")
        print(f"    Then: python stage4_answer_generation.py --mode train")

    elif args.mode == "build_vocab":
        build_vocabulary()

    elif args.mode == "train":
        train()

    elif args.mode == "eval":
        print("📊  Evaluation mode — loading checkpoint ...")
        ckpt  = torch.load(args.checkpoint, map_location=CFG["device"])
        vocab = ckpt["vocab"]
        model = Stage4AnswerGenerator(vocab).to(CFG["device"])
        model.load_state_dict(ckpt["model_state"])
        print(f"✅  Loaded checkpoint (epoch={ckpt['epoch']}"
              f"  best_acc={ckpt['best_acc']:.4f})")

    elif args.mode == "infer":
        if not args.image_path or not args.question:
            print("❌  --image_path and --question required for infer mode")
            print("    Example:")
            print("    python stage4_answer_generation.py --mode infer \\")
            print("        --checkpoint ./checkpoints/stage4_best.pt \\")
            print("        --image_path ./sample.jpg \\")
            print('        --question "Is there a polyp visible?"')
        else:
            predictor = FullPipelinePredictor(
                CFG["stage3_ckpt"], args.checkpoint)
            result = predictor.predict(args.image_path, args.question)
            print(f"\n{'='*50}")
            print(f"Question      : {args.question}")
            print(f"Answer        : {result['answer']}")
            print(f"Question type : {result['question_type']}")
            print(f"Route conf    : {result['routing_conf']:.4f}")
            print(f"Active diseases: {[i for i, p in enumerate(result['disease_vec']) if p > 0.5]}")
            print(f"{'='*50}")
