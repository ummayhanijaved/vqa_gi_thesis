"""
=============================================================================
THESIS STAGE 2: Question Analysis Module
Fine-tuned DistilBERT for 6-class Question Type Classification
=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
Dataset : Kvasir-VQA-x1  (SimulaMet/Kvasir-VQA-x1 on HuggingFace)
Classes : 0=Yes/No | 1=Single-Choice | 2=Multiple-Choice |
          3=Color   | 4=Location      | 5=Numerical-Count
=============================================================================
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
CFG = dict(
    model_name          = "distilbert-base-uncased",
    num_labels          = 6,
    max_length          = 128,
    epochs              = 10,
    batch_size          = 32,
    learning_rate       = 2e-5,
    weight_decay        = 0.01,
    warmup_steps        = 500,
    early_stop_patience = 3,
    grad_clip           = 1.0,
    dropout             = 0.1,
    data_dir            = "./data",
    checkpoint_dir      = "./checkpoints",
    log_dir             = "./logs",
    seed                = 42,
    device              = "cuda" if torch.cuda.is_available() else "cpu",
    fp16                = torch.cuda.is_available(),
)

LABEL2ID = {
    "yes/no"          : 0,
    "single-choice"   : 1,
    "multiple-choice" : 2,
    "color"           : 3,
    "location"        : 4,
    "numerical count" : 5,
}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(CFG["num_labels"])]

# ─────────────────────────────────────────────
# 1.  REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

# ─────────────────────────────────────────────
# 2.  LABEL INFERENCE  (question + answer text)
# ─────────────────────────────────────────────
def infer_label(question: str, answer: str) -> int:
    """
    Kvasir-VQA-x1 uses 'question_class' with semantic tags like
    ['polyp_count', 'instrument_presence'] — NOT the 6 VQA categories.
    We deterministically derive the category from linguistic patterns.

        0 = yes/no           answer is yes / no variant
        1 = single-choice    single descriptive answer (fallback)
        2 = multiple-choice  comma-separated list of items in answer
        3 = color            question asks about color/colour
        4 = location         question asks where / location / position
        5 = numerical count  question asks how many / count
    """
    q = question.lower().strip()
    a = answer.lower().strip()

    YES_NO = {
        "yes", "no", "yes, it is", "no, it is not",
        "yes, there is", "no, there is not",
        "yes, it does", "no, it does not",
        "yes, there are", "no, there are not",
    }

    if any(w in q for w in ("color", "colour", "what color", "what colour")):
        return 3
    if any(w in q for w in ("where", "location", "located", "position",
                             "which part", "which area", "which region", "which side")):
        return 4
    if any(w in q for w in ("how many", "count", "number of", "total number")):
        return 5
    if a in YES_NO or a.startswith("yes") or a.startswith("no"):
        return 0
    if "," in a and len(a) < 200:
        return 2
    return 1   # single-choice default

# ─────────────────────────────────────────────
# 3.  DATA LOADING
# ─────────────────────────────────────────────
def load_kvasir_vqa_x1():
    print("📦  Loading Kvasir-VQA-x1 from HuggingFace …")
    from datasets import load_dataset
    raw = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CFG["data_dir"])

    sample = raw["train"][0]
    print(f"   Columns : {list(sample.keys())}")
    print(f"   Sample  : q={sample['question']!r}  a={sample['answer']!r}")

    def hf_to_df(split):
        return pd.DataFrame([
            {"question": ex["question"].strip(),
             "label"   : infer_label(ex["question"], ex["answer"])}
            for ex in split
        ])

    test_df    = hf_to_df(raw["test"])
    full_train = hf_to_df(raw["train"])

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        full_train, test_size=0.20,
        stratify=full_train["label"],
        random_state=CFG["seed"],
    )

    print(f"   Train : {len(train_df):,}  |  Val : {len(val_df):,}  |  Test : {len(test_df):,}")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df["label"].value_counts().sort_index()
        print(f"   [{name}] " + "  ".join(f"{ID2LABEL[i]}={v}" for i, v in dist.items()))

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))

# ─────────────────────────────────────────────
# 4.  DATASET & DATALOADERS
# ─────────────────────────────────────────────
class QuestionTypeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.questions  = df["question"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.questions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }

def build_loaders(train_df, val_df, test_df, tokenizer):
    kw = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(
        QuestionTypeDataset(train_df, tokenizer, CFG["max_length"]),
        batch_size=CFG["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(
        QuestionTypeDataset(val_df,   tokenizer, CFG["max_length"]),
        batch_size=CFG["batch_size"], shuffle=False, **kw)
    test_loader  = DataLoader(
        QuestionTypeDataset(test_df,  tokenizer, CFG["max_length"]),
        batch_size=CFG["batch_size"], shuffle=False, **kw)
    return train_loader, val_loader, test_loader

# ─────────────────────────────────────────────
# 5.  MODEL
# ─────────────────────────────────────────────
class QuestionTypeClassifier(nn.Module):
    """
    DistilBERT (66 M params) + classification head → 6 classes.
    Uses DistilBERT's built-in seq_classif_dropout parameter.
    """
    def __init__(self, num_labels=6, dropout=0.1):
        super().__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            CFG["model_name"],
            num_labels=num_labels,
            seq_classif_dropout=dropout,   # correct param for DistilBERT
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

# ─────────────────────────────────────────────
# 6.  CLASS-WEIGHTED LOSS
# ─────────────────────────────────────────────
def compute_class_weights(train_df):
    labels        = train_df["label"].values
    present       = np.unique(labels)
    missing       = set(range(CFG["num_labels"])) - set(present.tolist())
    if missing:
        print(f"   ⚠️  Classes with 0 samples (will get weight=1.0): "
              f"{ {ID2LABEL[i] for i in missing} }")

    # Compute weights only for classes that actually appear
    partial = compute_class_weight(
        class_weight="balanced", classes=present, y=labels)

    # Build full weight tensor; missing classes get weight 1.0
    weights = np.ones(CFG["num_labels"], dtype=np.float32)
    for cls, w in zip(present, partial):
        weights[cls] = w

    print(f"⚖️   Class weights: { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32).to(CFG["device"])

# ─────────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0., 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(CFG["device"])
        attn_mask = batch["attention_mask"].to(CFG["device"])
        labels    = batch["labels"].to(CFG["device"])

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
            logits = model(input_ids, attn_mask).logits
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer); scaler.update(); scheduler.step()

        preds          = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_correct/total_samples:.4f}")

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, desc="val"):
    model.eval()
    total_loss, all_preds, all_labels = 0., [], []

    for batch in tqdm(loader, desc=f"  [{desc}]", leave=False):
        input_ids = batch["input_ids"].to(CFG["device"])
        attn_mask = batch["attention_mask"].to(CFG["device"])
        labels    = batch["labels"].to(CFG["device"])

        with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
            logits = model(input_ids, attn_mask).logits
            loss   = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n        = len(all_labels)
    avg_loss = total_loss / n
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, macro_f1, all_preds, all_labels

# ─────────────────────────────────────────────
# 8.  MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────
def train():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CFG["log_dir"],        exist_ok=True)

    train_df, val_df, test_df = load_kvasir_vqa_x1()
    tokenizer = DistilBertTokenizerFast.from_pretrained(CFG["model_name"])
    train_loader, val_loader, test_loader = build_loaders(train_df, val_df, test_df, tokenizer)

    model = QuestionTypeClassifier(num_labels=CFG["num_labels"],
                                   dropout=CFG["dropout"]).to(CFG["device"])
    print(f"🧠  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(train_df)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    no_decay   = ["bias", "LayerNorm.weight"]
    opt_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": CFG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(opt_groups, lr=CFG["learning_rate"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = CFG["warmup_steps"],
        num_training_steps = len(train_loader) * CFG["epochs"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["fp16"])

    history        = {"train_loss": [], "val_loss": [],
                      "train_acc":  [], "val_acc":  [], "val_f1": []}
    best_val_acc   = 0.
    patience_count = 0
    best_ckpt      = os.path.join(CFG["checkpoint_dir"], "best_model")

    print(f"\n🚀  Training on {CFG['device'].upper()}  |  FP16={CFG['fp16']}\n" + "="*60)

    for epoch in range(CFG["epochs"]):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, epoch)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, loss_fn, "val")

        history["train_loss"].append(tr_loss);  history["val_loss"].append(va_loss)
        history["train_acc"] .append(tr_acc);   history["val_acc"] .append(va_acc)
        history["val_f1"]    .append(va_f1)

        print(f"Epoch {epoch+1:02d}/{CFG['epochs']}  "
              f"| tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"| va_loss={va_loss:.4f}  va_acc={va_acc:.4f}  va_f1={va_f1:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc; patience_count = 0
            model.distilbert.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"   ✅  New best val_acc={best_val_acc:.4f}  → {best_ckpt}")
        else:
            patience_count += 1
            print(f"   ⏳  No improvement ({patience_count}/{CFG['early_stop_patience']})")
            if patience_count >= CFG["early_stop_patience"]:
                print(f"\n🛑  Early stopping at epoch {epoch+1}."); break

    # ── Final test evaluation ──
    print("\n" + "="*60 + "\n📊  Test evaluation …")
    model.distilbert = DistilBertForSequenceClassification.from_pretrained(
        best_ckpt).to(CFG["device"])
    _, te_acc, te_f1, te_preds, te_labels = evaluate(model, test_loader, loss_fn, "test")
    print(f"   Test Accuracy : {te_acc:.4f}   Macro-F1 : {te_f1:.4f}\n")
    print(classification_report(te_labels, te_preds, target_names=CLASS_NAMES))

    save_training_plots(history)
    save_confusion_matrix(te_labels, te_preds)

    with open(os.path.join(CFG["log_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅  Done.  Best val_acc = {best_val_acc:.4f}")

# ─────────────────────────────────────────────
# 9.  VISUALIZATION
# ─────────────────────────────────────────────
def save_training_plots(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val")
    axes[0].set_title("Cross-Entropy Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_f1"], "g-o")
    axes[2].set_title("Val Macro-F1"); axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CFG["log_dir"], "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📈  Saved training curves → {path}")


def save_confusion_matrix(labels, preds):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix — Question Type Classification")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    path = os.path.join(CFG["log_dir"], "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📉  Saved confusion matrix → {path}")

# ─────────────────────────────────────────────
# 10. INFERENCE CLASS  (used by Stage 3)
# ─────────────────────────────────────────────
class QuestionTypePredictor:
    """
    Drop-in predictor for Stage 3 fusion module.

    Usage:
        predictor = QuestionTypePredictor("./checkpoints/best_model")
        result    = predictor.predict("Is there a polyp visible?")
        # → {"label": "yes/no", "label_id": 0, "confidence": 0.987,
        #    "probabilities": {"yes/no": 0.987, "color": 0.002, ...}}
    """
    def __init__(self, checkpoint_dir):
        self.device    = CFG["device"]
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(
            checkpoint_dir).to(self.device)
        self.model.eval()
        print(f"✅  Predictor loaded from {checkpoint_dir}")

    @torch.no_grad()
    def predict(self, question: str) -> dict:
        enc    = self.tokenizer(question, max_length=CFG["max_length"],
                                padding="max_length", truncation=True,
                                return_tensors="pt")
        logits = self.model(enc["input_ids"].to(self.device),
                            enc["attention_mask"].to(self.device)).logits
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred   = int(probs.argmax())
        return {
            "label"        : ID2LABEL[pred],
            "label_id"     : pred,
            "confidence"   : float(probs[pred]),
            "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        }

    @torch.no_grad()
    def predict_batch(self, questions: list) -> list:
        enc    = self.tokenizer(questions, max_length=CFG["max_length"],
                                padding="max_length", truncation=True,
                                return_tensors="pt")
        logits = self.model(enc["input_ids"].to(self.device),
                            enc["attention_mask"].to(self.device)).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for p in probs:
            pred = int(p.argmax())
            results.append({
                "label"        : ID2LABEL[pred],
                "label_id"     : pred,
                "confidence"   : float(p[pred]),
                "probabilities": {ID2LABEL[i]: float(v) for i, v in enumerate(p)},
            })
        return results

# ─────────────────────────────────────────────
# 11. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model")
    parser.add_argument("--question",   default="Is there a polyp visible?")
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "eval":
        _, _, test_df = load_kvasir_vqa_x1()
        tokenizer     = DistilBertTokenizerFast.from_pretrained(args.checkpoint)
        model         = DistilBertForSequenceClassification.from_pretrained(
            args.checkpoint).to(CFG["device"])
        _, test_loader = build_loaders(
            pd.DataFrame(columns=["question","label"]),
            pd.DataFrame(columns=["question","label"]),
            test_df, tokenizer)[1:]
        loss_fn = nn.CrossEntropyLoss()
        _, acc, f1, preds, labels = evaluate(model, test_loader, loss_fn, "test")
        print(f"Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
        print(classification_report(labels, preds, target_names=CLASS_NAMES))

    elif args.mode == "infer":
        predictor = QuestionTypePredictor(args.checkpoint)
        result    = predictor.predict(args.question)
        print(f"\nQuestion  : {args.question}")
        print(f"Predicted : {result['label']}  (id={result['label_id']})")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        for cls, prob in result["probabilities"].items():
            print(f"  {cls:<20s}  {prob:.4f}  {'█' * int(prob*40)}")
