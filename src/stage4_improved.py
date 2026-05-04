"""
=============================================================================
STAGE 4 — IMPROVED ANSWER GENERATION
Fix 1: Canonical vocabulary (200→50 single-choice classes)
Fix 2: Class-weighted loss + label smoothing + longer training + cosine LR

Expected improvement: 73.97% → ~90%+
=============================================================================
"""

import os, sys, json, re, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Paths
    data_dir      = "./data/kvasir_local",
    vocab_file    = "./data/stage4_vocab_v2.json",   # new vocab
    stage3_ckpt   = "./checkpoints/stage3_best.pt",
    save_ckpt     = "./checkpoints/stage4_v2_best.pt",
    log_dir       = "./logs/stage4_v2",

    # Architecture
    fused_dim     = 512,
    disease_dim   = 23,
    head_hidden   = 256,
    dropout       = 0.2,

    # Fix 2: Training improvements
    epochs        = 40,          # was 20
    batch_size    = 64,
    lr            = 1e-4,        # was 3e-4 — lower + more stable
    weight_decay  = 0.01,
    label_smooth  = 0.1,         # NEW — prevents overconfidence
    warmup_steps  = 200,         # NEW — gentle warmup
    patience      = 8,           # was 5 — more patience
    fp16          = True,
    seed          = 42,
    device        = "cuda" if torch.cuda.is_available() else "cpu",

    # Answer classes (unchanged)
    yn_classes       = ["yes", "no"],
    color_classes    = ["red","pink","white","yellow","brown",
                        "green","purple","orange","black","blue",
                        "grey","mixed"],
    location_classes = ["cecum","sigmoid","rectum","descending colon",
                        "transverse colon","ascending colon","hepatic flexure",
                        "splenic flexure","terminal ileum","duodenum",
                        "stomach","esophagus","ileocecal valve",
                        "appendiceal orifice","retroflex"],
    count_classes    = ["0","1","2","3","4","5","6 or more","more than 10"],

    # Fix 1: Canonical vocab sizes
    n_single_canonical = 50,     # was 200
    n_multi_canonical  = 80,     # was 150
)

os.makedirs(CFG["log_dir"],    exist_ok=True)
os.makedirs("./checkpoints",   exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

DEVICE = CFG["device"]
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────
# FIX 1 — CANONICAL VOCABULARY BUILDER
# ─────────────────────────────────────────────────────────────────────────

# Manual canonical mapping — groups near-duplicate answers
CANONICAL_RULES = [
    # Polyp types
    (r"sessile polyp",          "sessile polyp"),
    (r"pedunculated polyp",     "pedunculated polyp"),
    (r"hyperplastic polyp",     "hyperplastic polyp"),
    (r"colonic polyp",          "colonic polyp"),
    (r"polyp",                  "colonic polyp"),           # catch-all polyp

    # Procedures
    (r"colonoscopy procedure",  "colonoscopy procedure"),
    (r"colonoscop",             "colonoscopy procedure"),
    (r"gastroscop",             "gastroscopy procedure"),
    (r"upper endoscop",         "upper endoscopy procedure"),

    # Artifacts
    (r"green.*black.*box",      "green/black box artifact"),
    (r"black.*green.*box",      "green/black box artifact"),
    (r"box artifact",           "green/black box artifact"),

    # Instruments
    (r"polyp snare",            "polyp snare instrument"),
    (r"biopsy forcep",          "biopsy forceps instrument"),
    (r"injection needle",       "injection needle instrument"),
    (r"instrument",             "surgical instrument"),

    # Normal findings
    (r"normal.*cecum|cecum.*normal",       "normal cecum"),
    (r"normal.*pylorus|pylorus.*normal",   "normal pylorus"),
    (r"normal.*z.line|z.line.*normal",     "normal z-line"),
    (r"ileocecal valve",                   "ileocecal valve"),

    # Pathologies
    (r"ulcerative colitis",     "ulcerative colitis"),
    (r"esophagitis",            "esophagitis"),
    (r"gastritis",              "gastritis"),
    (r"barrett",                "barrett's esophagus"),
    (r"hemorrhoid",             "hemorrhoids"),
    (r"diverticul",             "diverticulosis"),
    (r"erosion",                "erosion"),
    (r"ulcer",                  "gastric ulcer"),
    (r"crohn",                  "crohn's disease"),

    # Dyed
    (r"dyed.*lifted|lifted.*dyed",         "dyed lifted polyp"),
    (r"dyed.*resection|resection.*margin", "dyed resection margin"),

    # Text/image annotations
    (r"text is present",        "text present on image"),
    (r"text.*visible",          "text present on image"),
    (r"no.*text",               "no text on image"),

    # Residual polyp
    (r"no residual polyp",      "no residual polyp tissue"),
    (r"residual polyp",         "residual polyp present"),
    (r"polyp.*remain",          "residual polyp present"),

    # Foreign body
    (r"foreign body",           "foreign body present"),

    # Size descriptions
    (r"5.*10.*mm|small polyp",  "small polyp 5-10mm"),
    (r"10.*20.*mm|medium polyp","medium polyp 10-20mm"),
    (r"larger.*20|large polyp", "large polyp >20mm"),

    # No findings
    (r"no polyp|no polypoid",   "no polyp identified"),
    (r"no significant finding", "no significant findings"),
    (r"no abnormalit",          "no abnormality detected"),
    (r"normal finding",         "normal findings"),

    # Retroflex
    (r"retroflex.*rectum",      "retroflex rectum view"),
    (r"retroflex.*stomach",     "retroflex stomach view"),

    # Scattered / distributed
    (r"scatter|distribut|multipl.*area|central.*upper",
                                "scattered distribution"),
]

def canonicalise(text: str) -> str:
    """Map any answer string to its canonical form."""
    t = text.lower().strip()
    for pattern, canonical in CANONICAL_RULES:
        if re.search(pattern, t):
            return canonical
    # Fallback — take first 5 words as canonical
    words = t.split()[:5]
    return " ".join(words)


def build_canonical_vocabulary(n_samples: int = 100000):
    """
    Build reduced canonical vocabulary from training data.
    Fix 1: 200 → 50 single-choice classes.
    """
    from datasets import load_from_disk, Image as HFImage
    from stage3_multimodal_fusion import infer_qtype_label

    print("\n📦  Building canonical vocabulary (Fix 1) ...")
    raw = load_from_disk(CFG["data_dir"])

    single_counter = Counter()
    multi_counter  = Counter()
    total          = min(n_samples, len(raw["train"]))

    for i, ex in enumerate(tqdm(raw["train"],
                                total=total,
                                desc="   scanning")):
        if i >= total:
            break
        q = ex["question"]
        a = ex["answer"].lower().strip()
        r = infer_qtype_label(q, a)

        if r == 1:  # single-choice
            canon = canonicalise(a)
            single_counter[canon] += 1
        elif r == 2:  # multi-choice
            for tok in re.split(r"[,\.\s]+", a):
                tok = tok.strip()
                if len(tok) > 2:
                    multi_counter[tok] += 1

    # Take top-N canonical classes
    single_vocab = [w for w, _ in
                    single_counter.most_common(CFG["n_single_canonical"])]
    multi_vocab  = [w for w, _ in
                    multi_counter.most_common(CFG["n_multi_canonical"])]

    vocab = {
        "single"  : single_vocab,
        "multi"   : multi_vocab,
        "single_original_size": len(single_counter),
        "multi_original_size" : len(multi_counter),
        "version" : "v2_canonical",
    }

    with open(CFG["vocab_file"], "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\n   ✅  Single-choice: {len(single_counter)} → {len(single_vocab)} classes")
    print(f"   ✅  Multi-choice : {len(multi_counter)} unique tokens → {len(multi_vocab)}")
    print(f"\n   Top 20 canonical single-choice answers:")
    for w, c in single_counter.most_common(20):
        print(f"      {c:>5}×  {w}")
    print(f"\n   Vocab saved → {CFG['vocab_file']}")
    return vocab


def load_vocabulary_v2():
    if not os.path.exists(CFG["vocab_file"]):
        return build_canonical_vocabulary()
    with open(CFG["vocab_file"]) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────
class Stage4DatasetV2(Dataset):
    """
    Same as original Stage4Dataset but uses canonical answer mapping
    for single-choice route.
    """
    def __init__(self, hf_split, split, extractor,
                 text_prep, vocab, max_samples=None):
        from stage3_multimodal_fusion import infer_qtype_label

        self.vocab        = vocab
        self.extractor    = extractor
        self.text_prep    = text_prep
        self.split        = split
        self.infer_route  = infer_qtype_label

        # Index maps
        self.single_to_idx = {w: i for i, w in enumerate(vocab["single"])}
        self.multi_to_idx  = {w: i for i, w in enumerate(vocab["multi"])}

        # Pre-filter valid samples
        self.data = []
        total = max_samples or len(hf_split)
        for i, ex in enumerate(tqdm(hf_split,
                                    total=total,
                                    desc=f"   Building {split} dataset",
                                    leave=False)):
            if i >= total:
                break
            if ex["image"] is not None:
                self.data.append(ex)

        print(f"   Stage4DatasetV2 [{split}]: {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex    = self.data[idx]
        q     = ex["question"]
        a     = ex["answer"].lower().strip()
        route = self.infer_route(q, a)

        # Encode image + text → fused features
        from preprocessing import build_image_transform
        import torchvision.transforms as T
        tfm = build_image_transform("test")

        try:
            img_pil = ex["image"].convert("RGB")
            img_t   = tfm(img_pil).unsqueeze(0).to(DEVICE)
        except Exception:
            img_t = torch.zeros(1, 3, 224, 224).to(DEVICE)

        q_enc = self.text_prep.preprocess(q)
        ids   = q_enc["input_ids"].unsqueeze(0).to(DEVICE)
        mask  = q_enc["attention_mask"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = self.extractor.extract(img_t, ids, mask)

        fused   = out["fused_repr"].squeeze(0).cpu()
        disease = out["disease_vec"].squeeze(0).cpu()
        r_probs = out["routing_probs"].squeeze(0).cpu()

        # ── Labels ─────────────────────────────────────────────────
        yn_label = 1 if a.startswith("yes") else 0

        # FIX 1 — canonical mapping for single-choice
        canon    = canonicalise(a)
        single_label = self.single_to_idx.get(
            canon, self.single_to_idx.get(a, 0))

        multi_label = torch.zeros(len(self.multi_to_idx))
        for tok in re.split(r"[,\.\s]+", a):
            tok = tok.strip()
            if tok in self.multi_to_idx:
                multi_label[self.multi_to_idx[tok]] = 1.0

        color_label = next((i for i, c in enumerate(CFG["color_classes"])
                            if c in a), 0)
        loc_label   = next((i for i, l in enumerate(CFG["location_classes"])
                            if l.replace("-"," ") in a or l in a), 0)
        count_map   = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5}
        count_label = count_map.get(a.split()[0] if a else "0", 0)
        if any(str(n) in a for n in range(6, 11)): count_label = 6
        if "more than 10" in a or "many" in a:     count_label = 7

        return dict(
            fused_repr   = fused,
            disease_vec  = disease,
            route_probs  = r_probs,
            route        = torch.tensor(route,       dtype=torch.long),
            yn_label     = torch.tensor(yn_label,    dtype=torch.long),
            single_label = torch.tensor(single_label,dtype=torch.long),
            multi_label  = multi_label,
            color_label  = torch.tensor(color_label, dtype=torch.long),
            loc_label    = torch.tensor(loc_label,   dtype=torch.long),
            count_label  = torch.tensor(count_label, dtype=torch.long),
            answer_raw   = ex["answer"],
            question_raw = q,
        )


# ─────────────────────────────────────────────────────────────────────────
# MODEL — same architecture, smaller single-choice head
# ─────────────────────────────────────────────────────────────────────────
class AnswerHeadV2(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        in_dim = CFG["fused_dim"] + CFG["disease_dim"]  # 535
        self.net = nn.Sequential(
            nn.Linear(in_dim, CFG["head_hidden"]),
            nn.GELU(),
            nn.LayerNorm(CFG["head_hidden"]),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["head_hidden"], n_classes),
        )
    def forward(self, fused, disease):
        x = torch.cat([fused, disease], dim=-1)
        return self.net(x)


class Stage4AnswerGeneratorV2(nn.Module):
    def __init__(self, vocab: dict):
        super().__init__()
        self.heads = nn.ModuleDict({
            "yn"    : AnswerHeadV2(len(CFG["yn_classes"])),
            "single": AnswerHeadV2(len(vocab["single"])),
            "multi" : AnswerHeadV2(len(vocab["multi"])),
            "color" : AnswerHeadV2(len(CFG["color_classes"])),
            "loc"   : AnswerHeadV2(len(CFG["location_classes"])),
            "count" : AnswerHeadV2(len(CFG["count_classes"])),
        })
        self.route_to_head = {
            0:"yn", 1:"single", 2:"multi",
            3:"color", 4:"loc", 5:"count"
        }
        n_total = sum(p.numel() for p in self.parameters())
        print(f"\n🧠  Stage4AnswerGeneratorV2")
        print(f"    6 heads  |  Total params: {n_total:,}")
        for name, head in self.heads.items():
            n = sum(p.numel() for p in head.parameters())
            out = list(head.net.parameters())[-1].shape[0]
            print(f"    [{name:<6}] → {out:>4} classes  |  {n:,} params")

    def forward(self, fused, disease, route: int):
        key = self.route_to_head[route]
        return self.heads[key](fused, disease)

    @torch.no_grad()
    def predict(self, fused, disease, route: int, vocab: dict) -> list:
        logits = self.forward(fused, disease, route)
        if route == 2:
            probs = torch.sigmoid(logits.float())
            preds = (probs > 0.5).cpu().numpy()
            answers = []
            for row in preds:
                active = [vocab["multi"][j]
                          for j, v in enumerate(row) if v]
                answers.append(", ".join(active) or "no findings")
            return answers
        else:
            idx = logits.float().argmax(-1).cpu().numpy()
            classes = {
                0: CFG["yn_classes"],
                1: vocab["single"],
                3: CFG["color_classes"],
                4: CFG["location_classes"],
                5: CFG["count_classes"],
            }
            return [str(classes[route][i])
                    if i < len(classes[route]) else "<unk>"
                    for i in idx]


# ─────────────────────────────────────────────────────────────────────────
# FIX 2 — IMPROVED TRAINING
# ─────────────────────────────────────────────────────────────────────────
def compute_class_weights(dataset, vocab):
    """Compute inverse-frequency class weights per route."""
    print("\n⚖️   Computing class weights ...")
    route_labels = {r: [] for r in range(6)}
    label_keys   = {0:"yn_label", 1:"single_label", 2:None,
                    3:"color_label", 4:"loc_label", 5:"count_label"}

    for ex in tqdm(dataset, desc="   counting", leave=False):
        r = ex["route"].item()
        if label_keys[r] is not None:
            route_labels[r].append(ex[label_keys[r]].item())

    weights = {}
    for r in [0, 1, 3, 4, 5]:
        labels = route_labels[r]
        if not labels:
            weights[r] = None
            continue
        n_classes = {0:2, 1:len(vocab["single"]),
                     3:len(CFG["color_classes"]),
                     4:len(CFG["location_classes"]),
                     5:len(CFG["count_classes"])}[r]
        counts = Counter(labels)
        total  = len(labels)
        w = torch.ones(n_classes)
        for cls, cnt in counts.items():
            if cls < n_classes:
                w[cls] = total / (n_classes * cnt)
        w = w.clamp(max=10.0)   # cap extreme weights
        weights[r] = w.to(DEVICE)
        print(f"   Route {r}: {n_classes} classes  "
              f"max_weight={w.max():.2f}  min={w.min():.2f}")
    return weights


def train():
    from datasets import load_from_disk, Image as HFImage
    from stage3_multimodal_fusion import FusionExtractor
    from preprocessing import TextPreprocessor
    from transformers import get_cosine_schedule_with_warmup

    # Load data
    print("\n📂  Loading dataset ...")
    raw       = load_from_disk(CFG["data_dir"])
    raw       = raw.cast_column("image", HFImage())
    text_prep = TextPreprocessor()

    print("📦  Loading vocabulary (Fix 1 — canonical) ...")
    vocab = load_vocabulary_v2()

    print("🔗  Loading Stage 3 extractor ...")
    extractor = FusionExtractor(CFG["stage3_ckpt"])

    print("📊  Building datasets ...")
    train_ds = Stage4DatasetV2(raw["train"], "train",
                               extractor, text_prep, vocab)
    test_ds  = Stage4DatasetV2(raw["test"],  "test",
                               extractor, text_prep, vocab)

    train_ld = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          shuffle=True,  num_workers=0,
                          pin_memory=False)
    test_ld  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0)

    # Model
    model  = Stage4AnswerGeneratorV2(vocab).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(),
                               lr=CFG["lr"],
                               weight_decay=CFG["weight_decay"])
    total_steps = len(train_ld) * CFG["epochs"]
    sched  = get_cosine_schedule_with_warmup(
        opt, CFG["warmup_steps"], total_steps)
    scaler = GradScaler(enabled=CFG["fp16"])

    # FIX 2a — class-weighted + label-smoothed losses
    class_weights = compute_class_weights(train_ds, vocab)
    loss_fns = {}
    for r in range(6):
        if r == 2:
            loss_fns[r] = nn.BCEWithLogitsLoss()
        else:
            w = class_weights.get(r, None)
            loss_fns[r] = nn.CrossEntropyLoss(
                weight=w,
                label_smoothing=CFG["label_smooth"]   # FIX 2b
            )

    label_keys = {0:"yn_label", 1:"single_label", 2:"multi_label",
                  3:"color_label", 4:"loc_label",  5:"count_label"}

    # Training loop
    best_val   = 0.0
    patience_c = 0
    history    = []

    print(f"\n🚀  Training  ({CFG['epochs']} epochs, "
          f"lr={CFG['lr']}, label_smooth={CFG['label_smooth']})\n")

    for ep in range(1, CFG["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0; tr_correct = 0; tr_total = 0

        for batch in tqdm(train_ld,
                          desc=f"  Ep {ep:02d} train",
                          leave=False):
            opt.zero_grad()
            fused   = batch["fused_repr"].to(DEVICE)
            disease = batch["disease_vec"].to(DEVICE)
            loss    = torch.tensor(0., device=DEVICE)

            for r in range(6):
                mask = (batch["route"] == r)
                if mask.sum() == 0: continue
                f_r  = fused[mask]
                d_r  = disease[mask]
                lbls = batch[label_keys[r]][mask].to(DEVICE)
                with autocast(enabled=CFG["fp16"]):
                    logits = model(f_r, d_r, r)
                    if r == 2:
                        l = loss_fns[r](logits.float(),
                                        lbls.float())
                    else:
                        l = loss_fns[r](logits.float(), lbls)
                if not torch.isnan(l):
                    loss = loss + l
                    if r != 2:
                        tr_correct += (logits.float().argmax(-1)
                                       == lbls).sum().item()
                        tr_total   += mask.sum().item()

            if loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sched.step()
            tr_loss += loss.item()

        tr_acc  = tr_correct / max(tr_total, 1)
        tr_loss /= len(train_ld)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_correct = {r: 0 for r in range(6)}
        val_total   = {r: 0 for r in range(6)}

        with torch.no_grad():
            for batch in tqdm(test_ld,
                              desc=f"  Ep {ep:02d} val  ",
                              leave=False):
                fused   = batch["fused_repr"].to(DEVICE)
                disease = batch["disease_vec"].to(DEVICE)
                for r in range(6):
                    mask = (batch["route"] == r)
                    if mask.sum() == 0: continue
                    lbls   = batch[label_keys[r]][mask].to(DEVICE)
                    logits = model(fused[mask], disease[mask], r)
                    if r == 2:
                        p = (torch.sigmoid(logits.float())>0.5).float()
                        c = (p == lbls.float()).all(dim=1).float()
                    else:
                        p = logits.float().argmax(-1)
                        c = (p == lbls).float()
                    val_correct[r] += c.sum().item()
                    val_total[r]   += mask.sum().item()

        # Overall val accuracy
        val_acc = (sum(val_correct.values()) /
                   max(sum(val_total.values()), 1))

        # Per-route
        route_accs = {r: val_correct[r]/max(val_total[r],1)
                      for r in range(6)}

        is_best = val_acc > best_val
        if is_best:
            best_val   = val_acc
            patience_c = 0
            torch.save({"model_state": model.state_dict(),
                        "vocab"      : vocab,
                        "epoch"      : ep,
                        "val_acc"    : val_acc,
                        "cfg"        : CFG},
                       CFG["save_ckpt"])
            marker = " ← BEST"
        else:
            patience_c += 1
            marker = f" (p{patience_c})"

        print(f"  Ep {ep:02d}  "
              f"loss={tr_loss:.4f}  "
              f"tr_acc={tr_acc*100:.2f}%  "
              f"val_acc={val_acc*100:.2f}%"
              f"{marker}")
        print(f"         "
              + "  ".join(f"{['yn','sc','mc','col','loc','cnt'][r]}="
                          f"{route_accs[r]*100:.1f}%"
                          for r in range(6)))

        history.append(dict(epoch=ep, tr_loss=tr_loss,
                            tr_acc=tr_acc, val_acc=val_acc,
                            **{f"r{r}": route_accs[r] for r in range(6)}))

        if patience_c >= CFG["patience"]:
            print(f"\n  Early stopping (patience={CFG['patience']})")
            break

    # ── Save training history ──────────────────────────────────────
    df = pd.DataFrame(history)
    df.to_csv(f"{CFG['log_dir']}/training_log.csv", index=False)

    # ── Final plot ──────────────────────────────────────────────────
    _plot_training(df, best_val)
    print(f"\n{'='*60}")
    print(f"✅  Training complete  |  Best val_acc = {best_val*100:.2f}%")
    print(f"   Checkpoint → {CFG['save_ckpt']}")
    print(f"{'='*60}")
    return best_val


def _plot_training(df: pd.DataFrame, best_val: float):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stage 4 v2 Training — Best Val Acc = {best_val*100:.2f}%\n"
                 f"Fix 1: Canonical vocab ({CFG['n_single_canonical']} classes)  "
                 f"Fix 2: Class weights + Label smoothing ({CFG['label_smooth']})",
                 fontsize=11, fontweight="bold")

    axes[0].plot(df["epoch"], df["tr_loss"], "b-o", ms=4, label="Train Loss")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["tr_acc"]*100, "b-o", ms=4, label="Train")
    axes[1].plot(df["epoch"], df["val_acc"]*100, "r-o", ms=4, label="Val")
    axes[1].axhline(73.97, color="gray", linestyle="--", lw=1.5,
                    label="Baseline (73.97%)")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Per-route final
    route_names  = ["yes/no","single","multi","color","loc","count"]
    route_colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336","#00BCD4"]
    for r in range(6):
        if f"r{r}" in df.columns:
            axes[2].plot(df["epoch"], df[f"r{r}"]*100,
                         color=route_colors[r], lw=1.5,
                         label=route_names[r], marker=".", ms=3)
    axes[2].set_title("Per-Route Val Accuracy")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("%")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    # Baseline reference lines
    baselines = [100, 50.6, 33.3, 80.0, 88.1, 54.1]
    for r, b in enumerate(baselines):
        axes[2].axhline(b, color=route_colors[r],
                        linestyle=":", alpha=0.4, lw=1)

    plt.tight_layout()
    path = f"{CFG['log_dir']}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊  Plot → {path}")


# ─────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["build_vocab","train","all"],
        default="all")
    args = parser.parse_args()

    if args.mode in ("build_vocab","all"):
        build_canonical_vocabulary()

    if args.mode in ("train","all"):
        best = train()
        print(f"\n🎯  Final best val accuracy: {best*100:.2f}%")
        print(f"   Compare to baseline:      73.97%")
        print(f"   Improvement:             +{(best-0.7397)*100:.2f}pp")
