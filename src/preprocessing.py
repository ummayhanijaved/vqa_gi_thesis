"""
=============================================================================
PREPROCESSING MODULE
Image normalization/augmentation + Text cleaning/tokenization
For: Kvasir-VQA-x1 GI Endoscopy Dataset
=============================================================================
Author  : Ummay Hani Javed (24i-8211)
Thesis  : Advancing Medical AI with Explainable VQA on GI Imaging
=============================================================================

USAGE:
    python preprocessing.py --mode inspect   # inspect dataset samples
    python preprocessing.py --mode process   # run full preprocessing
    python preprocessing.py --mode verify    # verify outputs
"""

import os, re, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizerFast

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Image
    img_size        = 224,
    img_mean        = [0.485, 0.456, 0.406],   # ImageNet stats
    img_std         = [0.229, 0.224, 0.225],

    # Augmentation probabilities (training only)
    aug_hflip       = 0.5,
    aug_rotate      = 0.3,
    aug_rotate_deg  = 15,
    aug_crop        = 0.4,
    aug_crop_scale  = (0.90, 1.00),
    aug_color       = 0.5,
    aug_blur        = 0.3,
    aug_noise       = 0.2,
    aug_clahe       = True,      # contrast enhancement for dark endoscopy frames

    # Text
    max_text_len    = 128,
    tokenizer_name  = "distilbert-base-uncased",

    # Quality thresholds
    low_brightness  = 0.30,      # mean intensity below this → apply gamma brighten
    high_brightness = 0.70,      # mean intensity above this → apply gamma darken
    gamma_bright    = 0.80,
    gamma_dark      = 1.20,

    # Paths
    data_dir        = "./data",
    output_dir      = "./preprocessed",
    log_dir         = "./logs",
    seed            = 42,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
)

# GI disease labels (23 classes — TreeNet Stage 1 output)
DISEASE_LABELS = [
    "polyp-pedunculated", "polyp-sessile", "polyp-hyperplastic",
    "esophagitis", "gastritis", "ulcerative-colitis", "crohns-disease",
    "barretts-esophagus", "gastric-ulcer", "duodenal-ulcer",
    "erosion", "hemorrhoid", "diverticulum",
    "normal-cecum", "normal-pylorus", "normal-z-line",
    "ileocecal-valve", "retroflex-rectum", "retroflex-stomach",
    "dyed-lifted-polyp", "dyed-resection-margins",
    "foreign-body", "instrument"
]
assert len(DISEASE_LABELS) == 23, "Must have exactly 23 disease labels"

# Medical abbreviation map for text normalisation
MED_ABBREV = {
    r'\bgi\b'        : 'gastrointestinal',
    r'\bpolyp\b'     : 'polyp',
    r'\bz-line\b'    : 'z-line',
    r'\bsce\b'       : 'squamocolumnar epithelium',
    r'\bimo\b'       : 'intestinal metaplasia',
    r'\bmm\b'        : 'millimeter',
    r'\bcm\b'        : 'centimeter',
    r'\bclip\b'      : 'clip',
    r'\bendo\b'      : 'endoscopy',
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).
    Enhances local contrast in dark GI endoscopy frames.
    clip_limit=2.0, tile_grid=(8,8) as specified in thesis methodology.
    """
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid  = tile_grid

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            import cv2
            img_np  = np.array(img)
            lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe   = cv2.createCLAHE(clipLimit=self.clip_limit,
                                      tileGridSize=self.tile_grid)
            l_eq    = clahe.apply(l)
            lab_eq  = cv2.merge([l_eq, a, b])
            img_eq  = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_eq)
        except ImportError:
            # cv2 not available — return original (no crash)
            return img


class AdaptiveGammaCorrection:
    """
    Gamma correction based on mean image brightness.
    Brightens dark frames (mean < 0.30), darkens overexposed ones (mean > 0.70).
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        arr  = np.array(img).astype(np.float32) / 255.0
        mean = arr.mean()
        if mean < CFG["low_brightness"]:
            gamma = CFG["gamma_bright"]   # < 1 → brighter
        elif mean > CFG["high_brightness"]:
            gamma = CFG["gamma_dark"]     # > 1 → darker
        else:
            return img
        corrected = np.clip(arr ** gamma, 0, 1)
        return Image.fromarray((corrected * 255).astype(np.uint8))


class GaussianNoiseTransform:
    """Additive Gaussian noise for training augmentation (σ=0.01)."""
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)


def build_image_transform(split: str) -> T.Compose:
    """
    Build torchvision transform pipeline.

    TRAIN : resize → quality enhancement → augmentations → tensor → normalise → noise
    VAL/TEST: resize → quality enhancement → tensor → normalise

    Implements thesis Section 3.3.1 Image Preprocessing specification exactly.
    """
    assert split in ("train", "val", "test")

    base = [
        T.Resize((CFG["img_size"], CFG["img_size"])),   # bilinear interpolation
        AdaptiveGammaCorrection(),                        # brightness correction
    ]
    if CFG["aug_clahe"]:
        base.append(CLAHETransform(clip_limit=2.0, tile_grid=(8, 8)))

    if split == "train":
        augment = [
            T.RandomHorizontalFlip(p=CFG["aug_hflip"]),
            T.RandomApply([T.RandomRotation(CFG["aug_rotate_deg"])],
                          p=CFG["aug_rotate"]),
            T.RandomApply([T.RandomResizedCrop(CFG["img_size"],
                                               scale=CFG["aug_crop_scale"])],
                          p=CFG["aug_crop"]),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2,
                                         saturation=0.15, hue=0.05)],
                          p=CFG["aug_color"]),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))],
                          p=CFG["aug_blur"]),
        ]
        post = [
            T.ToTensor(),
            T.Normalize(mean=CFG["img_mean"], std=CFG["img_std"]),
            GaussianNoiseTransform(sigma=0.01),
        ]
        return T.Compose(base + augment + post)
    else:
        post = [
            T.ToTensor(),
            T.Normalize(mean=CFG["img_mean"], std=CFG["img_std"]),
        ]
        return T.Compose(base + post)


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation for visualisation."""
    mean = torch.tensor(CFG["img_mean"]).view(3, 1, 1)
    std  = torch.tensor(CFG["img_std"]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Text cleaning pipeline (thesis Section 3.3.2):
    1. Lowercase (preserve medical acronyms handled separately)
    2. Remove non-ASCII except ±, ×, degree symbol
    3. Expand medical abbreviations
    4. Collapse whitespace / standardise punctuation
    5. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — remove non-ASCII except medical symbols
    text = re.sub(r'[^\x00-\x7F±×°]', '', text)

    # Step 3 — expand medical abbreviations
    for pattern, replacement in MED_ABBREV.items():
        text = re.sub(pattern, replacement, text)

    # Step 4 — standardise punctuation & whitespace
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'\s([?.!,;:])', r'\1', text) # remove space before punct
    text = re.sub(r'([?.!,;:]){2,}', r'\1', text) # deduplicate punct

    # Step 5 — strip
    return text.strip()


class TextPreprocessor:
    """
    Full text preprocessing pipeline: clean → tokenise → encode.
    Uses DistilBERT WordPiece tokenizer (vocab=30,522, max_len=128).
    """
    def __init__(self, tokenizer_name: str = CFG["tokenizer_name"],
                 max_length: int = CFG["max_text_len"]):
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        print(f"✅  TextPreprocessor ready  |  vocab={self.tokenizer.vocab_size:,}"
              f"  max_len={max_length}")

    def preprocess(self, text: str) -> dict:
        """
        Returns dict with keys:
            cleaned_text   : str
            input_ids      : torch.Tensor  (max_length,)
            attention_mask : torch.Tensor  (max_length,)
            token_count    : int   (non-padding tokens)
        """
        cleaned = clean_text(text)
        enc = self.tokenizer(
            cleaned,
            max_length    = self.max_length,
            padding       = "max_length",
            truncation    = True,
            return_tensors= "pt",
        )
        n_tokens = int(enc["attention_mask"].sum().item())
        return {
            "cleaned_text"  : cleaned,
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_count"   : n_tokens,
        }

    def batch_preprocess(self, texts: list[str]) -> dict:
        """Efficient batch processing."""
        cleaned = [clean_text(t) for t in texts]
        enc = self.tokenizer(
            cleaned,
            max_length    = self.max_length,
            padding       = "max_length",
            truncation    = True,
            return_tensors= "pt",
        )
        return {
            "cleaned_texts" : cleaned,
            "input_ids"     : enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_counts"  : enc["attention_mask"].sum(dim=1).tolist(),
        }

    def analyse(self, texts: list[str]) -> dict:
        """Compute token length statistics over a corpus."""
        cleaned = [clean_text(t) for t in texts]
        lengths = []
        for t in tqdm(cleaned, desc="Analysing text lengths", leave=False):
            ids = self.tokenizer(t, truncation=False)["input_ids"]
            lengths.append(len(ids))
        arr = np.array(lengths)
        return {
            "mean"    : float(arr.mean()),
            "median"  : float(np.median(arr)),
            "std"     : float(arr.std()),
            "min"     : int(arr.min()),
            "max"     : int(arr.max()),
            "p95"     : float(np.percentile(arr, 95)),
            "p99"     : float(np.percentile(arr, 99)),
            "truncated_pct": float((arr > CFG["max_text_len"]).mean() * 100),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COMBINED DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class KvasirVQADataset(Dataset):
    """
    Unified dataset class for Kvasir-VQA-x1.
    Applies image and text preprocessing in one place.
    Used by all pipeline stages (Stage 1 → Stage 4).
    """
    def __init__(self, hf_split, split: str, text_preprocessor: TextPreprocessor):
        self.data      = hf_split
        self.split     = split
        self.img_tfm   = build_image_transform(split)
        self.text_prep = text_preprocessor
        print(f"   KvasirVQADataset [{split}] : {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # ── Image ──────────────────────────────────────
        img_pil = ex["image"].convert("RGB")   # PIL after cast_column
        img_tensor = self.img_tfm(img_pil)        # (3, 224, 224) float32

        # ── Text ───────────────────────────────────────
        q_enc = self.text_prep.preprocess(ex["question"])
        a_enc = self.text_prep.preprocess(ex["answer"])

        return {
            # Image
            "image"           : img_tensor,
            # Question
            "q_input_ids"     : q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "q_token_count"   : q_enc["token_count"],
            "question_raw"    : ex["question"],
            "question_clean"  : q_enc["cleaned_text"],
            # Answer
            "a_input_ids"     : a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"],
            "answer_raw"      : ex["answer"],
            "answer_clean"    : a_enc["cleaned_text"],
            # Metadata
            "img_id"          : ex.get("img_id", ""),
            "complexity"      : ex.get("complexity", 0),
        }


def build_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """Load Kvasir-VQA-x1 and return train/val/test DataLoaders."""
    from datasets import load_dataset
    print("📦  Loading Kvasir-VQA-x1 …")
    from datasets import Image as HFImage
    raw  = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CFG["data_dir"])
    raw  = raw.cast_column("image", HFImage())   # decode URL/path → PIL object
    text = TextPreprocessor()

    from sklearn.model_selection import train_test_split as sk_split
    import pandas as pd

    # Build train/val split (80/20 of official train)
    full_train_ds = raw["train"]
    indices       = list(range(len(full_train_ds)))
    tr_idx, va_idx = sk_split(indices, test_size=0.20, random_state=CFG["seed"])

    train_ds  = KvasirVQADataset(full_train_ds.select(tr_idx),  "train", text)
    val_ds    = KvasirVQADataset(full_train_ds.select(va_idx),  "val",   text)
    test_ds   = KvasirVQADataset(raw["test"],                    "test",  text)

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualise_augmentations(n_images: int = 4):
    """Show original vs preprocessed images side by side."""
    from datasets import load_dataset
    from datasets import Image as HFImage
    raw   = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CFG["data_dir"])
    raw   = raw.cast_column("image", HFImage())   # decode URL/path → PIL object
    train = build_image_transform("train")
    val   = build_image_transform("val")

    os.makedirs(CFG["log_dir"], exist_ok=True)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))
    fig.suptitle("Preprocessing Visualisation\n"
                 "Original | Val (normalised) | Train (augmented)",
                 fontsize=13, fontweight="bold")

    for i in range(n_images):
        print(f"   Processing image {i+1}/{n_images} …")
        ex  = raw["train"][i * 1000]
        img = ex["image"].convert("RGB")   # PIL after cast_column
        img_resized = img.resize((224, 224))
        img_val   = denormalise(val(img))
        img_train = denormalise(train(img))

        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"Original\n{img.size[0]}×{img.size[1]}", fontsize=9)
        axes[i, 1].imshow(img_val)
        axes[i, 1].set_title("Val transform\n(224×224, normalised)", fontsize=9)
        axes[i, 2].imshow(img_train)
        axes[i, 2].set_title("Train transform\n(augmented)", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
            ax.set_xlabel(ex["question"][:60], fontsize=7)

    plt.tight_layout()
    path = f"{CFG['log_dir']}/preprocessing_visualisation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved: {path}")


def visualise_text_stats():
    """Plot token length distribution for questions and answers."""
    from datasets import load_dataset
    from datasets import Image as HFImage
    raw  = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CFG["data_dir"])
    raw  = raw.cast_column("image", HFImage())   # decode URL/path → PIL object
    text = TextPreprocessor()

    print("   Sampling 500 examples for token length analysis …")
    sample = raw["train"].select(range(500))
    qs  = [ex["question"] for ex in sample]
    ans = [ex["answer"]   for ex in sample]

    # Batch tokenize — much faster than one-by-one loop
    tok = text.tokenizer
    q_enc = tok([clean_text(q) for q in qs],  truncation=False)
    a_enc = tok([clean_text(a) for a in ans], truncation=False)
    q_lens = [len(ids) for ids in q_enc["input_ids"]]
    a_lens = [len(ids) for ids in a_enc["input_ids"]]

    import numpy as _np
    q_stats = {"mean": float(_np.mean(q_lens)), "median": float(_np.median(q_lens)),
               "max": int(max(q_lens)), "min": int(min(q_lens)),
               "p95": float(_np.percentile(q_lens, 95)),
               "truncated_pct": float((_np.array(q_lens) > CFG["max_text_len"]).mean()*100)}
    a_stats = {"mean": float(_np.mean(a_lens)), "median": float(_np.median(a_lens)),
               "max": int(max(a_lens)), "min": int(min(a_lens)),
               "p95": float(_np.percentile(a_lens, 95)),
               "truncated_pct": float((_np.array(a_lens) > CFG["max_text_len"]).mean()*100)}
    print(f"\n📊  Question token stats : {q_stats}")
    print(f"📊  Answer token stats   : {a_stats}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(q_lens, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(q_lens),   color="red",    linestyle="--",
                    label=f"Mean={np.mean(q_lens):.1f}")
    axes[0].axvline(CFG["max_text_len"], color="orange", linestyle=":",
                    label=f"Max len={CFG['max_text_len']}")
    axes[0].set_title("Question Token Length Distribution", fontweight="bold")
    axes[0].set_xlabel("Token count"); axes[0].set_ylabel("Frequency")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(a_lens, bins=60, color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(a_lens),   color="red",    linestyle="--",
                    label=f"Mean={np.mean(a_lens):.1f}")
    axes[1].axvline(CFG["max_text_len"], color="steelblue", linestyle=":",
                    label=f"Max len={CFG['max_text_len']}")
    axes[1].set_title("Answer Token Length Distribution", fontweight="bold")
    axes[1].set_xlabel("Token count"); axes[1].set_ylabel("Frequency")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{CFG['log_dir']}/text_token_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved: {path}")

    with open(f"{CFG['log_dir']}/text_stats.json", "w") as f:
        json.dump({"question": q_stats, "answer": a_stats}, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect","process","verify"],
                        default="inspect")
    args = parser.parse_args()

    os.makedirs(CFG["log_dir"], exist_ok=True)

    if args.mode == "inspect":
        print("\n🔍  Inspecting dataset and showing preprocessing …\n")
        visualise_augmentations(n_images=4)
        visualise_text_stats()
        print("\n✅  Inspect complete. Check ./logs/ for output images.")

    elif args.mode == "process":
        print("\n⚙️   Running full preprocessing pipeline …\n")
        train_loader, val_loader, test_loader, text_prep = build_dataloaders(
            batch_size=32, num_workers=4)
        # Verify one batch
        batch = next(iter(train_loader))
        print(f"   Image tensor shape  : {batch['image'].shape}")
        print(f"   Q input_ids shape   : {batch['q_input_ids'].shape}")
        print(f"   A input_ids shape   : {batch['a_input_ids'].shape}")
        print(f"   Sample question     : {batch['question_raw'][0]}")
        print(f"   Sample cleaned Q    : {batch['question_clean'][0]}")
        print(f"   Sample answer       : {batch['answer_raw'][0]}")
        print(f"   Q token count       : {batch['q_token_count'][0].item()}")
        print("\n✅  Preprocessing complete. DataLoaders ready for Stage 1.")

    elif args.mode == "verify":
        print("\n✅  Verifying text cleaning on sample inputs …\n")
        test_cases = [
            "Is there a GI polyp visible in the image?",
            "What  is  the   COLOR of the lesion??",
            "Are there ±2 erosions present, yes or no?",
            "How many POLYPS can be identified in this ENDO image?",
            "Does the Z-LINE appear normal or abnormal?",
        ]
        text_prep = TextPreprocessor()
        print(f"{'Original':<55} {'Cleaned'}")
        print("-" * 110)
        for t in test_cases:
            print(f"{t:<55} {clean_text(t)}")
