<div align="center">

# 🔬 Advancing Medical AI with Explainable VQA on GI Imaging

### A modular, explainable Visual Question Answering pipeline for gastrointestinal endoscopy

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/🤗_Transformers-DistilBERT_·_T5-FFD21E?style=for-the-badge" alt="Transformers"/>
<img src="https://img.shields.io/badge/Ultralytics-YOLOv8-00CFE8?style=for-the-badge" alt="YOLOv8"/>
</p>

<p>
<img src="https://img.shields.io/badge/Dataset-Kvasir--VQA--x1-7B61FF?style=flat-square" alt="Dataset"/>
<img src="https://img.shields.io/badge/Stages-0→7-2C7BE5?style=flat-square" alt="Stages"/>
<img src="https://img.shields.io/badge/BERTScore-93.99%25-2CA02C?style=flat-square" alt="BERTScore"/>
<img src="https://img.shields.io/badge/Explainability-Grad--CAM_%2B_Textual-D6336C?style=flat-square" alt="Explainability"/>
<img src="https://img.shields.io/badge/Status-Thesis_Complete-success?style=flat-square" alt="Status"/>
</p>

**Ummay Hani Javed** (24i-8211) · National University of Computer and Emerging Sciences · 2026

[Repository](https://github.com/ummayhanijaved/vqa_gi_thesis) ·
[Dataset](https://arxiv.org/abs/2506.09958) ·
[Architecture](#-pipeline-architecture) ·
[Results](#-results) ·
[Quick Start](#-quick-start)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training from Scratch](#-training-from-scratch)
- [Explainability](#-explainability)
- [Citation](#-citation)

---

## 🩺 Overview

This repository contains the full implementation of a **modular, explainable
Visual Question Answering (VQA) pipeline** for gastrointestinal (GI) endoscopy
imaging, trained and evaluated on the **Kvasir-VQA-x1** dataset
(143,594 training and 15,955 test QA pairs).

> Rather than relying on a single monolithic model, the task is decomposed into
> **five specialised stages** followed by a dedicated **explainability layer**.
> The system answers clinical questions about GI endoscopy images while providing
> **both visual (Grad-CAM) and textual** explanations — addressing the clinical
> need to understand *why* an answer was produced and *where* in the image its
> evidence lies.

**Highlights**

- 🧩 **Modular** — each stage is independently trainable, inspectable, and swappable
- 🎯 **Specialised routing** — six question types, each handled by the best-suited model
- 🔍 **Dual explainability** — Grad-CAM heatmaps *and* structured textual rationales
- 📊 **Rigorously evaluated** — ROUGE, METEOR, BLEU, CHRF++, BERTScore + per-route breakdown
- ⚖️ **Honestly benchmarked** — against a fine-tuned PaliGemma-3B baseline

---

## 🏗 Pipeline Architecture

```
                    ┌──────────────────────────────────────────────┐
   Image + Question │  Stage 0 · Input & Preprocessing             │
        ──────────► │  transforms · normalisation · tokenisation   │
                    └───────────────────────┬──────────────────────┘
                                             ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  CORE VQA PIPELINE                                                        │
   │                                                                           │
   │  Stage 1 · Disease Classification   ResNet50 (frozen)        99.25% F1    │
   │  Stage 2 · Question Categorisation  DistilBERT               93.01%       │
   │  Stage 3 · Multimodal Fusion        CrossAttn + DiseaseGate  92.50%       │
   │  Stage 4 · Answer Generation        4× DistilBERT + 2× YOLO  per-route    │
   │  Stage 5 · Verbalisation            T5-small (seq2seq)       61.21% R-1   │
   └───────────────────────────────────────┬───────────────────────────────────┘
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │  EXPLAINABILITY LAYER                         │
                    │   Stage 6 · Textual   (Medical Response + NLG)│
                    │   Stage 7 · Visual    (Grad-CAM, answer-driven)│
                    └───────────────────────┬──────────────────────┘
                                            ▼
              Answer  +  Textual rationale  +  Grad-CAM heatmap
```

**Stage 4 routing** — each question is dispatched to a specialised model:
four fine-tuned **DistilBERT** classifiers (Yes/No, Single-Choice, Multi-Choice,
Colour) and two fine-tuned **YOLOv8** networks (YOLO-Seg → Location,
YOLO-Det → Count).

---

## 📊 Results

### Foundational stages

| Stage | Model | Headline Metric |
|:-----:|-------|:---------------:|
| **1** | ResNet50 — disease classifier | **99.25%** F1 |
| **2** | DistilBERT — question router | **93.01%** accuracy |
| **3** | CrossAttn + DiseaseGate + FusionMLP | **92.50%** accuracy |

### Stage 4 — per-route answer generation

| Route | Type | Model | Test Performance |
|:-----:|------|-------|------------------|
| 0 | Yes/No | DistilBERT | **89.02%** |
| 1 | Single-Choice | DistilBERT (CW) | 11.92% strict · **36.70%** fuzzy |
| 2 | Multi-Choice | DistilBERT | **6.31%** sample-F1 · (99.28% Hamming) |
| 3 | Colour | DistilBERT | **79.44%** |
| 4 | Location | YOLO-Seg | **54.20%** |
| 5 | Count | YOLO-Det | **69.80%** |

> ℹ️ Route 2's headline metric is **sample-averaged F1**; the 99.28% Hamming
> accuracy is sparsity-inflated and is *not* used as the headline. Routes 4/5 use
> task-appropriate protocols (region-keyword overlap and tolerance-based count
> matching).

### Stage 5 — natural-language generation *(919-sample test evaluation)*

| Metric | Score | | Metric | Score |
|--------|:-----:|-|--------|:-----:|
| ROUGE-1 | **61.21%** | | BLEU (sacreBLEU) | 35.07% |
| ROUGE-2 | 41.81% | | CHRF++ | 54.33% |
| ROUGE-L | 58.15% | | **BERTScore (F1)** | **93.99%** |
| METEOR | 55.87% | | Well-formedness | 92.59% |
| Soft Match | 62.46% | | Exact Match | 10.55% |

> 🎯 **BERTScore of 93.99%**, sustained above 92% on *every* route, shows the
> generated sentences are semantically faithful to the references even where
> surface n-gram overlap is lower.

<details>
<summary><b>📈 Baseline comparison</b> (click to expand)</summary>

<br>

| Model | Accuracy |
|-------|----------|
| Reference paper baseline | 79.23% |
| **PaliGemma-3B** (ours, fine-tuned LoRA baseline) | **83.12%** train / **83.94%** val |
| Proposed pipeline | decomposed + explainable *(see per-stage results)* |

A single large vision–language model can attain a higher single-number accuracy;
the value of the proposed pipeline lies in its **modularity**, dedicated
**spatial-localisation/counting routes**, and **multimodal explainability** —
capabilities a monolithic model does not provide.

</details>

---

## 🗂 Project Structure

```
vqa_gi_thesis/
├── src/                                   # Core pipeline modules
│   ├── preprocessing.py                   # Image transforms + DistilBERT tokeniser
│   ├── stage1_disease_classifier.py       # ResNet50 disease classifier (23 classes)
│   ├── stage2_question_categorizer.py     # DistilBERT question routing (6 routes)
│   ├── stage3_multimodal_fusion.py        # CrossAttn + DiseaseGate + FusionMLP
│   ├── stage4_revised.py                  # Stage 4 Phase 2 (4 DistilBERT + 2 YOLO)
│   ├── stage5_verbalizer.py               # T5-small verbaliser (training)
│   ├── stage5_pipeline_test.py            # Full end-to-end pipeline runner
│   ├── stage5_evaluate_enhanced.py        # NLG evaluation (ROUGE/METEOR/BLEU/…)
│   ├── stage6_explainability.py           # Textual Medical Response + NLG metrics
│   ├── stage7_gradcam.py                  # Grad-CAM visual explanations
│   ├── make_stage67_results.py            # Builds Stage 6/7 figures + LaTeX from CSVs
│   └── stage0_eda_plots.py                # Dataset EDA / preprocessing figures
│
├── plot_stage4_curves.py                  # Stage 4 training/val curve plots
├── plot_stage5_curves.py                  # Stage 5 training curve + NLG plots
│
├── analysis/                              # Per-stage analysis scripts
├── checkpoints/                           # Trained model weights
│   ├── stage1_best.pt                     #   ResNet50        (best_f1 = 0.9925)
│   ├── best_model/                        #   DistilBERT Stage 2 checkpoint
│   ├── stage3_best.pt                     #   Fusion model    (val_acc = 0.9250)
│   ├── stage4_revised/                    #   Stage 4 Phase 2 checkpoints (+ YOLO)
│   └── stage5_verbalizer/                 #   T5 verbaliser   (epoch 8, val_loss 0.5882)
│
├── data/
│   ├── kvasir_local/                      # HuggingFace Arrow dataset
│   ├── kvasir_raw/images/                 # Raw GI endoscopy images
│   └── yolo_dataset/                      # YOLO pseudo-annotations (routes 4/5)
│
├── cache/                                 # Pre-computed Stage 3/4 features
├── figures/                               # Thesis figures (per stage)
├── logs/                                  # Training logs + evaluation outputs
│   ├── stage4_plots/  stage5_plots/  stage67_results/
│   ├── stage6_explainability/             #   medical_responses.csv, metrics
│   └── stage7_gradcam/                    #   Grad-CAM outputs
├── results/                               # CSV result tables
├── configs/pipeline_config.yaml
├── demo/demo.py
├── requirements.txt
└── README.md
```

---

## 🗃 Dataset

**[Kvasir-VQA-x1](https://arxiv.org/abs/2506.09958)** (SimulaMet, 2025)

| | |
|---|---|
| **Training QA pairs** | 143,594 |
| **Test QA pairs** | 15,955 |
| **Question types** | yes/no · single-choice · multiple-choice · colour · location · count |
| **Disease / finding classes** | 23 |

---

## ⚙️ Installation

```bash
git clone https://github.com/ummayhanijaved/vqa_gi_thesis.git
cd vqa_gi_thesis

python -m venv hani_env
source hani_env/bin/activate        # Linux/Mac
# hani_env\Scripts\activate         # Windows

pip install -r requirements.txt
```

> **Environment:** Python 3.12 · PyTorch 2.x (CUDA) · Ultralytics (YOLOv8) ·
> HuggingFace Transformers

---

## 🚀 Quick Start

**Run the full pipeline on one image**
```bash
cd src
python stage5_pipeline_test.py            # end-to-end pipeline (Stages 1–5)
```

**Textual explainability + NLG metrics (Stage 6)**
```bash
python stage6_explainability.py --n_samples 1000
```

**Grad-CAM visual explanations (Stage 7)**
```bash
python stage7_gradcam.py --mode demo
# CPU fallback if GPU memory is tight:
# CUDA_VISIBLE_DEVICES="" python stage7_gradcam.py --mode demo
```

**Build Stage 6/7 results (figures + LaTeX) from existing CSVs — no model load**
```bash
python make_stage67_results.py
```

**Produce training/evaluation plots**
```bash
python plot_stage4_curves.py
python plot_stage5_curves.py
```

---

## 🧪 Training from Scratch

```bash
cd src
python stage1_disease_classifier.py        # Stage 1 · Disease classifier
python stage2_question_categorizer.py      # Stage 2 · Question router
python stage3_multimodal_fusion.py         # Stage 3 · Multimodal fusion
python stage4_revised.py                   # Stage 4 · Answer generation (Phase 2)
python stage5_verbalizer.py                # Stage 5 · T5 verbaliser
```

---

## 🔍 Explainability

Every prediction ships with **two complementary explanations**:

| Layer | What it provides | How it's evaluated |
|-------|------------------|--------------------|
| **Stage 6 · Textual** | Structured Medical Response (answer + rationale) | ROUGE · METEOR · BLEU · CHRF++ · BERTScore |
| **Stage 7 · Visual** | Answer-driven Grad-CAM heatmap on the Stage 1 ResNet50 | original / heatmap / overlay panels |

**Example trace**

```
Question       : Is there a polyp visible?
Stage 4 answer : no
Stage 5 output : No polyps are visible in the image.
Stage 6 expl.  : Based on the endoscopic image analysis, no polyps are visible…
Stage 7 output : Grad-CAM heatmap → logs/stage7_gradcam/
```

---

## 📚 Citation

```bibtex
@thesis{javed2026vqa,
  title  = {Advancing Medical AI with Explainable VQA on GI Imaging},
  author = {Javed, Ummay Hani},
  year   = {2026},
  school = {National University of Computer and Emerging Sciences},
  type   = {Master's Thesis}
}
```

---

## 🙏 Acknowledgements

**Dataset:** Kvasir-VQA-x1 (SimulaMet) ·
**Backbones:** ResNet50 (PyTorch), DistilBERT & T5 (HuggingFace Transformers),
YOLOv8 (Ultralytics)

<div align="center">
<sub>Built for the MS thesis · National University of Computer and Emerging Sciences · 2026</sub>
</div>
