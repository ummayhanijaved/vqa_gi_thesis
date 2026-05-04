# Advancing Medical AI with Explainable VQA on GI Imaging

**Author:** Ummay Hani Javed (24i-8211)  
**Thesis Supervisor:** [Supervisor Name]  
**Institution:** [University Name]  
**Year:** 2025

---

## Overview

This repository contains the full implementation of a 4-stage explainable Visual Question Answering (VQA) pipeline for Gastrointestinal (GI) endoscopy imaging, trained and evaluated on the **Kvasir-VQA-x1** dataset (143,594 QA pairs, ~6,500 images).

The pipeline is specifically designed to answer clinical questions about GI endoscopy images while providing full explainability through GradCAM spatial attention maps, disease probability vectors, and structured textual explanations.

---

## Pipeline Architecture

```
Input: GI Endoscopy Image + Clinical Question
              ↓
┌─────────────────────────────────────────────────────┐
│  Stage 1: Disease Classification                    │
│  ResNet50 + TreeNet MLP                             │
│  → disease_vec (23-D)  |  Test Acc: 96.86%         │
├─────────────────────────────────────────────────────┤
│  Stage 2: Question Categorisation                   │
│  DistilBERT fine-tuned                              │
│  → question_type (6 classes)  |  Test Acc: 93.01%  │
├─────────────────────────────────────────────────────┤
│  Stage 3: Multimodal Fusion                         │
│  CrossAttention + DiseaseGate + FusionMLP           │
│  → fused_repr (512-D)  |  Test Acc: 92.33%         │
├─────────────────────────────────────────────────────┤
│  Stage 4: Specialised Answer Generation             │
│  6 Specialised Classification Heads                 │
│  → Final Answer  |  Test Acc: 73.97%               │
└─────────────────────────────────────────────────────┘
              ↓
Output: Answer + GradCAM Heatmap + Disease Context + Explanation
```

---

## Project Structure

```
vqa_gi_thesis/
├── src/                          ← Core pipeline modules
│   ├── preprocessing.py          ← Image CLAHE+Gamma+Aug, DistilBERT tokeniser
│   ├── stage1_disease_classifier.py  ← ResNet50 + TreeNet (23 diseases)
│   ├── stage2_question_categorizer.py ← DistilBERT question routing
│   ├── stage3_multimodal_fusion.py   ← CrossAttn + DiseaseGate + FusionMLP
│   ├── stage4_answer_generation.py   ← 6 specialised answer heads
│   └── explainability.py             ← GradCAM + textual explanations
│
├── analysis/                     ← Analysis and evaluation scripts
│   ├── stage1_analysis.py
│   ├── stage2_analysis.py
│   ├── stage3_analysis.py
│   ├── stage3_extended_analysis.py
│   ├── stage4_analysis.py
│   ├── stage4_extended_analysis.py
│   ├── evaluate_pipeline.py      ← End-to-end evaluation (BLEU, METEOR, etc.)
│   └── baseline_comparison.py    ← Comparison with baselines
│
├── checkpoints/                  ← Trained model weights
│   ├── stage1_best.pt            ← ResNet50 + TreeNet (best_f1=0.9925)
│   ├── best_model/               ← DistilBERT Stage 2 checkpoint
│   ├── stage3_best.pt            ← Fusion model (val_acc=0.9250)
│   └── stage4_best.pt            ← Answer heads (val_acc=0.7411)
│
├── data/                         ← Dataset (symlinked)
│   ├── kvasir_local/             ← HuggingFace Arrow dataset
│   ├── kvasir_raw/images/        ← 6,500 raw JPG images
│   └── stage4_vocab.json         ← Answer vocabulary
│
├── figures/                      ← All thesis figures
│   ├── stage1/                   ← Stage 1 analysis plots
│   ├── stage2/                   ← Stage 2 analysis plots
│   ├── stage3/                   ← Stage 3 analysis plots
│   ├── stage4/                   ← Stage 4 analysis plots
│   ├── evaluation/               ← End-to-end evaluation plots
│   ├── baselines/                ← Baseline comparison plots
│   └── explainability/           ← GradCAM + explanation reports
│
├── results/                      ← All CSV result tables
│   ├── full_pipeline_results.csv
│   ├── evaluation_summary.csv
│   ├── baseline_comparison.csv
│   └── ...
│
├── configs/                      ← Configuration files
│   └── pipeline_config.yaml
│
├── utils/                        ← Shared utilities
│   └── metrics.py
│
├── tests/                        ← Sanity checks
│   └── test_pipeline.py
│
├── demo/                         ← Interactive demo
│   └── demo.py
│
├── README.md                     ← This file
└── requirements.txt              ← Python dependencies
```

---

## Dataset

**Kvasir-VQA-x1** (SimulaMet, published June 2025)
- 143,594 training QA pairs
- 15,955 test QA pairs
- ~6,500 unique GI endoscopy images
- 6 question types: yes/no, single-choice, multiple-choice, color, location, numerical count
- 23 disease/finding categories

Paper: arXiv:2506.09958

---

## Results Summary

| Stage | Model | Test Accuracy | Best Metric |
|-------|-------|--------------|-------------|
| Stage 1 | ResNet50 + TreeNet | 96.86% | Val F1 = 99.25% |
| Stage 2 | DistilBERT | 93.01% | Macro-F1 = 88.64% |
| Stage 3 | CrossAttn + DiseaseGate | 92.33% | Val Acc = 92.50% |
| Stage 4 | 6 Specialised Heads | 73.97% | Val Acc = 74.11% |

**End-to-End Evaluation (15,955 test samples):**
- Routing Accuracy: 92.33%
- Soft Match: 37.31%
- Token-Level F1: 13.21%
- BLEU-1: 0.0905

---

## Installation

```bash
# Clone / navigate to project
cd vqa_gi_thesis

# Create virtual environment
python -m venv hani_env
source hani_env/bin/activate   # Linux/Mac
hani_env\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Run Demo
```bash
python demo/demo.py \
    --image ./data/kvasir_raw/images/YOUR_IMAGE.jpg \
    --question "Is there a polyp visible?"
```

### Run Full Evaluation
```bash
# From project root — add src to path
export PYTHONPATH=$PYTHONPATH:./src

python analysis/evaluate_pipeline.py
```

### Generate Explainability Report
```bash
python src/explainability.py \
    --image ./data/kvasir_raw/images/YOUR_IMAGE.jpg \
    --question "Is there a polyp visible?"
```

### Train from Scratch
```bash
# Stage 1
python src/stage1_disease_classifier.py

# Stage 2
python src/stage2_question_categorizer.py

# Stage 3
python src/stage3_multimodal_fusion.py

# Stage 4
python src/stage4_answer_generation.py --mode build_vocab
python src/stage4_answer_generation.py --mode train
```

---

## Explainability

Each prediction comes with a full explainability report:

```
Question      : Is there a polyp visible?
Answer        : yes
Question type : yes/no   (confidence=99.97%)

Disease Context:
   Polyp (Sessile)      : 94.2%  ← drove the YES answer
   Polyp (Pedunculated) : 87.1%
   Normal Cecum         : 8.3%

Spatial Attention (GradCAM):
   → Heatmap saved to ./figures/explainability/
```

---

## Citation

```bibtex
@thesis{javed2025vqa,
  title     = {Advancing Medical AI with Explainable VQA on GI Imaging},
  author    = {Javed, Ummay Hani},
  year      = {2025},
  school    = {[University Name]},
  type      = {Master's Thesis}
}
```

---

## Acknowledgements

Dataset: Kvasir-VQA-x1 by SimulaMet  
Backbone: ResNet50 (PyTorch), DistilBERT (HuggingFace Transformers)
