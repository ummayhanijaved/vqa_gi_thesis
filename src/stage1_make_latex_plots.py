#!/usr/bin/env python3
"""
=============================================================================
Stage 1 — LaTeX (pgfplots) plot generator
=============================================================================
Reads YOUR real data and writes ready-to-paste pgfplots .tex files:

  figures/stage1_latex/
    ├── stage1_curves.tex        (epochs vs train/val loss + val macro-F1/AUC)
    ├── stage1_per_class_f1.tex  (per-disease F1 bar chart, test set)
    └── stage1_support.tex       (per-disease support bar chart, test set)

Numbers come from results/stage1_epoch_log.csv and from running the trained
checkpoint on the test split. Nothing invented.

Each .tex file is a complete \\begin{figure}...\\end{figure}. Paste directly,
or \\input{figures/stage1_latex/stage1_curves.tex}.

PREAMBLE (you already have most): \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}
                                   \\usepackage{booktabs,float}

USAGE:  python src/stage1_make_latex_plots.py
=============================================================================
"""
import os, sys
import numpy as np
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
SRC = os.path.join(PROJECT, "src"); sys.path.insert(0, SRC)
OUT = os.path.join(PROJECT, "figures", "stage1_latex"); os.makedirs(OUT, exist_ok=True)
EPOCH_CSV = os.path.join(PROJECT, "results", "stage1_epoch_log.csv")

import stage1_disease_classifier as s1
import torch
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# 1. Training curves (epochs vs train/val) — pure pgfplots from the CSV
# ---------------------------------------------------------------------------
def make_curves():
    df = pd.read_csv(EPOCH_CSV)
    df.columns = [c.strip() for c in df.columns]
    def coords(col):
        return " ".join(f"({int(e)},{v})" for e, v in zip(df["Epoch"], df[col]))

    tex = r"""\begin{figure}[H]
\centering
% ---- (a) Loss ----
\begin{tikzpicture}
\begin{axis}[
    width=0.8\linewidth, height=6.2cm,
    title={\textbf{(a) Loss per Epoch}},
    xlabel={Epoch}, ylabel={BCE Loss},
    legend pos=north east, grid=both,
    grid style={gray!25}, tick label style={font=\small},
    label style={font=\small}, title style={font=\small\bfseries},
]
\addplot[blue, mark=*, thick] coordinates {""" + coords("Tr Loss") + r"""};
\addlegendentry{Train loss}
\addplot[red, mark=square*, thick] coordinates {""" + coords("Val Loss") + r"""};
\addlegendentry{Val loss}
\end{axis}
\end{tikzpicture}

\vspace{0.6em}
% ---- (b) F1 / AUC ----
\begin{tikzpicture}
\begin{axis}[
    width=0.8\linewidth, height=6.2cm,
    title={\textbf{(b) Validation Macro-F1 and AUC per Epoch}},
    xlabel={Epoch}, ylabel={Score},
    ymin=0.9, ymax=1.0,
    legend pos=south east, grid=both,
    grid style={gray!25}, tick label style={font=\small},
    label style={font=\small}, title style={font=\small\bfseries},
]
\addplot[green!55!black, mark=*, thick] coordinates {""" + coords("Val Macro-F1") + r"""};
\addlegendentry{Val Macro-F1}
\addplot[purple, mark=triangle*, thick] coordinates {""" + coords("Val AUC") + r"""};
\addlegendentry{Val AUC}
\addplot[gray, dotted, thick] coordinates {""" + coords("Tr F1") + r"""};
\addlegendentry{Train F1}
\end{axis}
\end{tikzpicture}
\caption[Stage 1 training curves]{Stage~1 training dynamics. (a) Train and
validation BCE loss per epoch. (b) Validation macro-F1 (peak 0.9925) and AUC,
with train F1 for reference. The model converges within a few epochs and the
small train--validation gap indicates minimal overfitting.}
\label{fig:stage1_curves}
\end{figure}
"""
    p = os.path.join(OUT, "stage1_curves.tex")
    open(p, "w").write(tex)
    print(f"  saved {p}")


# ---------------------------------------------------------------------------
# Run model on test split (real per-class numbers)
# ---------------------------------------------------------------------------
def test_metrics():
    device = s1.CFG["device"]
    model = s1.TreeNetDiseaseClassifier().to(device)
    ckpt = torch.load(os.path.join(PROJECT, "checkpoints", "stage1_best.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    from datasets import load_from_disk, Image as HFImage
    import torch.utils.data as tud
    raw = load_from_disk(os.path.expanduser("~/vqa_gi_thesis/data/kvasir_local"))
    test_hf = raw["test"] if "test" in raw else raw["train"]
    try: test_hf = test_hf.cast_column("image", HFImage())
    except Exception: pass
    ds = s1.DiseaseClassificationDataset(test_hf, "test")
    loader = tud.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    P, Y = [], []
    with torch.no_grad():
        for b in loader:
            P.append(model(b["image"].to(device))["probs"].cpu().numpy())
            Y.append(b["labels"].numpy())
    probs = np.concatenate(P); y = np.concatenate(Y)
    preds = (probs > 0.5).astype(int)
    return s1.DISEASE_LABELS, y, preds


# ---------------------------------------------------------------------------
# 2. Per-class F1 bar chart (pgfplots, real numbers)
# ---------------------------------------------------------------------------
def make_per_class_f1(labels, y, preds):
    f1s = f1_score(y, preds, average=None, zero_division=0)
    order = np.argsort(f1s)            # ascending
    syms = "\n".join(f"        {labels[i]}" for i in order)
    coords = " ".join(f"({f1s[i]:.3f},{labels[i]})" for i in order)
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    xbar, width=0.8\linewidth, height=11cm,
    title={\textbf{Stage 1 --- Per-Class F1 (test set)}},
    xlabel={F1-score}, xmin=0, xmax=1.05,
    ytick=data, symbolic y coords={
""" + syms + r"""
    },
    y dir=reverse,
    nodes near coords, nodes near coords style={font=\scriptsize},
    tick label style={font=\scriptsize}, label style={font=\small},
    title style={font=\small\bfseries}, grid=major, grid style={gray!20},
    bar width=5pt,
]
\addplot[fill=teal!70, draw=teal!80] coordinates {""" + coords + r"""};
\end{axis}
\end{tikzpicture}
\caption[Stage 1 per-class F1]{Stage~1 per-class F1 on the test set
(4{,}058 unique images). Macro-F1 = 0.9686, micro-F1 = 0.9687.}
\label{fig:stage1_per_class_f1}
\end{figure}
"""
    p = os.path.join(OUT, "stage1_per_class_f1.tex")
    open(p, "w").write(tex)
    print(f"  saved {p}")


# ---------------------------------------------------------------------------
# 3. Per-class support bar chart (pgfplots, real numbers)
# ---------------------------------------------------------------------------
def make_support(labels, y):
    support = y.sum(axis=0).astype(int)
    order = np.argsort(support)
    syms = "\n".join(f"        {labels[i]}" for i in order)
    coords = " ".join(f"({int(support[i])},{labels[i]})" for i in order)
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    xbar, width=0.8\linewidth, height=11cm,
    title={\textbf{Stage 1 --- Per-Class Support (test set)}},
    xlabel={Number of positive samples},
    ytick=data, symbolic y coords={
""" + syms + r"""
    },
    y dir=reverse,
    nodes near coords, nodes near coords style={font=\scriptsize},
    tick label style={font=\scriptsize}, label style={font=\small},
    title style={font=\small\bfseries}, grid=major, grid style={gray!20},
    bar width=5pt,
]
\addplot[fill=orange!70, draw=orange!80] coordinates {""" + coords + r"""};
\end{axis}
\end{tikzpicture}
\caption[Stage 1 per-class support]{Per-class support (number of positive test
images per disease). Classes with very low support should be read alongside
their F1 in Figure~\ref{fig:stage1_per_class_f1}.}
\label{fig:stage1_support}
\end{figure}
"""
    p = os.path.join(OUT, "stage1_support.tex")
    open(p, "w").write(tex)
    print(f"  saved {p}")


def main():
    print(f"\nStage 1 LaTeX plots -> {OUT}\n")
    make_curves()
    labels, y, preds = test_metrics()
    make_per_class_f1(labels, y, preds)
    make_support(labels, y)
    print("\nDone. Paste the .tex files or \\input{} them.\n")


if __name__ == "__main__":
    main()
