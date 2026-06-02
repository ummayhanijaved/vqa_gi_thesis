#!/usr/bin/env python3
"""
Stage 2 — LaTeX (pgfplots) plot generator. Reads real data, writes .tex.
Outputs in figures/stage2_latex/:
  stage2_curves.tex, stage2_confusion.tex, stage2_per_class_prf.tex
PREAMBLE: \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}
          \\usepackage{booktabs,float}
USAGE: python src/stage2_make_latex_plots.py
"""
import os, sys
import numpy as np
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
SRC = os.path.join(PROJECT, "src"); sys.path.insert(0, SRC)
OUT = os.path.join(PROJECT, "figures", "stage2_latex"); os.makedirs(OUT, exist_ok=True)
EPOCH_CSV = os.path.join(PROJECT, "results", "epoch_log.csv")

import stage2_question_categorizer as s2
import torch

SHORT = ["yes/no", "single", "multi", "color", "location", "count"]


def make_curves():
    df = pd.read_csv(EPOCH_CSV); df.columns = [c.strip() for c in df.columns]
    def co(c): return " ".join(f"({int(e)},{v})" for e, v in zip(df["Epoch"], df[c]))
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.8\linewidth, height=6cm,
    title={\textbf{(a) Accuracy per Epoch}}, title style={font=\small\bfseries},
    xlabel={Epoch}, ylabel={Accuracy (\%)}, legend pos=south east,
    grid=both, grid style={gray!25}, tick label style={font=\small},
    label style={font=\small}]
\addplot[blue, mark=*, thick] coordinates {""" + co("Train Acc") + r"""};
\addlegendentry{Train acc}
\addplot[red, mark=square*, thick] coordinates {""" + co("Val Acc") + r"""};
\addlegendentry{Val acc}
\end{axis}\end{tikzpicture}

\vspace{0.6em}
\begin{tikzpicture}
\begin{axis}[width=0.8\linewidth, height=6cm,
    title={\textbf{(b) Loss and Validation Macro-F1}}, title style={font=\small\bfseries},
    xlabel={Epoch}, ylabel={Value}, legend pos=east,
    grid=both, grid style={gray!25}, tick label style={font=\small},
    label style={font=\small}]
\addplot[blue, mark=*, thick] coordinates {""" + co("Train Loss") + r"""};
\addlegendentry{Train loss}
\addplot[red, mark=square*, thick] coordinates {""" + co("Val Loss") + r"""};
\addlegendentry{Val loss}
\addplot[green!55!black, mark=triangle*, thick] coordinates {""" + co("Val Macro-F1") + r"""};
\addlegendentry{Val Macro-F1}
\end{axis}\end{tikzpicture}
\caption[Stage 2 training curves]{Stage~2 question-router training dynamics over
8 epochs. The best checkpoint (epoch~5) reaches 92.77\% validation accuracy and
0.8886 macro-F1.}
\label{fig:stage2_curves}
\end{figure}
"""
    open(os.path.join(OUT, "stage2_curves.tex"), "w").write(tex)
    print("  saved stage2_curves.tex")


def test_predictions():
    from transformers import AutoTokenizer, DistilBertForSequenceClassification
    import torch.utils.data as tud
    device = s2.CFG["device"]
    ckpt = os.path.join(PROJECT, "checkpoints", "best_model")
    print(f"  checkpoint: {ckpt}")
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = DistilBertForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()

    _, _, test_df = s2.load_kvasir_vqa_x1()
    ds = s2.QuestionTypeDataset(test_df, tok, s2.CFG["max_length"])
    loader = tud.DataLoader(ds, batch_size=64, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            preds.append(out.logits.argmax(1).cpu().numpy())
            labels.append(b["labels"].numpy())
    return np.concatenate(labels), np.concatenate(preds)


def make_confusion(y, p):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y, p, labels=list(range(6)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    print(f"  TEST accuracy={accuracy_score(y,p)*100:.2f}%  (n={len(y)})")
    rows = [f"{j} {5-i} {cmn[i,j]:.3f}" for i in range(6) for j in range(6)]
    data = "\n".join(rows)
    xt = ",".join(SHORT); yt = ",".join(reversed(SHORT))
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.7\linewidth, height=0.7\linewidth,
    title={\textbf{Stage 2 --- Confusion Matrix (row-normalised)}},
    title style={font=\small\bfseries}, xlabel={Predicted}, ylabel={True},
    xtick={0,1,2,3,4,5}, xticklabels={""" + xt + r"""},
    ytick={0,1,2,3,4,5}, yticklabels={""" + yt + r"""},
    x tick label style={rotate=40, anchor=east, font=\scriptsize},
    y tick label style={font=\scriptsize},
    enlargelimits=false, colormap/Blues, colorbar,
    point meta min=0, point meta max=1,
    nodes near coords={\pgfmathprintnumber[fixed,precision=2]{\pgfplotspointmeta}},
    nodes near coords style={font=\tiny}]
\addplot[matrix plot*, point meta=explicit, mesh/cols=6, mesh/rows=6]
    table[meta=C] {
x y C
""" + data + r"""
};
\end{axis}\end{tikzpicture}
\caption[Stage 2 confusion matrix]{Stage~2 row-normalised confusion matrix on the
test set. Colour, location, and count are near-perfect; main confusions are
between yes/no and single-choice, and within the rare multiple-choice class.}
\label{fig:stage2_confusion}
\end{figure}
"""
    open(os.path.join(OUT, "stage2_confusion.tex"), "w").write(tex)
    print("  saved stage2_confusion.tex")


def make_per_class_prf(y, p):
    from sklearn.metrics import precision_recall_fscore_support
    P, R, F, S = precision_recall_fscore_support(y, p, labels=list(range(6)),
                                                 zero_division=0)
    def co(v): return " ".join(f"({SHORT[i]},{v[i]:.3f})" for i in range(6))
    syms = ",".join(SHORT)
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[ybar, width=0.85\linewidth, height=7cm,
    title={\textbf{Stage 2 --- Per-Class Precision / Recall / F1}},
    title style={font=\small\bfseries}, ylabel={Score}, ymin=0, ymax=1.05,
    symbolic x coords={""" + syms + r"""}, xtick=data,
    x tick label style={rotate=25, anchor=east, font=\scriptsize},
    legend pos=south west, bar width=5pt, grid=major, grid style={gray!20}]
\addplot[fill=blue!60] coordinates {""" + co(P) + r"""};
\addlegendentry{Precision}
\addplot[fill=red!55] coordinates {""" + co(R) + r"""};
\addlegendentry{Recall}
\addplot[fill=green!55!black] coordinates {""" + co(F) + r"""};
\addlegendentry{F1}
\end{axis}\end{tikzpicture}
\caption[Stage 2 per-class metrics]{Stage~2 per-class precision, recall, and F1
on the test set. Support: """ + ", ".join(f"{SHORT[i]}~{int(S[i])}" for i in range(6)) + r""".}
\label{fig:stage2_per_class_prf}
\end{figure}
"""
    open(os.path.join(OUT, "stage2_per_class_prf.tex"), "w").write(tex)
    print("  saved stage2_per_class_prf.tex")


def main():
    print(f"\nStage 2 LaTeX plots -> {OUT}\n")
    make_curves()
    y, p = test_predictions()
    make_confusion(y, p)
    make_per_class_prf(y, p)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
