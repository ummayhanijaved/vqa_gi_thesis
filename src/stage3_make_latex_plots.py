#!/usr/bin/env python3
"""
Stage 3 — LaTeX (pgfplots) plot generator. Reads CSVs only (no model needed).
Outputs in figures/stage3_latex/:
  stage3_curves.tex      (epochs vs train/val acc + loss + macro-F1)
  stage3_ablation.tex    (component ablation bar chart)
  stage3_per_class.tex   (per-class P/R/F1 of the auxiliary router head)
PREAMBLE: \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\\usepackage{booktabs,float}
USAGE: python src/stage3_make_latex_plots.py
"""
import os
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
RES = os.path.join(PROJECT, "results")
OUT = os.path.join(PROJECT, "figures", "stage3_latex"); os.makedirs(OUT, exist_ok=True)


def make_curves():
    df = pd.read_csv(os.path.join(RES, "stage3_epoch_log.csv"))
    df.columns = [c.strip() for c in df.columns]
    def co(col, scale=1.0):
        return " ".join(f"({int(e)},{v*scale})" for e, v in zip(df["Epoch"], df[col]))
    # accuracies are stored as fractions (0.925) -> show as %
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.8\linewidth, height=6cm,
    title={\textbf{(a) Accuracy per Epoch}}, title style={font=\small\bfseries},
    xlabel={Epoch}, ylabel={Accuracy (\%)}, legend pos=south west,
    xtick={1,2,3,4,5}, grid=both, grid style={gray!25},
    tick label style={font=\small}, label style={font=\small}]
\addplot[blue, mark=*, thick] coordinates {""" + co("Tr Acc", 100) + r"""};
\addlegendentry{Train acc}
\addplot[red, mark=square*, thick] coordinates {""" + co("Val Acc", 100) + r"""};
\addlegendentry{Val acc}
\end{axis}\end{tikzpicture}

\vspace{0.6em}
\begin{tikzpicture}
\begin{axis}[width=0.8\linewidth, height=6cm,
    title={\textbf{(b) Loss and Validation Macro-F1}}, title style={font=\small\bfseries},
    xlabel={Epoch}, ylabel={Value}, legend pos=east,
    xtick={1,2,3,4,5}, grid=both, grid style={gray!25},
    tick label style={font=\small}, label style={font=\small}]
\addplot[blue, mark=*, thick] coordinates {""" + co("Tr Loss") + r"""};
\addlegendentry{Train loss}
\addplot[red, mark=square*, thick] coordinates {""" + co("Val Loss") + r"""};
\addlegendentry{Val loss}
\addplot[green!55!black, mark=triangle*, thick] coordinates {""" + co("Val Macro-F1") + r"""};
\addlegendentry{Val Macro-F1}
\end{axis}\end{tikzpicture}
\caption[Stage 3 training curves]{Stage~3 fusion training dynamics over 5 epochs,
measured on the auxiliary question-type routing head. The best validation
accuracy (92.50\%) is reached at epoch~1; later epochs slightly overfit, so the
epoch-1 checkpoint is retained.}
\label{fig:stage3_curves}
\end{figure}
"""
    open(os.path.join(OUT, "stage3_curves.tex"), "w").write(tex)
    print("  saved stage3_curves.tex")


def make_ablation():
    df = pd.read_csv(os.path.join(RES, "stage3_ablation_table.csv"))
    df.columns = [c.strip() for c in df.columns]
    # keep order; build coordinates (variant, val acc)
    coords = " ".join(f"({r['Variant'].replace('(','').replace(')','')},{r['Val Acc (%)']:.2f})"
                      for _, r in df.iterrows())
    syms = ",".join(r['Variant'].replace('(','').replace(')','') for _, r in df.iterrows())
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[ybar, width=0.85\linewidth, height=7cm,
    title={\textbf{Stage 3 --- Component Ablation (auxiliary routing accuracy)}},
    title style={font=\small\bfseries},
    ylabel={Validation Accuracy (\%)}, ymin=91.5, ymax=94,
    symbolic x coords={""" + syms + r"""}, xtick=data,
    x tick label style={rotate=20, anchor=east, font=\scriptsize},
    nodes near coords, nodes near coords style={font=\scriptsize,
        /pgf/number format/fixed, /pgf/number format/precision=2},
    bar width=22pt, grid=major, grid style={gray!20},
    enlarge x limits=0.18]
\addplot[fill=blue!55, draw=blue!70] coordinates {""" + coords + r"""};
\end{axis}\end{tikzpicture}
\caption[Stage 3 ablation]{Stage~3 component ablation on the auxiliary routing
head. Removing the disease gate or cross-attention costs only 0.18 and 0.41
percentage points respectively. Notably, a text-only variant marginally
\emph{exceeds} the full model on this routing metric (93.56\% vs 92.50\%),
confirming that question-type routing is essentially a textual task; the visual
and disease signals are introduced for the benefit of downstream answer
generation (Stage~4), not for routing.}
\label{fig:stage3_ablation}
\end{figure}
"""
    open(os.path.join(OUT, "stage3_ablation.tex"), "w").write(tex)
    print("  saved stage3_ablation.tex")


def make_per_class():
    df = pd.read_csv(os.path.join(RES, "stage3_results_table.csv"))
    df.columns = [c.strip() for c in df.columns]
    short = {"yes/no":"yes/no","single-choice":"single","multiple-choice":"multi",
             "color":"color","location":"location","numerical count":"count"}
    df["S"] = df["Class"].map(lambda c: short.get(c, c))
    def co(col): return " ".join(f"({r['S']},{r[col]:.3f})" for _, r in df.iterrows())
    syms = ",".join(df["S"].tolist())
    sup = ", ".join(f"{r['S']}~{int(r['Support'])}" for _, r in df.iterrows())
    tex = r"""\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[ybar, width=0.85\linewidth, height=7cm,
    title={\textbf{Stage 3 --- Auxiliary Router Per-Class P/R/F1}},
    title style={font=\small\bfseries}, ylabel={Score}, ymin=0, ymax=1.05,
    symbolic x coords={""" + syms + r"""}, xtick=data,
    x tick label style={rotate=25, anchor=east, font=\scriptsize},
    legend pos=south west, bar width=5pt, grid=major, grid style={gray!20}]
\addplot[fill=blue!60] coordinates {""" + co("Precision") + r"""};
\addlegendentry{Precision}
\addplot[fill=red!55] coordinates {""" + co("Recall") + r"""};
\addlegendentry{Recall}
\addplot[fill=green!55!black] coordinates {""" + co("F1") + r"""};
\addlegendentry{F1}
\end{axis}\end{tikzpicture}
\caption[Stage 3 per-class router]{Per-class precision, recall, and F1 of the
Stage~3 auxiliary routing head on the test set. Support: """ + sup + r""".}
\label{fig:stage3_per_class}
\end{figure}
"""
    open(os.path.join(OUT, "stage3_per_class.tex"), "w").write(tex)
    print("  saved stage3_per_class.tex")


def main():
    print(f"\nStage 3 LaTeX plots -> {OUT}\n")
    make_curves()
    make_ablation()
    make_per_class()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
