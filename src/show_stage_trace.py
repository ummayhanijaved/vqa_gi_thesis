#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
import random
import traceback
import gc
import shutil
import re

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

import stage5_pipeline_test as ppt
import stage6_explainability as s6

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
OUT_DIR = os.path.join(PROJECT, "logs", "stage_trace")
IMG_OUT_DIR = os.path.join(OUT_DIR, "selected_images")
PER_ROUTE_OUT_DIR = os.path.join(OUT_DIR, "selected_per_route")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(PER_ROUTE_OUT_DIR, exist_ok=True)


def print_mem(label):
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / 1024**3
        vm = psutil.virtual_memory()
        used = vm.used / 1024**3
        total = vm.total / 1024**3
        print(
            f"  [MEM] {label}: process={rss:.2f} GB | system={used:.2f}/{total:.2f} GB",
            flush=True,
        )
    except Exception:
        pass


def safe_filename(s, max_len=80):
    """Convert text to a safe short filename component."""
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "na"
    return s[:max_len]


def get_route_dir(route, route_name):
    route_folder = f"R{route}_{safe_filename(route_name, max_len=40)}"
    route_dir = os.path.join(PER_ROUTE_OUT_DIR, route_folder)
    os.makedirs(route_dir, exist_ok=True)
    return route_dir, route_folder


def make_base_name(route, route_name, kept_no, img_id, s4_answer, disease_name):
    route_part = f"R{route}_{safe_filename(route_name, max_len=40)}"
    img_part = safe_filename(img_id, max_len=60)
    ans_part = safe_filename(s4_answer, max_len=60)
    focus_part = safe_filename(disease_name, max_len=60)
    return (
        f"{route_part}_sample{kept_no:04d}_"
        f"img_{img_part}_ans_{ans_part}_focus_{focus_part}"
    )


def get_stage1_confidence(res):
    disease_vec = res.get("disease_vec", None)
    try:
        if disease_vec is not None:
            arr = np.array(disease_vec).reshape(-1)
            if arr.size > 0:
                return float(np.max(arr)) * 100.0
    except Exception:
        pass
    return None


def all_routes_filled(route_counts, max_per_route):
    if max_per_route <= 0:
        return False
    return all(route_counts.get(r, 0) >= max_per_route for r in range(6))


def latex_escape(s):
    s = "" if s is None else str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def write_latex_2_per_route(df, tex_path):
    """Create LaTeX section with selected examples grouped by route."""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated qualitative examples: two high-score samples per route if available.\n")
        f.write("% Copy selected_per_route/ folder into figures/stage67/ if using paths below.\n\n")
        for route in sorted(df["route"].unique()):
            sub = df[df["route"] == route].copy()
            route_name = str(sub.iloc[0]["route_name"])
            f.write(f"\\subsubsection{{R{route}: {latex_escape(route_name)}}}\n\n")
            for _, r in sub.iterrows():
                heatmap_rel = str(r.get("stage7_heatmap_rel", ""))
                image_rel = str(r.get("image_file_rel", ""))
                f.write("\\begin{figure*}[p]\n\\centering\n")
                if heatmap_rel.endswith(".png"):
                    f.write(f"\\includegraphics[width=\\linewidth]{{{heatmap_rel}}}\n")
                elif image_rel.endswith((".jpg", ".png", ".jpeg")):
                    f.write(f"\\includegraphics[width=0.65\\linewidth]{{{image_rel}}}\n")
                f.write(
                    "\\caption{High-confidence example from "
                    f"R{route} ({latex_escape(route_name)}). "
                    f"Question: {latex_escape(r['question'])} "
                    f"Stage~4 answer: {latex_escape(r['stage4_answer'])}; "
                    f"Stage~5 answer: {latex_escape(r['stage5_sentence'])}; "
                    f"average score: {latex_escape(r['avg_score'])}\\%.}}\n"
                )
                f.write(f"\\label{{fig:stage67_R{route}_sample{int(r['kept_sample']):04d}}}\n")
                f.write("\\end{figure*}\n\n")

                f.write("\\begin{table*}[p]\n\\centering\n\\small\n")
                f.write("\\begin{tabular}{p{0.23\\linewidth} p{0.70\\linewidth}}\n")
                f.write("\\toprule\n\\textbf{Component} & \\textbf{Output} \\\\\n\\midrule\n")
                f.write(f"Image ID & {latex_escape(r['img_id'])} \\\\\n")
                f.write(f"Question & {latex_escape(r['question'])} \\\\\n")
                f.write(f"Stage 1 disease/focus & {latex_escape(r['disease_name'])} \\\\\n")
                f.write(f"Stage 2 route & R{route} ({latex_escape(route_name)}) \\\\\n")
                f.write("Stage 3 fusion & Cross-attention fusion of visual, question, and disease-context features. \\\\\n")
                f.write(f"Stage 4 structured answer & {latex_escape(r['stage4_answer'])} \\\\\n")
                f.write(f"Stage 5 verbal sentence & {latex_escape(r['stage5_sentence'])} \\\\\n")
                f.write(f"Stage 6 explanation & {latex_escape(r['stage6_explanation'])} \\\\\n")
                f.write(f"Stage 7 Grad-CAM & {latex_escape(os.path.basename(str(r['stage7_heatmap_name'])))} \\\\\n")
                f.write(f"Ground truth & {latex_escape(r['ground_truth'])} \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write(f"\\caption{{Stage-wise outputs for selected R{route} ({latex_escape(route_name)}) example.}}\n")
                f.write(f"\\label{{tab:stage67_R{route}_sample{int(r['kept_sample']):04d}}}\n")
                f.write("\\end{table*}\n\n")


def find_image(img_id, image_dir):
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        p = os.path.join(image_dir, f"{img_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def safe_get_split(raw):
    try:
        if "test" in raw:
            return raw["test"]
        return raw["train"]
    except Exception:
        return raw["test"] if "test" in raw.keys() else raw["train"]


def get_disease_name(res):
    disease_name = ""

    for key in ["disease_name", "predicted_disease", "stage1_disease", "top_disease"]:
        if key in res and res[key]:
            disease_name = str(res[key])
            break

    if not disease_name:
        disease_vec = res.get("disease_vec", None)
        try:
            if disease_vec is not None:
                arr = np.array(disease_vec).reshape(-1)
                if arr.size > 0:
                    idx = int(np.argmax(arr))
                    names = getattr(ppt, "DISEASE_NAMES", None)
                    if names and idx < len(names):
                        disease_name = str(names[idx])
                    else:
                        disease_name = f"disease_{idx}"
        except Exception:
            pass

    return disease_name if disease_name else "unknown"


def pass_score_filter(args, rouge_score, meteor_score, chrf_score):
    avg_score = (rouge_score + meteor_score + chrf_score) / 3.0

    if args.score_mode == "all":
        return (
            rouge_score >= args.min_score
            and meteor_score >= args.min_score
            and chrf_score >= args.min_score
        )

    if args.score_mode == "avg":
        return avg_score >= args.min_score

    if args.score_mode == "rouge1":
        return rouge_score >= args.min_score

    if args.score_mode == "meteor":
        return meteor_score >= args.min_score

    if args.score_mode == "chrf":
        return chrf_score >= args.min_score

    return False


def select_samples(test_split, n, balanced=False):
    rng = random.Random(42)
    length = len(test_split)

    if length == 0:
        return []

    if not balanced:
        idxs = rng.sample(range(length), min(n, length))
        return [test_split[int(i)] for i in idxs]

    selected = []
    route_to_indices = {r: [] for r in range(6)}

    scan_n = min(length, max(200, n * 50))
    scan_indices = list(range(scan_n))
    rng.shuffle(scan_indices)

    for idx in scan_indices:
        try:
            s = test_split[int(idx)]
            q = s.get("question", "")
            if not q:
                continue
            r = ppt.infer_route(q)
            if r in route_to_indices:
                route_to_indices[r].append(int(idx))
        except Exception:
            continue

    for r in range(6):
        if len(selected) >= n:
            break
        if route_to_indices[r]:
            selected.append(test_split[route_to_indices[r][0]])

    while len(selected) < n:
        idx = rng.randrange(length)
        s = test_split[int(idx)]
        if s.get("question", ""):
            selected.append(s)

    return selected[:n]


def copy_sample_image(img_path, base_name, route_dir):
    try:
        ext = os.path.splitext(img_path)[1]
        saved_img_name = f"{base_name}_original{ext}"
        saved_img_path = os.path.join(route_dir, saved_img_name)
        shutil.copy2(img_path, saved_img_path)
        return saved_img_path, saved_img_name
    except Exception:
        return "", ""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--balanced", action="store_true")

    ap.add_argument(
        "--scan_all",
        action="store_true",
        help="scan dataset lazily and keep samples passing score filter",
    )
    ap.add_argument(
        "--max_scan",
        type=int,
        default=0,
        help="scan only first N samples; 0 means no limit",
    )
    ap.add_argument(
        "--max_keep",
        type=int,
        default=0,
        help="stop after keeping N high-score samples; 0 means no limit",
    )

    ap.add_argument(
        "--max_per_route",
        type=int,
        default=0,
        help="keep at most N high-score samples per route; e.g. 2 gives two examples per R0-R5",
    )
    ap.add_argument(
        "--stop_when_routes_full",
        action="store_true",
        help="stop scanning once every route has max_per_route samples",
    )
    ap.add_argument(
        "--make_latex_examples",
        action="store_true",
        help="write LaTeX examples grouped by route",
    )

    ap.add_argument(
        "--min_score",
        type=float,
        default=90.0,
        help="minimum score threshold in percent",
    )
    ap.add_argument(
        "--score_mode",
        default="all",
        choices=["all", "avg", "rouge1", "meteor", "chrf"],
        help="all means ROUGE, METEOR and CHRF must all pass",
    )

    ap.add_argument(
        "--group_by",
        default="route",
        choices=["route", "question", "disease", "none"],
        help="summary grouping for selected samples",
    )

    ap.add_argument(
        "--with_gradcam",
        action="store_true",
        help="load Stage 7 Grad-CAM model; disabled by default to save RAM",
    )
    ap.add_argument("--show_mem", action="store_true")

    args = ap.parse_args()

    print("\n" + "=" * 74)
    print("  HIGH-SCORE STAGE TRACE REPORT")
    print("=" * 74)
    print(f"  min_score  : {args.min_score}")
    print(f"  score_mode : {args.score_mode}")
    print(f"  scan_all   : {args.scan_all}")
    print(f"  max_per_route: {args.max_per_route}")
    print("=" * 74 + "\n")

    if args.show_mem:
        print_mem("start")

    print("  Loading FullPipelinePredictor + Stage6 generator ...", flush=True)
    predictor = ppt.FullPipelinePredictor()
    generator = s6.MedicalResponseGenerator()

    if args.show_mem:
        print_mem("after FullPipelinePredictor + Stage6 generator")

    gradcam = None
    transform = None
    s7 = None

    if args.with_gradcam:
        print("  Loading Grad-CAM model ...", flush=True)
        try:
            import stage7_gradcam as s7
            s1_model = s7.load_stage1_model()
            gradcam = s7.GradCAM(s1_model)
            transform = s7.get_eval_transform()
            print("  Grad-CAM ready\n", flush=True)
        except Exception as e:
            print(f"  Grad-CAM unavailable: {str(e)[:120]}\n", flush=True)
            gradcam = None
            transform = None
            s7 = None

    print("  Loading dataset ...", flush=True)
    from datasets import load_from_disk

    raw = load_from_disk(ppt.CFG["data_dir"])
    test_split = safe_get_split(raw)

    if args.show_mem:
        print_mem("after dataset load")

    image_dir = ppt.S4_CFG.get("image_dir", "")
    if not image_dir or not os.path.exists(image_dir):
        for cand in [
            os.path.expanduser("~/data/kvasir_local/images"),
            os.path.expanduser("~/vqa_gi_thesis/data/images"),
            os.path.expanduser("~/vqa_gi_thesis/data/kvasir_raw/images"),
        ]:
            if os.path.exists(cand):
                image_dir = cand
                break

    print(f"  Dataset samples: {len(test_split)}")
    print(f"  Image dir      : {image_dir}")

    if args.scan_all:
        scan_len = len(test_split)
        if args.max_scan and args.max_scan > 0:
            scan_len = min(scan_len, args.max_scan)
        selected = range(scan_len)  # lazy indexing: avoids loading all rows into RAM
    else:
        selected = select_samples(test_split, args.n, balanced=args.balanced)

    print(f"  Candidate samples to process: {len(selected)}\n", flush=True)

    rows = []
    heatmap_paths = []
    processed = 0
    kept = 0
    route_counts = {r: 0 for r in range(6)}
    route_seen_passing = {r: 0 for r in range(6)}

    for i, item in enumerate(selected, 1):
        try:
            processed += 1
            sample = test_split[int(item)] if args.scan_all else item

            img_id = sample.get("img_id", sample.get("image_id", ""))
            img_path = find_image(img_id, image_dir)

            if not img_path:
                print(f"  [{i}] skipped: image not found: {img_id}", flush=True)
                continue

            question = str(sample.get("question", "")).strip()
            gt = str(sample.get("answer", "")).strip()

            if not question:
                print(f"  [{i}] skipped: empty question", flush=True)
                continue

            if args.scan_all:
                print(f"  [{i}/{len(selected)}] processing img_id={img_id}", flush=True)
            else:
                print(f"\n  Processing sample {i}: {img_id}", flush=True)

            # Prediction does not need gradients.
            if torch is not None:
                with torch.no_grad():
                    res = predictor.predict(img_path, question)
            else:
                res = predictor.predict(img_path, question)

            route = int(res.get("route", 0))
            route_name = res.get("route_name", ppt.ROUTE_NAMES.get(route, "?"))
            s4_answer = str(res.get("s4_answer", ""))
            s5_sentence = str(res.get("s5_sentence", ""))

            disease_name = get_disease_name(res)
            stage1_conf = get_stage1_confidence(res)

            med = generator.generate(
                route=route,
                sentence=s5_sentence,
                disease_vec=res.get("disease_vec", None),
                s4_answer=s4_answer,
            )

            rouge = s6.rouge_metrics(s5_sentence, gt)
            meteor = s6.meteor_metric(s5_sentence, gt)
            chrf = s6.chrf_metric(s5_sentence, gt)

            rouge_score = rouge["rouge1"] * 100
            meteor_score = meteor * 100
            chrf_score = chrf * 100
            avg_score = (rouge_score + meteor_score + chrf_score) / 3.0

            if args.scan_all:
                if not pass_score_filter(args, rouge_score, meteor_score, chrf_score):
                    continue

            route_seen_passing[route] = route_seen_passing.get(route, 0) + 1

            if args.max_per_route and args.max_per_route > 0:
                if route_counts.get(route, 0) >= args.max_per_route:
                    print(
                        f"  [{i}] high-score but skipped: R{route} already has "
                        f"{args.max_per_route} samples",
                        flush=True,
                    )
                    if args.stop_when_routes_full and all_routes_filled(route_counts, args.max_per_route):
                        print("  All routes filled. Stopping scan.", flush=True)
                        break
                    continue

            kept += 1
            route_counts[route] = route_counts.get(route, 0) + 1

            route_dir, route_folder = get_route_dir(route, route_name)
            base_name = make_base_name(route, route_name, kept, img_id, s4_answer, disease_name)
            saved_img_path, saved_img_name = copy_sample_image(img_path, base_name, route_dir)

            heatmap_file = ""
            heatmap_name = ""
            if gradcam is not None and transform is not None and s7 is not None:
                try:
                    # Grad-CAM MUST have gradients enabled.
                    if torch is not None:
                        with torch.enable_grad():
                            viz = s7.process_image(
                                gradcam,
                                gradcam.model,
                                transform,
                                img_path,
                                question,
                                s5_sentence,
                            )
                    else:
                        viz = s7.process_image(
                            gradcam,
                            gradcam.model,
                            transform,
                            img_path,
                            question,
                            s5_sentence,
                        )

                    heatmap_name = f"{base_name}_gradcam.png"
                    heatmap_file = os.path.join(route_dir, heatmap_name)
                    title = (
                        f"Sample {kept} | R{route} {route_name} | "
                        f"A: {s4_answer[:30]} | focus: {viz['disease_name']}"
                    )
                    s7.save_visualization(
                        viz["rgb"],
                        viz["heatmap"],
                        viz["overlay"],
                        viz["heatmap_color"],
                        title,
                        heatmap_file,
                    )
                    heatmap_paths.append(heatmap_file)
                except Exception as e:
                    heatmap_file = f"gradcam failed: {str(e)[:80]}"
                    heatmap_name = heatmap_file
                    print(f"  Grad-CAM failed for kept sample {kept}: {e}", flush=True)

            image_file_rel = os.path.join(
                "figures", "stage67", "selected_per_route", route_folder, saved_img_name
            ) if saved_img_name else ""

            stage7_heatmap_rel = os.path.join(
                "figures", "stage67", "selected_per_route", route_folder, heatmap_name
            ) if heatmap_name and heatmap_name.endswith(".png") else heatmap_name

            print("\n  " + "-" * 70)
            print(f"  KEPT SAMPLE {kept}")
            print(f"  Route        : R{route} ({route_name}) [{route_counts.get(route,0)}/{args.max_per_route or '∞'}]")
            print(f"  Disease      : {disease_name}")
            if stage1_conf is not None:
                print(f"  Stage1 conf  : {stage1_conf:.2f}%")
            print(f"  Image        : {saved_img_name}")
            print(f"  Question     : {question}")
            print(f"  Stage 4 ans  : {s4_answer}")
            print(f"  Stage 5 sent : {s5_sentence}")
            print(f"  Stage 6 expl : {str(med.get('explanation', ''))}")
            print(
                f"  Scores       : ROUGE-1={rouge_score:.1f}, "
                f"METEOR={meteor_score:.1f}, CHRF++={chrf_score:.1f}, "
                f"AVG={avg_score:.1f}"
            )
            print(f"  Ground truth : {gt}")
            print(f"  Grad-CAM     : {os.path.basename(heatmap_file) if heatmap_file else '-'}")
            print("  " + "-" * 70 + "\n")

            rows.append(
                {
                    "kept_sample": kept,
                    "original_index": i,
                    "img_id": img_id,
                    "image_file": saved_img_path,
                    "image_file_name": saved_img_name,
                    "image_file_rel": image_file_rel,
                    "route": route,
                    "route_name": route_name,
                    "route_folder": route_folder,
                    "disease_name": disease_name,
                    "stage1_confidence": round(stage1_conf, 2) if stage1_conf is not None else "",
                    "question": question,
                    "stage4_answer": s4_answer,
                    "stage5_sentence": s5_sentence,
                    "stage6_explanation": med.get("explanation", ""),
                    "rouge1": round(rouge_score, 2),
                    "meteor": round(meteor_score, 2),
                    "chrf": round(chrf_score, 2),
                    "avg_score": round(avg_score, 2),
                    "stage7_heatmap": heatmap_file,
                    "stage7_heatmap_name": heatmap_name,
                    "stage7_heatmap_rel": stage7_heatmap_rel,
                    "ground_truth": gt,
                }
            )

            if args.show_mem:
                print_mem(f"after kept sample {kept}")

            gc.collect()
            if (
                torch is not None
                and hasattr(torch, "cuda")
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

            if args.max_keep and args.max_keep > 0 and kept >= args.max_keep:
                print(f"  Reached max_keep={args.max_keep}. Stopping scan.")
                break

            if (
                args.stop_when_routes_full
                and args.max_per_route > 0
                and all_routes_filled(route_counts, args.max_per_route)
            ):
                print("  All routes filled. Stopping scan.")
                break

        except Exception as e:
            print(f"\n  Error while processing candidate {i}: {e}")
            traceback.print_exc()
            continue

    if gradcam is not None:
        try:
            gradcam.remove_hooks()
        except Exception:
            pass

    if not rows:
        print("\n  No high-score samples found.")
        print("  Try lower threshold, for example:")
        print("  python show_stage_trace.py --scan_all --max_scan 500 --min_score 80 --score_mode avg")
        return

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUT_DIR, "high_score_samples.csv")
    md_path = os.path.join(OUT_DIR, "high_score_samples_with_images.md")
    txt_path = os.path.join(OUT_DIR, "high_score_samples_text.txt")
    latex_examples_path = os.path.join(OUT_DIR, "stage67_2_per_route_examples.tex")

    df.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HIGH-SCORE SAMPLE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Threshold: {args.min_score}\n")
        f.write(f"Score mode: {args.score_mode}\n")
        f.write(f"Processed: {processed}\n")
        f.write(f"Kept: {kept}\n")
        f.write(f"Max per route: {args.max_per_route}\n")
        f.write(f"Route counts: {route_counts}\n")
        f.write(f"Passing route counts before cap: {route_seen_passing}\n\n")

        for _, r in df.iterrows():
            f.write("-" * 80 + "\n")
            f.write(f"Sample: {r['kept_sample']}\n")
            f.write(f"Image file: {r['image_file']}\n")
            f.write(f"Image rel : {r['image_file_rel']}\n")
            f.write(f"Route: {r['route']} ({r['route_name']})\n")
            f.write(f"Disease: {r['disease_name']}\n")
            f.write(f"Question: {r['question']}\n")
            f.write(f"Stage 4 answer: {r['stage4_answer']}\n")
            f.write(f"Stage 5 sentence: {r['stage5_sentence']}\n")
            f.write(f"Stage 6 explanation: {r['stage6_explanation']}\n")
            f.write(
                f"Scores: ROUGE-1={r['rouge1']} | METEOR={r['meteor']} | "
                f"CHRF++={r['chrf']} | AVG={r['avg_score']}\n"
            )
            f.write(f"Grad-CAM file: {r['stage7_heatmap']}\n")
            f.write(f"Grad-CAM rel : {r['stage7_heatmap_rel']}\n")
            f.write(f"Ground truth: {r['ground_truth']}\n\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# High-Score Correct Samples with Images\n\n")
        f.write(f"**Threshold:** {args.min_score}%  \n")
        f.write(f"**Score mode:** `{args.score_mode}`  \n")
        f.write(f"**Processed candidates:** {processed}  \n")
        f.write(f"**Kept samples:** {kept}  \n")
        f.write(f"**Max per route:** {args.max_per_route}  \n")
        f.write(f"**Route counts:** `{route_counts}`  \n\n")

        f.write("## Score Mode\n\n")
        f.write("- `all`: ROUGE-1, METEOR, and CHRF++ must all be above threshold.\n")
        f.write("- `avg`: average of ROUGE-1, METEOR, and CHRF++ must be above threshold.\n")
        f.write("- `rouge1`, `meteor`, `chrf`: only that metric is checked.\n\n")

        if args.group_by != "none":
            f.write("## Summary by Group\n\n")

            if args.group_by == "route":
                group_col = "route_name"
            elif args.group_by == "question":
                group_col = "question"
            elif args.group_by == "disease":
                group_col = "disease_name"
            else:
                group_col = None

            if group_col:
                summary = (
                    df.groupby(group_col)
                    .agg(
                        count=("kept_sample", "count"),
                        avg_rouge1=("rouge1", "mean"),
                        avg_meteor=("meteor", "mean"),
                        avg_chrf=("chrf", "mean"),
                        avg_score=("avg_score", "mean"),
                    )
                    .reset_index()
                )

                f.write("| Group | Count | Avg ROUGE-1 | Avg METEOR | Avg CHRF++ | Avg Score |\n")
                f.write("|---|---:|---:|---:|---:|---:|\n")
                for _, s in summary.iterrows():
                    f.write(
                        f"| {s[group_col]} | {int(s['count'])} | "
                        f"{s['avg_rouge1']:.2f} | {s['avg_meteor']:.2f} | "
                        f"{s['avg_chrf']:.2f} | {s['avg_score']:.2f} |\n"
                    )
                f.write("\n")

        f.write("## Selected Samples\n\n")

        for _, r in df.iterrows():
            f.write(
                f"## Sample {r['kept_sample']} - Route {r['route']} "
                f"({r['route_name']})\n\n"
            )

            if r.get("image_file", ""):
                f.write(f"![Sample image]({r['image_file']})\n\n")

            if r.get("stage7_heatmap", "") and str(r["stage7_heatmap"]).endswith(".png"):
                f.write(f"![Grad-CAM heatmap]({r['stage7_heatmap']})\n\n")

            f.write(f"**Image ID:** `{r['img_id']}`  \n")
            f.write(f"**Disease category:** {r['disease_name']}  \n")
            f.write(f"**Question:** {r['question']}  \n\n")

            f.write("| Stage | Output |\n")
            f.write("|---|---|\n")
            f.write(f"| Stage 4 structured answer | `{r['stage4_answer']}` |\n")
            f.write(f"| Stage 5 verbal sentence | {r['stage5_sentence']} |\n")
            f.write(f"| Stage 6 textual explanation | {r['stage6_explanation']} |\n")
            f.write(f"| Ground truth | {r['ground_truth']} |\n\n")

            f.write("| Metric | Score |\n")
            f.write("|---|---:|\n")
            f.write(f"| ROUGE-1 | {r['rouge1']} |\n")
            f.write(f"| METEOR | {r['meteor']} |\n")
            f.write(f"| CHRF++ | {r['chrf']} |\n")
            f.write(f"| Average | {r['avg_score']} |\n\n")

            if r.get("stage7_heatmap", ""):
                f.write(f"**Grad-CAM heatmap:** `{r['stage7_heatmap']}`\n\n")

            f.write("---\n\n")

    if args.make_latex_examples:
        write_latex_2_per_route(df, latex_examples_path)

    if args.with_gradcam and heatmap_paths:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt

            n = len(heatmap_paths)
            cols = 2
            rows_g = (n + cols - 1) // cols

            fig, axes = plt.subplots(rows_g, cols, figsize=(14, 4 * rows_g))
            axes = np.array(axes).reshape(-1)

            for ax in axes:
                ax.axis("off")

            for ax, p in zip(axes, heatmap_paths):
                ax.imshow(mpimg.imread(p))
                ax.set_title(os.path.basename(p), fontsize=7)

            grid_path = os.path.join(OUT_DIR, "stage_trace_gradcam_grid.png")
            plt.tight_layout()
            plt.savefig(grid_path, dpi=110, bbox_inches="tight")
            plt.close()

            print(f"\n  Combined heatmap grid: {grid_path}")

        except Exception as e:
            print(f"  Grid build skipped: {str(e)[:80]}")

    print("\n" + "=" * 74)
    print("  DONE")
    print(f"  Processed candidates : {processed}")
    print(f"  Kept high-score      : {kept}")
    print(f"  CSV                  : {csv_path}")
    print(f"  Markdown with images : {md_path}")
    print(f"  Text report          : {txt_path}")
    print(f"  Copied images folder : {IMG_OUT_DIR}")
    print(f"  Per-route output dir : {PER_ROUTE_OUT_DIR}")
    print(f"  Route counts         : {route_counts}")
    if args.make_latex_examples:
        print(f"  LaTeX examples       : {latex_examples_path}")
    if args.with_gradcam:
        print(f"  Grad-CAM heatmaps    : {PER_ROUTE_OUT_DIR}/R*/")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n  SCRIPT CRASHED WITH PYTHON ERROR:")
        print(str(e))
        traceback.print_exc()
        sys.exit(1)
