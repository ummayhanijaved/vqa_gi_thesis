#!/usr/bin/env python3
"""
Stage 4 — Interactive Inference Tester

Loads real test questions from Kvasir-VQA-x1 and runs them through
the complete Stage 4 pipeline, showing what answer the model produces
vs the ground truth.

Modes:
  --mode samples    Show N random samples from EACH route (default 3 each)
  --mode interactive  Type your own question, get an answer
  --mode all_routes Run all 6 routes once with one sample each

Usage:
  python stage4_inference_test.py --mode samples --n 3
  python stage4_inference_test.py --mode interactive
  python stage4_inference_test.py --mode all_routes
"""
import argparse
import os
import random
import sys
import textwrap

import torch
from datasets import load_from_disk

SRC_DIR = os.path.expanduser("~/vqa_gi_thesis/src")
sys.path.insert(0, SRC_DIR)

from stage4_revised import (
    CFG, ROUTE_NAMES, DistilBERTAnswerModel,
    FusionExtractor, TextPreprocessor, infer_route,
    YOLOLocationModel, YOLOCountModel,
)
from preprocessing import build_image_transform

# Map route → display label (built locally since stage4_revised exposes
# only ROUTE_NAMES like "yes_no", we want pretty labels for output)
ROUTE_LABELS = {
    0: "Yes/No",
    1: "Single Choice",
    2: "Multi Choice",
    3: "Colour",
    4: "Location",
    5: "Count",
}

# ─────────────────────────────────────────────────────────────────────────────
# ANSI Colours
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
BLUE   = "\033[94m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
BOLD   = "\033[1m";  RESET = "\033[0m"; GRAY = "\033[90m"

ROUTE_COLORS = {
    0: BLUE, 1: CYAN, 2: MAGENTA, 3: YELLOW, 4: GREEN, 5: RED,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 unified predictor
# ─────────────────────────────────────────────────────────────────────────────
class Stage4Predictor:
    """Loads all 6 route models and answers any test question end-to-end."""

    def __init__(self):
        from transformers import DistilBertTokenizerFast

        print(f"\n{BOLD}{CYAN}Loading Stage 4 models ...{RESET}\n")

        # Stage 3 feature extractor (frozen, used to compute fused vector)
        self.extractor = FusionExtractor(CFG["stage3_ckpt"])
        self.text_prep = TextPreprocessor()

        # build_image_transform may have either signature depending on
        # which preprocessing.py version is installed. Try both.
        try:
            self.transform = build_image_transform("test")
        except TypeError:
            try:
                self.transform = build_image_transform(is_train=False)
            except TypeError:
                self.transform = build_image_transform()

        # DistilBERT for routes 0-3
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            DistilBERTAnswerModel.MODEL_NAME)

        self.models = {}
        self.vocabs = {}
        for route in [0, 1, 2, 3]:
            ckpt_path = os.path.join(
                CFG["ckpt_dir"],
                f"stage4_revised_{ROUTE_NAMES[route]}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  ⚠️   No checkpoint for route {route} ({ROUTE_NAMES[route]})")
                continue

            ckpt  = torch.load(ckpt_path, map_location=CFG["device"],
                                weights_only=False)
            vocab = ckpt["vocab"]
            model = DistilBERTAnswerModel(vocab_per_route={route: vocab})
            model.load_state_dict(ckpt["model_state"])
            model = model.to(CFG["device"]).eval()

            self.models[route] = model
            self.vocabs[route] = vocab
            print(f"  ✅  Route {route} ({ROUTE_NAMES[route]:<15}) "
                  f"loaded — {len(vocab)} classes")

        # YOLO models
        seg_ckpt = os.path.join(
            CFG["ckpt_dir"], "yolo_seg_finetuned", "weights", "best.pt")
        det_ckpt = os.path.join(
            CFG["ckpt_dir"], "yolo_det_finetuned", "weights", "best.pt")
        self.yolo_loc = YOLOLocationModel(
            weights_path=seg_ckpt if os.path.exists(seg_ckpt) else None)
        self.yolo_cnt = YOLOCountModel(
            weights_path=det_ckpt if os.path.exists(det_ckpt) else None)
        print(f"  ✅  Route 4 (location) YOLO-Seg loaded")
        print(f"  ✅  Route 5 (count)    YOLO-Det loaded\n")

    @torch.no_grad()
    def predict(self, image_pil, question: str):
        """
        Run Stage 4 end-to-end on a (image, question) pair.
        Returns dict with predicted answer + route used.
        """
        route = infer_route(question)

        if route in [0, 1, 2, 3]:
            # DistilBERT routes — need fused features from Stage 3
            if route not in self.models:
                return {"route": route, "answer": "[no model loaded]",
                         "method": "ERROR"}

            # Step 1: Build image tensor + question tokens for Stage 3
            img_tensor = self.transform(image_pil.convert("RGB"))
            img_tensor = img_tensor.unsqueeze(0).to(CFG["device"])

            tp_out = self.text_prep.preprocess(question)
            input_ids = tp_out["input_ids"]
            attn_mask = tp_out["attention_mask"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attn_mask = attn_mask.unsqueeze(0)
            input_ids = input_ids.to(CFG["device"])
            attn_mask = attn_mask.to(CFG["device"])

            # Step 2: Run Stage 3 to get fused 535-D vector + disease one-hot
            with torch.no_grad():
                s3_out = self.extractor.extract(
                    img_tensor, input_ids, attn_mask)
            fused       = s3_out["fused_repr"]      # (1, 512)
            disease_vec = s3_out["disease_vec"]     # (1, 23)

            # Step 3: Run DistilBERT route head with fused conditioning
            model = self.models[route]
            qtokens = self.tokenizer(
                question, padding="max_length", truncation=True,
                max_length=CFG["max_input_len"], return_tensors="pt")
            q_ids = qtokens["input_ids"].to(CFG["device"])
            q_msk = qtokens["attention_mask"].to(CFG["device"])

            cls_repr = model._encode(fused, disease_vec, q_ids, q_msk)
            logits   = model.heads[str(route)](cls_repr)

            vocab = self.vocabs[route]
            if route == 2:
                # Multi-label
                probs = torch.sigmoid(logits)[0]
                preds = [vocab[i] for i, p in enumerate(probs)
                         if p.item() >= CFG["threshold"]]
                answer = ", ".join(preds) if preds else "(none)"
            else:
                pred_idx = logits.argmax(dim=-1).item()
                answer   = vocab[pred_idx]

            return {"route": route, "answer": answer, "method": "DistilBERT"}

        elif route == 4:
            # YOLO-Seg location — needs image file path
            tmp_path = "/tmp/stage4_inf.jpg"
            image_pil.save(tmp_path)
            answer = self.yolo_loc.predict(tmp_path)
            return {"route": route, "answer": answer,
                     "method": "YOLO-Segmentation"}

        elif route == 5:
            # YOLO-Det count — needs image file path
            tmp_path = "/tmp/stage4_inf.jpg"
            image_pil.save(tmp_path)
            answer = self.yolo_cnt.predict(tmp_path)
            return {"route": route, "answer": answer,
                     "method": "YOLO-Detection"}

        return {"route": -1, "answer": "[unknown route]", "method": "ERROR"}


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def short(s, n=80):
    s = str(s).strip()
    return s if len(s) <= n else s[:n - 3] + "..."


def print_result(question, gt, result, sample_num=None):
    route = result["route"]
    color = ROUTE_COLORS.get(route, RESET)
    pred  = str(result["answer"]).strip()
    gt    = str(gt).strip()
    correct = (pred.lower() == gt.lower()
               or (pred and pred.lower() in gt.lower()))
    icon = f"{GREEN}✅{RESET}" if correct else f"{RED}❌{RESET}"

    sep = "─" * 75
    if sample_num is not None:
        print(f"\n{color}{BOLD}{sep}")
        print(f"  Sample #{sample_num}  |  Route {route}: "
              f"{ROUTE_LABELS[route].upper()}  |  {result['method']}")
        print(f"{sep}{RESET}")
    else:
        print(f"\n{color}{BOLD}{sep}{RESET}")

    print(f"  {BOLD}Question:{RESET} {short(question, 200)}")
    print(f"  {BOLD}GT:      {RESET} {short(gt, 200)}")
    print(f"  {BOLD}Pred:    {RESET} {color}{short(pred, 200)}{RESET}  {icon}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — sample N questions from each route
# ─────────────────────────────────────────────────────────────────────────────
def mode_samples(predictor, raw_test, n_per_route=3, seed=42):
    print(f"\n{BOLD}{'='*75}{RESET}")
    print(f"{BOLD}  Stage 4 Inference Tester — {n_per_route} samples per route{RESET}")
    print(f"{BOLD}{'='*75}{RESET}")

    # Group test samples by route
    by_route = {r: [] for r in range(6)}
    for i, sample in enumerate(raw_test):
        q = sample.get("question", "")
        r = infer_route(q)
        by_route[r].append(i)

    print(f"\n  {BOLD}Test samples per route:{RESET}")
    for r in range(6):
        print(f"    Route {r} ({ROUTE_NAMES[r]:<15}): {len(by_route[r]):,} samples")

    rng = random.Random(seed)

    correct_by_route = {r: 0 for r in range(6)}
    total_by_route   = {r: 0 for r in range(6)}

    for route in range(6):
        if not by_route[route]:
            continue
        print(f"\n\n{BOLD}{'#'*75}")
        print(f"#  Route {route} — {ROUTE_LABELS[route].upper()}")
        print(f"{'#'*75}{RESET}")

        chosen = rng.sample(by_route[route], min(n_per_route, len(by_route[route])))
        for i, idx in enumerate(chosen, 1):
            s = raw_test[idx]
            try:
                result = predictor.predict(s["image"], s["question"])
                print_result(s["question"], s["answer"], result, sample_num=i)

                pred = str(result["answer"]).lower().strip()
                gt   = str(s["answer"]).lower().strip()
                if pred == gt or (pred and pred in gt):
                    correct_by_route[route] += 1
                total_by_route[route] += 1
            except Exception as e:
                print(f"\n  {RED}❌  Error on sample {idx}: {e}{RESET}")

    # Summary
    print(f"\n\n{BOLD}{'='*75}")
    print(f"  Sample Run Summary")
    print(f"{'='*75}{RESET}")
    total_c = sum(correct_by_route.values())
    total_n = sum(total_by_route.values())
    for r in range(6):
        if total_by_route[r] > 0:
            pct = 100 * correct_by_route[r] / total_by_route[r]
            print(f"  Route {r} ({ROUTE_NAMES[r]:<15}): "
                  f"{correct_by_route[r]}/{total_by_route[r]}  ({pct:.0f}%)")
    print(f"  {BOLD}Overall (substring match): "
          f"{total_c}/{total_n}  "
          f"({100*total_c/max(total_n,1):.1f}%){RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — interactive REPL
# ─────────────────────────────────────────────────────────────────────────────
def mode_interactive(predictor, raw_test):
    print(f"\n{BOLD}{'='*75}{RESET}")
    print(f"{BOLD}  Stage 4 Interactive Tester{RESET}")
    print(f"{BOLD}{'='*75}{RESET}")
    print(f"\n  Pick a test image by index (0-{len(raw_test)-1}), then ask any question.")
    print(f"  Type {BOLD}'random'{RESET} for a random sample, "
          f"{BOLD}'quit'{RESET} to exit.\n")

    while True:
        try:
            choice = input(f"{CYAN}Image index (or 'random'/'quit'):{RESET} ").strip()
            if choice.lower() in ("quit", "exit", "q"):
                break
            if choice.lower() == "random":
                idx = random.randint(0, len(raw_test) - 1)
            else:
                idx = int(choice)
                if not (0 <= idx < len(raw_test)):
                    print(f"  {RED}Index out of range{RESET}")
                    continue

            sample = raw_test[idx]
            print(f"\n  {GRAY}Selected test #{idx}{RESET}")
            print(f"  {GRAY}Original question: {short(sample['question'], 100)}{RESET}")
            print(f"  {GRAY}Original answer  : {short(sample['answer'], 100)}{RESET}\n")

            question = input(f"{CYAN}Your question (Enter = use original):{RESET} ").strip()
            if not question:
                question = sample["question"]

            result = predictor.predict(sample["image"], question)
            print_result(question, sample["answer"], result)
            print()

        except (KeyboardInterrupt, EOFError):
            print()
            break
        except ValueError:
            print(f"  {RED}Please enter a number, 'random', or 'quit'{RESET}")
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}")

    print(f"\n{GREEN}Thanks for testing!{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — one sample per route, quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
def mode_all_routes(predictor, raw_test):
    print(f"\n{BOLD}{'='*75}{RESET}")
    print(f"{BOLD}  Stage 4 — One Sample Per Route (Quick Smoke Test){RESET}")
    print(f"{BOLD}{'='*75}{RESET}")

    by_route = {r: None for r in range(6)}
    for i, sample in enumerate(raw_test):
        r = infer_route(sample.get("question", ""))
        if r in by_route and by_route[r] is None:
            by_route[r] = i
        if all(v is not None for v in by_route.values()):
            break

    for route in range(6):
        if by_route[route] is None:
            print(f"\n  {YELLOW}No test sample found for Route {route}{RESET}")
            continue
        s = raw_test[by_route[route]]
        try:
            result = predictor.predict(s["image"], s["question"])
            print_result(s["question"], s["answer"], result, sample_num=route)
        except Exception as e:
            print(f"\n  {RED}❌  Error on Route {route}: {e}{RESET}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 4 inference tester")
    parser.add_argument("--mode", default="samples",
                        choices=["samples", "interactive", "all_routes"])
    parser.add_argument("--n", type=int, default=3,
                        help="Samples per route in 'samples' mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}Loading test dataset ...{RESET}")
    raw      = load_from_disk(CFG["data_dir"])
    raw_test = raw["test"] if "test" in raw else raw["train"]
    print(f"  Test split: {len(raw_test):,} samples")

    predictor = Stage4Predictor()

    if args.mode == "samples":
        mode_samples(predictor, raw_test, n_per_route=args.n, seed=args.seed)
    elif args.mode == "interactive":
        mode_interactive(predictor, raw_test)
    elif args.mode == "all_routes":
        mode_all_routes(predictor, raw_test)


if __name__ == "__main__":
    main()