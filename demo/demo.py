"""
=============================================================================
INTERACTIVE DEMO — Full VQA Pipeline
Thesis: Advancing Medical AI with Explainable VQA on GI Imaging

USAGE:
    # Single prediction
    python demo/demo.py \
        --image ./data/kvasir_raw/images/xxx.jpg \
        --question "Is there a polyp visible?"

    # Interactive mode (type questions live)
    python demo/demo.py --image ./data/kvasir_raw/images/xxx.jpg

    # Auto demo on random test images
    python demo/demo.py --auto --n 5
=============================================================================
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.expanduser("~"))

import torch
from PIL import Image as PILImage


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Explainable VQA on GI Imaging — Interactive Demo        ║
║     Thesis: Ummay Hani Javed (24i-8211)                     ║
║     Pipeline: ResNet50 → DistilBERT → CrossAttn → Answer    ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_result(result: dict):
    d_vec = result["disease_vec"]
    active = [(i, p) for i, p in enumerate(d_vec) if p > 0.3]
    active.sort(key=lambda x: -x[1])

    DISEASE_NAMES = [
        "Polyp (Pedunculated)","Polyp (Sessile)","Polyp (Hyperplastic)",
        "Esophagitis","Gastritis","Ulcerative Colitis","Crohn's Disease",
        "Barrett's Esophagus","Gastric Ulcer","Duodenal Ulcer",
        "Erosion","Hemorrhoids","Diverticulosis",
        "Normal Cecum","Normal Pylorus","Normal Z-Line",
        "Ileocecal Valve","Retroflex Rectum","Retroflex Stomach",
        "Dyed Lifted Polyp","Dyed Resection Margin",
        "Foreign Body","Instrument"
    ]

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  QUESTION   : {result['question'][:60]}
│  ANSWER     : {result['answer'].upper()}
│  TYPE       : {result['route_name']}
│  CONFIDENCE : {result['route_conf']*100:.1f}%
├─────────────────────────────────────────────────────────────┤
│  DISEASE CONTEXT:""")
    if active:
        for idx, prob in active[:5]:
            bar = "█" * int(prob * 20)
            print(f"│    {DISEASE_NAMES[idx]:<28} {bar:<20} {prob*100:.1f}%")
    else:
        print("│    No diseases detected above 30% threshold")
    print(f"""├─────────────────────────────────────────────────────────────┤
│  SPATIAL ATTENTION: GradCAM saved to ./figures/explainability/
└─────────────────────────────────────────────────────────────┘""")


def run_demo(image_path: str, question: str, predictor) -> dict:
    result = predictor.predict_full(image_path, question)
    print_result(result)
    # Save explainability report
    from explainability import plot_explainability_report
    path = plot_explainability_report(result)
    print(f"\n  📊  Report: {path}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--auto",     action="store_true")
    parser.add_argument("--n",        type=int, default=5)
    args = parser.parse_args()

    print_banner()
    print("  Loading pipeline (this takes ~30 seconds) ...")

    from explainability import ExplainabilityPredictor
    predictor = ExplainabilityPredictor()

    if args.auto:
        # Run on random test images
        from datasets import load_from_disk, Image as HFImage
        from stage4_answer_generation import CFG as S4_CFG
        from stage3_multimodal_fusion import infer_qtype_label
        import random

        raw = load_from_disk(S4_CFG["data_dir"])
        raw = raw.cast_column("image", HFImage())
        samples = random.sample(list(raw["test"]), args.n)

        for i, ex in enumerate(samples):
            print(f"\n  Sample {i+1}/{args.n}")
            img_path = f"/tmp/demo_{i}.jpg"
            ex["image"].convert("RGB").save(img_path, format="JPEG")
            run_demo(img_path, ex["question"], predictor)

    elif args.image and args.question:
        run_demo(args.image, args.question, predictor)

    elif args.image:
        # Interactive mode
        print(f"  Image: {args.image}")
        print("  Type your questions below (q to quit)\n")
        SAMPLE_QUESTIONS = [
            "Is there a polyp visible?",
            "What color is the finding?",
            "Where is the finding located?",
            "How many polyps are visible?",
            "What abnormalities are present?",
        ]
        print("  Suggested questions:")
        for i, q in enumerate(SAMPLE_QUESTIONS):
            print(f"    [{i+1}] {q}")
        print()

        while True:
            try:
                user_input = input("  Question (or 1-5 for suggestion, q to quit): ").strip()
                if user_input.lower() == "q":
                    print("  Goodbye!")
                    break
                if user_input.isdigit() and 1 <= int(user_input) <= 5:
                    question = SAMPLE_QUESTIONS[int(user_input)-1]
                    print(f"  Using: {question}")
                else:
                    question = user_input
                if question:
                    run_demo(args.image, question, predictor)
            except KeyboardInterrupt:
                print("\n  Goodbye!")
                break
    else:
        print("  Usage examples:")
        print("    python demo/demo.py --image img.jpg --question 'Is there a polyp?'")
        print("    python demo/demo.py --image img.jpg    # interactive mode")
        print("    python demo/demo.py --auto --n 5       # auto demo")
