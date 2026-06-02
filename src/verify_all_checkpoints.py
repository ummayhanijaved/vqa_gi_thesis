#!/usr/bin/env python3
"""
=============================================================================
ALL STAGES — Complete Checkpoint Verification & Audit
=============================================================================

Verifies EVERY checkpoint in your VQA pipeline:

  STAGE 1 (Disease Classification)
    - stage1_best.pt                       (ResNet50 + MLP)

  STAGE 2 (Question Categorization)
    - best_model/                          (DistilBERT fine-tuned)

  STAGE 3 (Multimodal Fusion)
    - stage3_best.pt                       (Cross-Attn + Disease Gate)

  STAGE 4 PHASE 1 (MLP Answer Heads)
    - stage4_best.pt                       (6 MLP heads, joint training)

  STAGE 4 PHASE 2 (DistilBERT + YOLO)
    - stage4_revised/yes_no_best.pt
    - stage4_revised/single_choice_best.pt
    - stage4_revised/multi_choice_best.pt
    - stage4_revised/color_best.pt
    - stage4_revised/yolo_seg_finetuned/weights/best.pt
    - stage4_revised/yolo_det_finetuned/weights/best.pt

  STAGE 5 (T5 Verbalizer)
    - stage5_verbalizer/stage5_verbalizer_best.pt    (60M T5-small)
    - stage5_verbalizer/stage5_verbalizer_v2_best.pt (if retrained)

  EVAL CSVs (stored predictions)
    - All route0-5_eval.csv files

THIS IS READ-ONLY — never modifies anything.

USAGE:
    python verify_all_checkpoints.py
=============================================================================
"""
import os
import sys
import json
from datetime import datetime

import torch
import pandas as pd

PROJECT = os.path.expanduser("~/vqa_gi_thesis")
CKPT_BASE = os.path.join(PROJECT, "checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def human_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(b) < 1024.0: return f"{b:6.1f} {unit}"
        b /= 1024.0
    return f"{b:6.1f} TB"


def header(text, char="="):
    print(f"\n{char*72}\n  {text}\n{char*72}\n")


def file_info(path):
    """Get size + modified time for any file."""
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        "size": human_size(stat.st_size),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M"),
    }


def safe_load(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt, None
    except Exception as e:
        return None, str(e)


def get_state_dict(ckpt):
    """Find model state in a checkpoint dict, returns None if not found."""
    if not isinstance(ckpt, dict): return None
    for k in ["model_state", "model_state_dict", "state_dict", "model"]:
        if k in ckpt:
            v = ckpt[k]
            if isinstance(v, dict) and v: return v
    # Maybe ckpt IS the state dict directly?
    if all(hasattr(v, "shape") or hasattr(v, "numel") for v in ckpt.values()):
        return ckpt
    return None


def get_param_count(state_dict):
    if not state_dict: return 0
    return sum(t.numel() for t in state_dict.values()
               if hasattr(t, "numel"))


def get_metrics(ckpt):
    metrics = {}
    if not isinstance(ckpt, dict): return metrics
    for k in ["epoch", "best_epoch", "val_acc", "val_loss", "val_f1",
              "best_val_acc", "best_val_f1", "best_acc", "best_f1",
              "test_acc", "test_f1", "best_loss", "best_val_loss"]:
        if k in ckpt:
            v = ckpt[k]
            if hasattr(v, "item"):
                try: v = v.item()
                except Exception: pass
            metrics[k] = v
    return metrics


def architecture_hints(state_dict):
    """Inspect parameter names to identify architecture."""
    if not state_dict: return {}
    keys = list(state_dict.keys())
    hints = {
        "has_distilbert": any("distilbert" in k.lower() for k in keys),
        "has_resnet":     any("resnet" in k.lower() or "layer1" in k or
                               "layer2" in k or "conv1" in k for k in keys),
        "has_transformer": any("transformer" in k.lower() or "attention" in k.lower()
                                for k in keys),
        "has_classifier":  any("classifier" in k.lower() or "fc" in k.lower()
                                or "head" in k.lower() for k in keys),
        "has_projector":   any("projector" in k.lower() or "proj" in k.lower()
                                for k in keys),
        "has_disease":     any("disease" in k.lower() for k in keys),
        "has_cross_attn":  any("cross" in k.lower() and "attn" in k.lower()
                                for k in keys),
        "has_fusion":      any("fusion" in k.lower() for k in keys),
        "has_yolo":        any("yolo" in k.lower() or "darknet" in k.lower()
                                for k in keys),
        "n_layers":        len([k for k in keys if ".weight" in k]),
    }
    return hints


def find_head_output_size(state_dict):
    """Find the final classifier layer's output dimension."""
    if not state_dict: return None
    candidates = []
    for k, v in state_dict.items():
        if not hasattr(v, "shape") or v.dim() != 2: continue
        kl = k.lower()
        if ("head" in kl or "classifier" in kl or "fc" in kl) and "weight" in kl:
            candidates.append((k, v.shape[0]))
    if not candidates: return None
    # Pick the last one alphabetically (usually the final layer)
    candidates.sort()
    return candidates[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Stage-specific verifiers
# ─────────────────────────────────────────────────────────────────────────────
def verify_pytorch_ckpt(path, label, expected=None):
    """Generic PyTorch ckpt verifier with rich diagnostics."""
    print(f"\n  ─── {label} ───")
    print(f"     Path:    {path}")
    info = file_info(path)
    if not info:
        print(f"     Status:  ❌  MISSING")
        return {"status": "missing", "label": label, "path": path}

    print(f"     Size:    {info['size']}")
    print(f"     Modified:{info['modified']}")

    ckpt, err = safe_load(path)
    if err:
        print(f"     Status:  ❌  LOAD FAILED — {err[:100]}")
        return {"status": "load_failed", "label": label, "path": path,
                "error": err}

    msd = get_state_dict(ckpt)
    n_params = get_param_count(msd)
    metrics = get_metrics(ckpt)
    arch = architecture_hints(msd)
    head = find_head_output_size(msd)

    if msd is None:
        print(f"     Status:  ⚠️   NO MODEL STATE FOUND")
        print(f"     Keys:    {list(ckpt.keys()) if isinstance(ckpt, dict) else 'NOT A DICT'}")
        return {"status": "no_state", "label": label, "path": path}

    print(f"     Params:  {n_params:,}")

    # Architecture summary
    arch_str = []
    if arch.get("has_distilbert"):  arch_str.append("DistilBERT")
    if arch.get("has_resnet"):       arch_str.append("ResNet")
    if arch.get("has_transformer"):  arch_str.append("Transformer")
    if arch.get("has_projector"):    arch_str.append("Projector")
    if arch.get("has_classifier"):   arch_str.append("Classifier")
    if arch.get("has_disease"):      arch_str.append("Disease")
    if arch.get("has_cross_attn"):   arch_str.append("CrossAttn")
    if arch.get("has_fusion"):       arch_str.append("Fusion")
    if arch_str:
        print(f"     Arch:    {' + '.join(arch_str)}")

    if head:
        print(f"     Head:    {head[0]} → {head[1]} classes")

    if metrics:
        print(f"     Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"        {k:<15} = {v:.4f}")
            else:
                print(f"        {k:<15} = {v}")

    # Vocab if stored
    for vk in ["vocab", "vocab_single", "vocab_list", "classes", "labels"]:
        if isinstance(ckpt, dict) and vk in ckpt:
            v = ckpt[vk]
            if isinstance(v, (list, tuple)):
                print(f"     Vocab:   {len(v)} classes — first 5: {list(v)[:5]}")
                break

    # Expected vs actual checks
    issues = []
    if expected:
        if "n_classes" in expected and head:
            if head[1] != expected["n_classes"]:
                issues.append(f"⚠️  Head output {head[1]} != "
                               f"expected {expected['n_classes']}")
        if "min_params" in expected and n_params < expected["min_params"]:
            issues.append(f"⚠️  Only {n_params:,} params, "
                           f"expected ≥{expected['min_params']:,}")

    if issues:
        print(f"     ISSUES:")
        for i in issues:
            print(f"        {i}")
        return {"status": "issues", "label": label, "path": path,
                "issues": issues}

    print(f"     Status:  ✅  VALID")
    return {"status": "valid", "label": label, "path": path,
            "params": n_params, "metrics": metrics}


def verify_yolo(path, label, expected_task):
    print(f"\n  ─── {label} ───")
    print(f"     Path:    {path}")
    info = file_info(path)
    if not info:
        print(f"     Status:  ❌  MISSING")
        return {"status": "missing", "label": label, "path": path}

    print(f"     Size:    {info['size']}")
    print(f"     Modified:{info['modified']}")

    try:
        from ultralytics import YOLO
        model = YOLO(path)
        names = model.names if hasattr(model, "names") else {}
        task = getattr(model, "task", "unknown")
        n_params = sum(p.numel() for p in model.model.parameters())

        print(f"     Task:    {task}")
        print(f"     Params:  {n_params:,}")
        print(f"     Classes: {len(names)} — {list(names.values())[:5]}")

        issues = []
        if task != expected_task:
            issues.append(f"⚠️  Task '{task}' != expected '{expected_task}'")
        if issues:
            for i in issues: print(f"     ISSUES:  {i}")
            return {"status": "issues", "label": label, "issues": issues}
        print(f"     Status:  ✅  VALID")
        return {"status": "valid", "label": label, "params": n_params}
    except ImportError:
        print(f"     Status:  ⚠️   ultralytics not installed")
        return {"status": "no_ultralytics", "label": label}
    except Exception as e:
        print(f"     Status:  ❌  LOAD FAILED — {str(e)[:80]}")
        return {"status": "load_failed", "label": label, "error": str(e)}


def verify_stage2_folder(folder, label):
    print(f"\n  ─── {label} ───")
    print(f"     Path:    {folder}")
    if not os.path.exists(folder):
        print(f"     Status:  ❌  MISSING")
        return {"status": "missing", "label": label}

    if not os.path.isdir(folder):
        print(f"     Status:  ⚠️  Not a directory")
        return {"status": "wrong_type", "label": label}

    files = sorted(os.listdir(folder))
    print(f"     Files:   {len(files)} files")
    total_size = 0
    for fname in files[:10]:
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            total_size += sz
            print(f"        {fname:<35} {human_size(sz)}")
    print(f"     Total:   {human_size(total_size)}")

    # Try to load DistilBERT from this folder
    try:
        from transformers import DistilBertModel
        # Suppress the load report by capturing stderr temporarily
        m = DistilBertModel.from_pretrained(folder, local_files_only=True)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"     Params:  {n_params:,}")
        print(f"     Status:  ✅  VALID DistilBERT")
        return {"status": "valid", "label": label, "params": n_params}
    except Exception as e:
        print(f"     Status:  ⚠️   Cannot verify as DistilBERT: {str(e)[:80]}")
        return {"status": "unknown", "label": label}


def verify_eval_csv(path, label, expected_cols=None):
    print(f"\n  ─── {label} ───")
    print(f"     Path:    {path}")
    info = file_info(path)
    if not info:
        print(f"     Status:  ❌  MISSING")
        return {"status": "missing", "label": label}

    print(f"     Size:    {info['size']}")
    print(f"     Modified:{info['modified']}")

    try:
        df = pd.read_csv(path)
        print(f"     Rows:    {len(df):,}")
        print(f"     Columns: {list(df.columns)}")

        if "prediction" in df.columns:
            top = df["prediction"].astype(str).value_counts().head(3)
            print(f"     Top preds: {dict(top)}")

        if "ground_truth" in df.columns and "prediction" in df.columns:
            # Quick accuracy
            df_str = df.astype(str)
            same = (df_str["prediction"].str.lower().str.strip() ==
                    df_str["ground_truth"].str.lower().str.strip()).mean() * 100
            substr = df_str.apply(
                lambda r: r["prediction"].lower() in r["ground_truth"].lower()
                if r["prediction"] else False, axis=1).mean() * 100
            print(f"     Exact match: {same:.2f}%  Substring match: {substr:.2f}%")

        print(f"     Status:  ✅  READABLE")
        return {"status": "valid", "label": label, "rows": len(df)}
    except Exception as e:
        print(f"     Status:  ❌  READ FAILED — {str(e)[:80]}")
        return {"status": "load_failed", "label": label}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'█'*72}")
    print(f"  COMPLETE PIPELINE CHECKPOINT VERIFICATION")
    print(f"  Project: {PROJECT}")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*72}")

    all_results = []

    # ──── STAGE 1 ────────────────────────────────────────────────────────
    header("STAGE 1 — Disease Classification (ResNet50)")
    r = verify_pytorch_ckpt(
        os.path.join(CKPT_BASE, "stage1_best.pt"),
        "Stage 1: stage1_best.pt",
        expected={"min_params": 20_000_000})
    all_results.append(r)

    # ──── STAGE 2 ────────────────────────────────────────────────────────
    header("STAGE 2 — Question Categorization (DistilBERT)")
    r = verify_stage2_folder(
        os.path.join(CKPT_BASE, "best_model"),
        "Stage 2: best_model/")
    all_results.append(r)

    # Also check if there's a stage2_best.pt file
    r2 = verify_pytorch_ckpt(
        os.path.join(CKPT_BASE, "stage2_best.pt"),
        "Stage 2 (alt): stage2_best.pt")
    all_results.append(r2)

    # ──── STAGE 3 ────────────────────────────────────────────────────────
    header("STAGE 3 — Multimodal Fusion (Cross-Attn + Disease Gate)")
    r = verify_pytorch_ckpt(
        os.path.join(CKPT_BASE, "stage3_best.pt"),
        "Stage 3: stage3_best.pt",
        expected={"min_params": 50_000_000})
    all_results.append(r)

    # ──── STAGE 4 PHASE 1 ────────────────────────────────────────────────
    header("STAGE 4 PHASE 1 — MLP Answer Heads (joint training)")
    r = verify_pytorch_ckpt(
        os.path.join(CKPT_BASE, "stage4_best.pt"),
        "Stage 4 Phase 1: stage4_best.pt")
    all_results.append(r)

    # ──── STAGE 4 PHASE 2 — DistilBERT routes ────────────────────────────
    header("STAGE 4 PHASE 2 — DistilBERT Routes (0-3)")

    # Find ckpt dir
    s4_ckpt_dir = None
    for cand in [os.path.join(CKPT_BASE, "stage4_revised"),
                  os.path.join(CKPT_BASE, "stage4_revised", "stage4_revised")]:
        if os.path.exists(os.path.join(cand, "stage4_revised_yes_no_best.pt")):
            s4_ckpt_dir = cand
            break
    if s4_ckpt_dir is None:
        s4_ckpt_dir = os.path.join(CKPT_BASE, "stage4_revised")
    print(f"  (Using Stage 4 Phase 2 dir: {s4_ckpt_dir})")

    route_specs = {
        0: ("stage4_revised_yes_no_best.pt",        "Route 0: Yes/No",        2),
        1: ("stage4_revised_single_choice_best.pt", "Route 1: Single Choice", 50),
        2: ("stage4_revised_multi_choice_best.pt",  "Route 2: Multi Choice",  200),
        3: ("stage4_revised_color_best.pt",         "Route 3: Color",          13),
    }
    for route, (fname, label, n_cls) in route_specs.items():
        r = verify_pytorch_ckpt(
            os.path.join(s4_ckpt_dir, fname), label,
            expected={"n_classes": n_cls, "min_params": 10_000_000})
        all_results.append(r)

    # ──── STAGE 4 PHASE 2 — YOLO routes ──────────────────────────────────
    header("STAGE 4 PHASE 2 — YOLO Routes (4-5)")
    r = verify_yolo(
        os.path.join(s4_ckpt_dir, "yolo_seg_finetuned", "weights", "best.pt"),
        "Route 4: YOLO-Seg (Location)",
        expected_task="segment")
    all_results.append(r)
    r = verify_yolo(
        os.path.join(s4_ckpt_dir, "yolo_det_finetuned", "weights", "best.pt"),
        "Route 5: YOLO-Det (Count)",
        expected_task="detect")
    all_results.append(r)

    # ──── STAGE 5 ────────────────────────────────────────────────────────
    header("STAGE 5 — T5-small Verbalizer")
    r = verify_pytorch_ckpt(
        os.path.join(CKPT_BASE, "stage5_verbalizer", "stage5_verbalizer_best.pt"),
        "Stage 5: stage5_verbalizer_best.pt (v1)",
        expected={"min_params": 50_000_000})
    all_results.append(r)
    r = verify_pytorch_ckpt(
        os.path.join(CKPT_BASE, "stage5_verbalizer", "stage5_verbalizer_v2_best.pt"),
        "Stage 5: stage5_verbalizer_v2_best.pt (v2, if retrained)")
    all_results.append(r)

    # ──── EVAL CSVs ──────────────────────────────────────────────────────
    header("STAGE 4 EVAL CSVs (saved predictions)")
    log_dir = os.path.join(PROJECT, "logs", "stage4_revised")
    eval_csvs = {
        0: "route0_yes_no_eval.csv",
        1: "route1_single_choice_eval.csv",
        2: "route2_multi_choice_eval.csv",
        3: "route3_color_eval.csv",
        4: "route4_location_yolo_eval.csv",
        5: "route5_count_yolo_eval.csv",
    }
    for route, fname in eval_csvs.items():
        r = verify_eval_csv(
            os.path.join(log_dir, fname),
            f"Route {route}: {fname}")
        all_results.append(r)

    # ──── FINAL SUMMARY ──────────────────────────────────────────────────
    header("SUMMARY TABLE", char="█")

    n_valid = n_missing = n_issues = n_other = 0
    print(f"  {'Component':<55} {'Status':<15}")
    print(f"  {'-'*55} {'-'*15}")
    for r in all_results:
        if not r: continue
        label = r.get("label", "?")[:53]
        status = r.get("status", "?")
        emoji = {"valid": "✅", "missing": "❌", "issues": "⚠️ ",
                  "load_failed": "❌", "no_state": "⚠️ ",
                  "no_ultralytics": "⚠️ ", "unknown": "❓",
                  "wrong_type": "⚠️ "}.get(status, "❓")
        print(f"  {label:<55} {emoji} {status}")
        if status == "valid": n_valid += 1
        elif status == "missing": n_missing += 1
        elif status in ["issues", "load_failed", "wrong_type"]: n_issues += 1
        else: n_other += 1

    print(f"\n  {'='*72}")
    print(f"   ✅  VALID:    {n_valid}")
    print(f"   ❌  MISSING:  {n_missing}")
    print(f"   ⚠️   ISSUES:   {n_issues}")
    print(f"   ❓  OTHER:    {n_other}")
    print(f"  {'='*72}\n")

    # ──── FAQ section ────────────────────────────────────────────────────
    header("FREQUENTLY ASKED QUESTIONS", char="─")
    print(f"""
  Q: What about the 'UNEXPECTED keys' warnings?
  
     The 'vocab_layer_norm', 'vocab_transform', 'vocab_projector' warnings
     mean that the downloaded DistilBERT-base file includes an MLM head from
     its pretraining, which we don't use for VQA. This is EXPECTED — the
     warning explicitly says:
       "can be ignored when loading from different task/architecture"
     
     Your VQA-fine-tuned weights load correctly. These are just unused parts
     of the original Hugging Face pretrained file.

  Q: How can I tell if my trained models are intact?

     Check:
     1. Each route's val_acc/val_loss metric matches what you reported earlier
        (e.g. Route 0 = 88.65%, Route 2 = 84.20% sample-F1)
     2. Modification dates show files were trained around the expected time
     3. Parameter counts match expected architecture sizes
     4. Vocab sizes match (2/50/200/13 for routes 0/1/2/3)

  Q: What if a checkpoint shows ISSUES?

     - Wrong n_classes → checkpoint may belong to a different route
     - Low params → architecture changed during loading
     - Old modification date → may have been overwritten

  Q: Should I re-train any models?

     Only if a checkpoint shows MISSING or ISSUES status above. Otherwise,
     your models are fine — any quality concerns are from Stage 5 verbalizer
     or evaluation methodology, not from corrupted Stage 4 checkpoints.
""")


if __name__ == "__main__":
    main()
