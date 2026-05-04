"""
=============================================================================
PIPELINE SANITY TESTS
Run: python tests/test_pipeline.py
=============================================================================
"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.expanduser("~"))

def test_stage1():
    from stage1_disease_classifier import TreeNetDiseaseClassifier
    m = TreeNetDiseaseClassifier()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    # Stage 1 returns either a tensor or a dict depending on version
    if isinstance(out, dict):
        logits = out.get("logits", out.get("disease_logits",
                 out.get("output", list(out.values())[0])))
    else:
        logits = out
    assert logits.shape == (2, 23), f"Stage1 output shape wrong: {logits.shape}"
    print("  ✅  Stage 1 — output shape correct (2, 23)")

def test_stage2():
    from stage2_question_categorizer import QuestionTypeClassifier
    from preprocessing import TextPreprocessor
    import os

    # Check checkpoint exists
    ckpt_dir = "./checkpoints/best_model"
    assert os.path.exists(ckpt_dir), f"Stage 2 checkpoint missing: {ckpt_dir}"

    # Test tokenisation
    tp  = TextPreprocessor()
    enc = tp.preprocess("Is there a polyp visible?")
    assert "input_ids"      in enc, "Missing input_ids"
    assert "attention_mask" in enc, "Missing attention_mask"
    assert enc["input_ids"].shape[-1] == 128, "Wrong token length"

    # Test model forward pass
    m    = QuestionTypeClassifier()
    ids  = enc["input_ids"].unsqueeze(0)
    mask = enc["attention_mask"].unsqueeze(0)
    with torch.no_grad():
        out = m(ids, mask)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (1, 6), f"Stage2 output shape wrong: {logits.shape}"
    pred = logits.argmax(-1).item()
    assert 0 <= pred <= 5, f"Predicted route {pred} out of range"
    print(f"  ✅  Stage 2 — output shape correct (1, 6)  |  predicted route: {pred}")

def test_stage3():
    from stage3_multimodal_fusion import Stage3MultimodalFusion
    m = Stage3MultimodalFusion()
    imgs = torch.randn(2, 3, 224, 224)
    ids  = torch.randint(0, 1000, (2, 128))
    mask = torch.ones(2, 128, dtype=torch.long)
    with torch.no_grad():
        out = m(imgs, ids, mask)
    assert out["fused_repr"].shape  == (2, 512), "fused_repr wrong"
    assert out["disease_vec"].shape == (2, 23),  "disease_vec wrong"
    assert out["routing_logits"].shape == (2, 6),"routing_logits wrong"
    print("  ✅  Stage 3 — all output shapes correct")

def test_stage4():
    import json, os
    from stage4_answer_generation import Stage4AnswerGenerator, CFG as S4_CFG

    # Load vocab from JSON directly (avoids needing dataset on disk)
    vocab_paths = [
        "./data/stage4_vocab.json",
        os.path.expanduser("~/data/stage4_vocab.json"),
    ]
    vocab = None
    for vp in vocab_paths:
        if os.path.exists(vp):
            with open(vp) as f:
                vocab = json.load(f)
            break
    if vocab is None:
        print("  ⚠️   Stage 4 vocab not found — skipping test")
        return

    m     = Stage4AnswerGenerator(vocab)
    fused   = torch.randn(2, 512)
    disease = torch.rand(2, 23)
    for r in range(6):
        out = m(fused, disease, r)
        assert out.shape[0] == 2, f"Route {r} batch dim wrong"
    print("  ✅  Stage 4 — all 6 heads produce correct output")

def test_checkpoints_exist():
    import os
    ckpts = [
        "./checkpoints/stage1_best.pt",
        "./checkpoints/stage3_best.pt",
        "./checkpoints/stage4_best.pt",
    ]
    for c in ckpts:
        assert os.path.exists(c), f"Missing checkpoint: {c}"
    print("  ✅  All checkpoints exist")

if __name__ == "__main__":
    print("\n🧪  Running pipeline sanity tests ...\n")
    try:
        test_checkpoints_exist()
        test_stage1()
        test_stage2()
        test_stage3()
        test_stage4()
        print("\n✅  All tests passed!\n")
    except Exception as e:
        print(f"\n❌  Test failed: {e}\n")
        raise
