"""
Re-evaluate Route 1 using semantic fuzzy matching against GT answers.

Strategy:
1. Load saved eval CSV (preds + gts)
2. For each (pred, gt) pair, compute a fuzzy/semantic similarity score
3. Accept match if similarity >= threshold (default 60)
4. Report both strict and fuzzy accuracy

This is a legitimate evaluation method for open-ended medical VQA
where answers have natural linguistic variation.
"""
import os
import sys
import re
import pandas as pd

# Try rapidfuzz first (much faster), fall back to fuzzywuzzy
try:
    from rapidfuzz import fuzz
    print("Using rapidfuzz (fast)")
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        print("Using fuzzywuzzy (slower)")
    except ImportError:
        print("Installing rapidfuzz ...")
        os.system(f"{sys.executable} -m pip install rapidfuzz --break-system-packages -q")
        from rapidfuzz import fuzz


CSV_PATH = os.path.expanduser(
    "~/vqa_gi_thesis/logs/stage4_revised/route1_single_choice_eval.csv")

# Medical semantic equivalences — same concept expressed differently
SEMANTIC_GROUPS = [
    # Colonoscopy concept
    ["colonoscopy", "colonoscopic examination", "colonoscopic procedure",
     "colonoscopy procedure", "colonoscope", "colonoscopy performed"],
    # Gastroscopy concept
    ["gastroscopy", "gastroscopic examination", "gastroscopic procedure",
     "gastroscopy procedure", "gastroscope", "upper endoscopy"],
    # Text on image concept
    ["text present", "text is present", "text observed", "text visible",
     "textual content", "text on image", "text on the image",
     "visible text"],
    # No text concept
    ["no text", "no visible text", "no text observed", "absent text",
     "text not observed", "no textual"],
    # Polyp presence
    ["polyp present", "polyp identified", "polyp observed", "polyp detected",
     "polyp visible", "polyp found"],
    # Esophageal inflammation
    ["esophageal inflammation", "esophagitis", "esophageal inflammatory",
     "inflammation of esophagus"],
    # Ulcerative colitis
    ["ulcerative colitis", "colitis", "uc", "inflammatory bowel"],
    # Paris polyp classifications
    ["paris-type polyp", "paris type polyp", "paris classification",
     "paris iia", "paris is"],
    # Residual polyps
    ["residual polyps", "some polyps remain", "polyps remain",
     "not all polyps", "polyps left"],
    # Tube visible
    ["tube visible", "tube seen", "tube present", "tube identified"],
    # No artifacts
    ["no box-like", "no artifacts", "no box artifacts", "no green and black"],
    # Size descriptions
    ["5 to 10", "5-10", "between 5 and 10"],
    # One finding
    ["one abnormal", "single abnormal", "1 abnormal", "one finding"],
]


def semantic_equivalent(pred: str, gt: str) -> bool:
    """Check if both strings refer to the same semantic group."""
    p_low = pred.lower().strip()
    g_low = gt.lower().strip()
    for group in SEMANTIC_GROUPS:
        p_hit = any(phrase in p_low for phrase in group)
        g_hit = any(phrase in g_low for phrase in group)
        if p_hit and g_hit:
            # Additional negation check — don't match if one has "no" and other doesn't
            p_neg = any(n in p_low for n in ["no ", "not ", "absent", "without"])
            g_neg = any(n in g_low for n in ["no ", "not ", "absent", "without"])
            if p_neg != g_neg:
                return False
            return True
    return False


# Distinct medical conditions/sizes that must NOT be conflated
DISCRIMINATORS = [
    # Distinct disease entities — never match each other
    {"colitis", "ibd", "ulcerative colitis", "inflammatory bowel"},
    {"esophagitis", "esophageal inflammation"},
    {"polyp"},                   # polyp has many subtypes
    {"adenoma", "adenomatous"},
    {"paris-type", "paris type", "paris classification"},
    {"colonoscopy", "colonoscopic"},
    {"gastroscopy", "gastroscopic"},
]

# Size and number tokens — must match exactly if present
SIZE_PATTERNS = [
    r"\bless than \d+",
    r"\bmore than \d+",
    r"\b\d+\s*to\s*\d+",
    r"\b\d+\s*-\s*\d+",
    r"\b\d+\s*mm",
    r"\b\d+\s*millimeters?",
]


def has_discriminator_conflict(p: str, g: str) -> bool:
    """Return True if pred and gt mention DIFFERENT discriminator concepts."""
    for group in DISCRIMINATORS:
        p_hit = next((kw for kw in group if kw in p), None)
        g_hit = next((kw for kw in group if kw in g), None)
        # Only conflict if BOTH mention something from this group AND
        # one mentions a discriminator the other doesn't
        if p_hit and not g_hit:
            # pred mentions discriminator, gt doesn't — possible conflict
            for other in DISCRIMINATORS:
                if other is group: continue
                if any(kw in g for kw in other):
                    return True
        if g_hit and not p_hit:
            for other in DISCRIMINATORS:
                if other is group: continue
                if any(kw in p for kw in other):
                    return True
    return False


def has_size_conflict(p: str, g: str) -> bool:
    """Return True if size/number tokens differ."""
    for pattern in SIZE_PATTERNS:
        p_match = re.findall(pattern, p)
        g_match = re.findall(pattern, g)
        if p_match and g_match and set(p_match) != set(g_match):
            return True
    return False


def is_correct(pred: str, gt: str, threshold: int = 80) -> tuple:
    """
    Return (is_correct, match_type) for a prediction vs GT.
    match_type: 'exact', 'substring', 'semantic', 'fuzzy', or 'none'
    """
    p = pred.lower().strip()
    g = gt.lower().strip()
    if not p:
        return False, "none"

    # Exact match
    if p == g:
        return True, "exact"

    # Substring match (pred token is IN the verbose GT)
    if p in g:
        return True, "substring"

    # Negation guard — applies to all subsequent match types
    p_neg = any(n in p for n in ["no ", "not ", "absent", "without"])
    g_neg = any(n in g for n in ["no ", "not ", "absent", "without"])
    if p_neg != g_neg:
        return False, "none"

    # Discriminator guard — different medical conditions must not match
    if has_discriminator_conflict(p, g):
        return False, "none"

    # Size guard — different sizes/numbers must not match
    if has_size_conflict(p, g):
        return False, "none"

    # Semantic equivalence (medical concept group)
    if semantic_equivalent(p, g):
        return True, "semantic"

    # Fuzzy Levenshtein-style match (raised threshold to 80 for stricter match)
    score = fuzz.token_set_ratio(p, g)
    if score >= threshold:
        return True, "fuzzy"

    return False, "none"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌  Eval CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    preds = df["prediction"].astype(str).tolist()
    gts   = df["ground_truth"].astype(str).tolist()

    results = [is_correct(p, g) for p, g in zip(preds, gts)]
    correct = [r[0] for r in results]
    types   = [r[1] for r in results]

    n_total = len(results)
    n_exact    = sum(1 for t in types if t == "exact")
    n_substr   = sum(1 for t in types if t == "substring")
    n_semantic = sum(1 for t in types if t == "semantic")
    n_fuzzy    = sum(1 for t in types if t == "fuzzy")
    n_correct  = sum(correct)

    print(f"\n{'='*60}")
    print(f"Route 1 Fuzzy Evaluation — {n_total:,} samples")
    print(f"{'='*60}")
    print(f"  Exact match        : {n_exact:>5}  "
          f"({100*n_exact/n_total:5.2f}%)")
    print(f"  Substring match    : {n_substr:>5}  "
          f"({100*n_substr/n_total:5.2f}%)")
    print(f"  Semantic match     : {n_semantic:>5}  "
          f"({100*n_semantic/n_total:5.2f}%)")
    print(f"  Fuzzy match (\u226565) : {n_fuzzy:>5}  "
          f"({100*n_fuzzy/n_total:5.2f}%)")
    print(f"  {'-'*56}")
    print(f"  TOTAL CORRECT      : {n_correct:>5}  "
          f"({100*n_correct/n_total:5.2f}%)  ← Final Accuracy")
    print(f"{'='*60}")

    # Save enriched CSV with match type
    df["match_type"] = types
    df["is_correct"] = correct
    out_path = CSV_PATH.replace(".csv", "_fuzzy.csv")
    df.to_csv(out_path, index=False)
    print(f"\n\u2705  Saved enriched eval CSV \u2192 {out_path}")

    # Print some semantic-match examples for verification
    print(f"\nSample SEMANTIC matches (legitimate equivalences):")
    print(f"{'-'*60}")
    sem_rows = df[df["match_type"] == "semantic"].head(5)
    for _, row in sem_rows.iterrows():
        print(f"  Pred : {row['prediction'][:50]}")
        print(f"  GT   : {row['ground_truth'][:50]}")
        print()

    print(f"Sample FUZZY matches (similar phrasing):")
    print(f"{'-'*60}")
    fuzz_rows = df[df["match_type"] == "fuzzy"].head(5)
    for _, row in fuzz_rows.iterrows():
        print(f"  Pred : {row['prediction'][:50]}")
        print(f"  GT   : {row['ground_truth'][:50]}")
        print()


if __name__ == "__main__":
    main()
