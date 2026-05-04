"""Quick inspect of FusionExtractor API on your machine."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/vqa_gi_thesis/src"))
from stage4_revised import CFG, FusionExtractor
ex = FusionExtractor(CFG["stage3_ckpt"])
print("\nFusionExtractor public methods:")
for name in sorted(dir(ex)):
    if not name.startswith("_"):
        attr = getattr(ex, name)
        kind = "method" if callable(attr) else "attribute"
        print(f"  {kind}: {name}")

import sys, os, inspect
sys.path.insert(0, os.path.expanduser("~/vqa_gi_thesis/src"))
from stage4_revised import CFG, FusionExtractor
 
ex = FusionExtractor(CFG["stage3_ckpt"])
print("\nFusionExtractor.extract() signature:")
print(f"  {inspect.signature(ex.extract)}")
print()
print("Source code:")
try:
    print(inspect.getsource(ex.extract))
except Exception as e:
    print(f"  (could not read source: {e})")
 

"""Inspect TextPreprocessor's actual API."""
import sys, os, inspect
sys.path.insert(0, os.path.expanduser("~/vqa_gi_thesis/src"))
from stage4_revised import TextPreprocessor

tp = TextPreprocessor()
print("\nTextPreprocessor public methods:")
for name in sorted(dir(tp)):
    if not name.startswith("_"):
        attr = getattr(tp, name)
        kind = "method" if callable(attr) else "attribute"
        print(f"  {kind}: {name}")

# Show signatures of all callables
print("\nMethod signatures:")
for name in sorted(dir(tp)):
    if not name.startswith("_") and callable(getattr(tp, name)):
        try:
            sig = inspect.signature(getattr(tp, name))
            print(f"  {name}{sig}")
        except (ValueError, TypeError):
            pass