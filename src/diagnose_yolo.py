"""
Diagnostic for YOLO dataset preparation failure.
Run this BEFORE retraining YOLO to see what the cache looks like
and whether img_ids match actual files on disk.
"""
import os, torch
from collections import Counter

HOME    = os.path.expanduser("~")
cache   = torch.load(
    os.path.join(HOME, "vqa_gi_thesis", "cache",
                 "stage3_features", "stage3_cache_train.pt"),
    weights_only=False)

imgs_dir = os.path.join(HOME, "data", "kvasir_raw", "images")

print(f"Cache records    : {len(cache):,}")
print(f"Image directory  : {imgs_dir}")

actual_files = set(os.listdir(imgs_dir))
print(f"Images in dir    : {len(actual_files):,}")

# Sample a few cache records
print("\n" + "="*60)
print("FIRST 5 CACHE RECORDS")
print("="*60)
for i, r in enumerate(cache[:5]):
    keys = list(r.keys())
    print(f"\n  Record {i}: keys={keys}")
    print(f"    route   = {r.get('route')}")
    img_id = r.get('img_id', r.get('image_id', 'MISSING'))
    print(f"    img_id  = {str(img_id)[:80]!r}")
    print(f"    answer  = {str(r.get('answer',''))[:60]!r}")

# Sample actual filenames
print("\n" + "="*60)
print("FIRST 5 ACTUAL FILENAMES")
print("="*60)
for f in sorted(actual_files)[:5]:
    print(f"  {f!r}")

# Count matches
print("\n" + "="*60)
print("MATCHING ANALYSIS")
print("="*60)
matched = 0
unmatched_examples = []
for r in cache[:5000]:
    img_id = r.get("img_id", r.get("image_id", ""))
    if not img_id:
        continue
    img_id = str(img_id)
    found = False
    for ext in ["", ".jpg", ".jpeg", ".png"]:
        if (img_id + ext) in actual_files:
            found = True
            break
    if found:
        matched += 1
    elif len(unmatched_examples) < 5:
        unmatched_examples.append(img_id)

total_with_id = sum(1 for r in cache[:5000]
                    if r.get("img_id", r.get("image_id")))
print(f"  Sampled: first 5000 cache records")
print(f"  With img_id: {total_with_id:,}")
print(f"  Matched on disk: {matched:,}")
print(f"  Match rate: {100*matched/max(total_with_id,1):.1f}%")
if unmatched_examples:
    print(f"\n  Unmatched examples:")
    for u in unmatched_examples:
        print(f"    {u!r}")

# Route distribution
print("\n" + "="*60)
print("ROUTE DISTRIBUTION")
print("="*60)
routes = Counter(r.get("route") for r in cache)
for r in sorted(routes.keys()):
    print(f"  Route {r}: {routes[r]:,}")
