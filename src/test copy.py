import torch, os, sys
from datasets import load_from_disk
from collections import Counter
sys.path.insert(0, os.path.expanduser("~/vqa_gi_thesis/src"))
from stage4_revised import CFG

raw = load_from_disk(CFG["data_dir"])["train"]

# Build (question, answer) → list of img_ids from HF
key_to_imgs = {}
for s in raw:
    k = (s["question"].strip().lower(),
         s.get("answer","").strip().lower())
    key_to_imgs.setdefault(k, []).append(
        str(s.get("img_id", s.get("image_id",""))))

uniq = sum(1 for v in key_to_imgs.values() if len(v) == 1)
dup  = sum(1 for v in key_to_imgs.values() if len(v)  > 1)
print(f"unique (q,a) keys: {uniq:,}")
print(f"colliding keys:    {dup:,}")

# How many train cache records can be uniquely matched?
c = torch.load(os.path.expanduser(
    "~/vqa_gi_thesis/cache/stage3_features/stage3_cache_train.pt"),
    map_location="cpu", weights_only=False)
matched = uniqmatch = 0
for r in c:
    k = (r["question"].strip().lower(), r["answer"].strip().lower())
    imgs = key_to_imgs.get(k, [])
    if imgs: matched += 1
    if len(imgs) == 1: uniqmatch += 1
print(f"cache records matched at all:        {matched:,} / {len(c):,}")
print(f"cache records uniquely matched:      {uniqmatch:,} / {len(c):,}")
