"""
D5 Stage 1 — build the reason-first training set from ADE-Affordance.

Stage 0 showed that a text-only mapper from explanation to code tops out at 0.374 even on
held-out HUMAN explanations, so the reason text alone under-determines the type. Stage 1
therefore trains a model to produce the reason and the code *jointly from the image*, which
is the smallest change to the task that can beat that ceiling.

This script only emits records. Pixels are resolved at training time from --images_dir and
--seg_dir, so the same JSONL works for any crop policy.

Each record:
    {"image": "ADE_train_00000014", "action": "sit", "instance_id": 43,
     "code": 4, "explanation": "...", "consequence": "..."}

Codes are the canonical 7-way labels used everywhere else in this repo (see REL7). The
train and validation splits carry one annotator group, the test split three, and the
aggregation here matches analysis/stage0_mapper.py exactly (majority, ties to most severe)
so that Stage 1's labels and Stage 0's are the same quantity.

    python3 build_dataset.py                        # train+validation -> data/
    python3 build_dataset.py --splits test --out data/test.jsonl --verify_pixels
"""
import os
import re
import json
import glob
import argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "..", "..", "datasets", "ADE-Affordance-flat")

ACTION_POS = {"sit": 0, "run": 1, "grasp": 2}
FILE2CANON = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
REL7 = ["Positive", "FirmlyNegative", "ObjectNonFunctional", "PhysicalObstacle",
        "SociallyAwkward", "SociallyForbidden", "Dangerous"]
EXC = (2, 3, 4, 5, 6)


def _aggregate(votes):
    code, n = Counter(votes).most_common(1)[0]
    return code if n >= 2 else max(votes)


def parse_relationship(path):
    """instance_id -> {action: canonical code}, matching analysis/stage0_mapper.py."""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        groups = [[int(x) for x in re.findall(r"-?\d+", g)] for g in line.split("|")]
        if not groups or not groups[0]:
            continue
        iid = groups[0][0]
        groups[0] = groups[0][1:]
        groups = [g for g in groups if len(g) == 3]
        if not groups:
            continue
        out[iid] = {a: FILE2CANON[_aggregate([g[p] for g in groups])]
                    for a, p in ACTION_POS.items()}
    return out


def first_text(v):
    """exco text fields are a string or a list of annotator sentences."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        for t in v:
            if isinstance(t, str) and t.strip():
                return t.strip()
    return ""


def collect(split):
    records = []
    for rel_path in sorted(glob.glob(os.path.join(FLAT, split, "*_relationship.txt"))):
        image = os.path.basename(rel_path)[: -len("_relationship.txt")]
        rel = parse_relationship(rel_path)
        exco_path = rel_path.replace("_relationship.txt", "_exco.json")
        exco = json.load(open(exco_path)) if os.path.exists(exco_path) else {}
        for iid, per_action in rel.items():
            for action, code in per_action.items():
                expl, cons = "", ""
                if code in EXC:
                    v = exco.get(action, {}).get(str(iid)) or exco.get(action, {}).get(iid)
                    if v:
                        expl, cons = first_text(v.get("explanation")), first_text(v.get("consequence"))
                    # An exception with no annotator text cannot teach reason-first ordering.
                    if not expl:
                        continue
                records.append({"image": image, "action": action, "instance_id": iid,
                                "code": code, "explanation": expl, "consequence": cons})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,validation",
                    help="Comma list of ADE-Affordance-flat splits to pool.")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "train.jsonl"))
    ap.add_argument("--images_dir", default=None,
                    help="If given with --verify_pixels, drop records whose image is missing.")
    ap.add_argument("--seg_dir", default=None)
    ap.add_argument("--verify_pixels", action="store_true",
                    help="Keep only records whose image AND instance mask exist on disk.")
    ap.add_argument("--max_per_class", type=int, default=None,
                    help="Cap records per canonical code (the Positive class dominates).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = []
    for split in args.splits.split(","):
        got = collect(split.strip())
        print(f"{split.strip():12s} {len(got):7d} records")
        records += got

    if args.verify_pixels:
        if not (args.images_dir and args.seg_dir):
            raise SystemExit("--verify_pixels needs --images_dir and --seg_dir")
        before = len(records)
        have_img, have_seg = set(os.listdir(args.images_dir)), set(os.listdir(args.seg_dir))
        records = [r for r in records
                   if f"{r['image']}.jpg" in have_img and f"{r['image']}_seg.png" in have_seg]
        print(f"pixel check: kept {len(records)}/{before}"
              f" ({len({r['image'] for r in records})} distinct scenes)")

    if args.max_per_class:
        import random
        rng = random.Random(args.seed)
        by_code = defaultdict(list)
        for r in records:
            by_code[r["code"]].append(r)
        capped = []
        for c, rs in by_code.items():
            rng.shuffle(rs)
            capped += rs[: args.max_per_class]
        records = capped
        rng.shuffle(records)

    dist = Counter(r["code"] for r in records)
    print("\ncode distribution:")
    for c in range(7):
        n = dist.get(c, 0)
        print(f"  {c} {REL7[c]:22s} {n:7d}  {n / max(1, len(records)):6.1%}")
    n_exc = sum(dist.get(c, 0) for c in EXC)
    print(f"\ntotal {len(records)}  exceptions {n_exc} ({n_exc / max(1, len(records)):.1%})"
          f"  scenes {len({r['image'] for r in records})}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
