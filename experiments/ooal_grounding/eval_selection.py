"""
Ground-truth evaluation on AGD20K's test heatmaps. Two questions:

1. Checkpoint sanity — do the OOAL saliency maps score in the ballpark of the OOAL paper on
   the standard metrics (KLD down / SIM up / NSS up)? If yes, the checkpoints and the adapter
   are wired correctly.
2. Selection quality — of the ground-truth heatmap mass, how much falls inside the top-K
   regions each selection strategy picks (GT recall@K)? Compares area-ranked vs OOAL-ranked
   SAM masks with real spatial GT, which the submitted paper could not do.

AGD20K layout expected (defaults for the local copy):
  <root>/testset/egocentric/<affordance>/<object>/<image>.jpg
  <root>/testset/GT/<affordance>/<object>/<image>.png       (8-bit heatmap)

    python3 eval_selection.py --split Seen --metrics             # question 1 (needs heatmaps)
    python3 eval_selection.py --split Seen --recall \
        --regions_area regions_area.jsonl --regions_ooal regions_sam2_ooal_K3.jsonl

For --metrics, first dump heatmaps for the egocentric test images with ooal_infer.py using
the affordance names found under testset/egocentric/ (they are the class-dir names).
"""
import os
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
AGD_ROOT = os.path.join(HERE, "..", "..", "datasets", "AGD20K")


def norm_dist(m):
    m = np.clip(m.astype(np.float64), 0, None)
    s = m.sum()
    return m / s if s > 0 else np.full_like(m, 1.0 / m.size)


def kld(pred, gt):
    p, g = norm_dist(pred), norm_dist(gt)
    eps = 1e-12
    return float((g * np.log(eps + g / (p + eps))).sum())


def sim(pred, gt):
    return float(np.minimum(norm_dist(pred), norm_dist(gt)).sum())


def nss(pred, gt):
    p = (pred - pred.mean()) / (pred.std() + 1e-12)
    fix = gt > (0.1 * gt.max() if gt.max() > 0 else 0)
    return float(p[fix].mean()) if fix.any() else 0.0


def gt_pairs(root, split):
    """yield (affordance, object, image_stem, img_path, gt_path)"""
    base = os.path.join(root, split, "testset")
    for gt in glob.glob(os.path.join(base, "GT", "*", "*", "*.png")):
        parts = gt.split(os.sep)
        aff, obj, name = parts[-3], parts[-2], os.path.splitext(parts[-1])[0]
        img = os.path.join(base, "egocentric", aff, obj, name + ".jpg")
        if os.path.exists(img):
            yield aff, obj, name, img, gt


def run_metrics(args):
    per_aff = defaultdict(list)
    n = 0
    for aff, obj, name, img, gt in gt_pairs(args.root, args.split):
        hp = os.path.join(args.heatmaps, f"{name}__{aff}.npy")
        if not os.path.exists(hp):
            continue
        pred = np.load(hp)
        g = np.asarray(Image.open(gt)).astype(np.float64)
        if pred.shape != g.shape:
            pred = np.asarray(Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8))
                              .resize((g.shape[1], g.shape[0]))) / 255.0
        per_aff[aff].append((kld(pred, g), sim(pred, g), nss(pred, g)))
        n += 1
    if not n:
        raise SystemExit("no (heatmap, GT) pairs found — run ooal_infer.py on the egocentric "
                         "test images first (heatmap name pattern <stem>__<affordance>.npy)")
    allv = [v for vs in per_aff.values() for v in vs]
    K, S, N = (np.mean([v[i] for v in allv]) for i in range(3))
    print(f"{args.split}: n={n}  KLD {K:.3f}  SIM {S:.3f}  NSS {N:.3f}   "
          f"(OOAL paper reports ~1.07/0.46/1.14 Seen, ~1.30/0.38/0.94 Unseen)")
    return {"n": n, "KLD": K, "SIM": S, "NSS": N,
            "per_affordance": {a: {"n": len(v), "KLD": float(np.mean([x[0] for x in v])),
                                   "SIM": float(np.mean([x[1] for x in v])),
                                   "NSS": float(np.mean([x[2] for x in v]))}
                               for a, v in sorted(per_aff.items())}}


def run_recall(args):
    gt_ix = {}
    for aff, obj, name, img, gt in gt_pairs(args.root, args.split):
        gt_ix[(name, aff)] = gt
    out = {}
    for label, path in (("area", args.regions_area), ("ooal", args.regions_ooal)):
        if not path:
            continue
        sel = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            stem = os.path.splitext(r["image"])[0]
            sel[(stem, r["action"])].append(r["bbox"])
        recalls = []
        for key, boxes in sel.items():
            if key not in gt_ix:
                continue
            g = np.asarray(Image.open(gt_ix[key])).astype(np.float64)
            tot = g.sum()
            if tot <= 0:
                continue
            cover = np.zeros_like(g, dtype=bool)
            for x1, y1, x2, y2 in boxes:
                cover[y1:y2 + 1, x1:x2 + 1] = True
            recalls.append(g[cover].sum() / tot)
        out[label] = {"n": len(recalls), "gt_recall_at_K": float(np.mean(recalls)) if recalls else None}
        print(f"{label}: n={len(recalls)}  GT-mass recall@K {out[label]['gt_recall_at_K']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=AGD_ROOT)
    ap.add_argument("--split", choices=["Seen", "Unseen"], default="Seen")
    ap.add_argument("--heatmaps", default="heatmaps_agd")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--regions_area", default=None)
    ap.add_argument("--regions_ooal", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "out_eval.json"))
    args = ap.parse_args()

    res = {}
    if args.metrics:
        res["metrics"] = run_metrics(args)
    if args.recall:
        res["recall"] = run_recall(args)
    if res:
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    else:
        print("nothing to do: pass --metrics and/or --recall")


if __name__ == "__main__":
    main()
