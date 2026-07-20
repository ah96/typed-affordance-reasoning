"""
SAM 2 + OOAL saliency region selection — the third selection strategy the submitted paper
cut. Lab PC only. SAM 2 proposes masks (segment-everything, same settings as Experiment B);
each (mask, action) pair is scored by the mean OOAL saliency inside the mask; top-K per
action are emitted in the replay schema, so local_vlms/replay_regions.py can query models on
them directly and experiment_b_agreement.py can score the result as mode "sam2_ooal".

    python3 rank_regions.py --images ../experiment_b_bundle/images \
        --heatmaps heatmaps_expb/ --K 3 --out regions_sam2_ooal_K3.jsonl

Heatmaps come from ooal_infer.py (<stem>__<action>.npy, image-sized). Masks smaller than
--min_area px are ignored (SAM speckle). Also writes mask pixel counts so downstream
analyses can weigh selections.
"""
import os
import json
import glob
import argparse

import numpy as np
from PIL import Image


def sam2_masks(image, ckpt, device):
    from ultralytics import SAM
    if not hasattr(sam2_masks, "_m"):
        sam2_masks._m = SAM(ckpt)
    res = sam2_masks._m(np.array(image.convert("RGB")), device=device,
                        retina_masks=True, verbose=False)
    r = res[0] if isinstance(res, (list, tuple)) else res
    masks = getattr(r, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return []
    data = masks.data
    data = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
    return [m > 0.5 for m in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--heatmaps", required=True, help="dir of <stem>__<action>.npy from ooal_infer.py")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--rank_by", choices=["ooal", "area"], default="ooal",
                    help="'area' gives the size-ranked baseline from the same masks")
    ap.add_argument("--min_area", type=int, default=400)
    ap.add_argument("--sam_ckpt", default="sam2.1_l.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="regions_sam2_ooal_K3.jsonl")
    args = ap.parse_args()

    heat = {}
    for p in glob.glob(os.path.join(args.heatmaps, "*.npy")):
        stem, action = os.path.basename(p)[:-4].rsplit("__", 1)
        heat.setdefault(stem, {})[action] = p

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            done.add(json.loads(line)["image"])

    paths = sorted(sum((glob.glob(os.path.join(args.images, e)) for e in ("*.jpg", "*.png")), []))
    with open(args.out, "a") as f:
        for p in paths:
            name = os.path.basename(p)
            stem = os.path.splitext(name)[0]
            if name in done or stem not in heat:
                continue
            img = Image.open(p)
            masks = [m for m in sam2_masks(img, args.sam_ckpt, args.device)
                     if m.sum() >= args.min_area]
            rows, rid = [], 0
            for action, hp in sorted(heat[stem].items()):
                h = np.load(hp)
                scored = []
                for m in masks:
                    if m.shape != h.shape:
                        continue
                    score = float(m.sum()) if args.rank_by == "area" else float(h[m].mean())
                    scored.append((score, m))
                scored.sort(key=lambda t: t[0], reverse=True)
                for score, m in scored[: args.K]:
                    ys, xs = np.where(m)
                    rows.append({"image": name, "region_id": rid, "action": action,
                                 "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                 "ooal_score": round(score, 4), "area": int(m.sum())})
                    rid += 1
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{name}: {len(masks)} masks -> {len(rows)} selections")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
