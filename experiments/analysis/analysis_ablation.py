"""
D3a — same-weights reasoning ablation: Qwen3-VL-8B Instruct vs Thinking on Experiment A.

The Thinking variant drops a few calls (long <think> blocks that never yield parseable JSON),
so scoring each file on its own rows would compare the two variants on different item sets.
Everything here is therefore computed on the PAIRED subset both variants answered, and the
Instruct-vs-Thinking difference gets its own image-clustered bootstrap CI.

    python3 analysis_ablation.py
    python3 analysis_ablation.py --boot 5000
"""
import json
import os
import random
import argparse
from collections import defaultdict

from analysis_confusion import metrics, axis_breakdown, bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "experiment_a", "results")
INSTRUCT, THINKING = "qwen3_vl_8b_instruct", "qwen3_vl_8b_thinking"


def load_map(indir, name):
    """(image, action, region) -> (image, gt, pred), valid rows only."""
    out = {}
    for line in open(os.path.join(indir, f"raw_{name}.jsonl")):
        r = json.loads(line)
        if r["gt"] is None or not (0 <= r["pred"] <= 6):
            continue
        out[(r["image"], r["action"], r["region_id"])] = (r["image"], r["gt"], r["pred"])
    return out


def paired_diff(by_img, imgs, nboot, seed=0):
    """Bootstrap CI for (thinking - instruct), resampling images to respect clustering."""
    def diff(sample):
        mi = metrics([(g, a) for g, a, _ in sample])
        mt = metrics([(g, b) for g, _, b in sample])
        return {k: mt[k] - mi[k] for k in ("Type", "Detect", "mAcc7", "mAcc3")}

    obs = diff([x for i in imgs for x in by_img[i]])
    rng = random.Random(seed)
    dists = defaultdict(list)
    for _ in range(nboot):
        s = [x for i in rng.choices(imgs, k=len(imgs)) for x in by_img[i]]
        for k, v in diff(s).items():
            dists[k].append(v)
    out = {}
    for k, d in dists.items():
        d.sort()
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d))]
        out[k] = {"diff": obs[k], "ci": [lo, hi], "significant": lo > 0 or hi < 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "ablation_d3a.json"))
    args = ap.parse_args()

    ins, thi = load_map(args.dir, INSTRUCT), load_map(args.dir, THINKING)
    shared = sorted(ins.keys() & thi.keys())
    res = {"n_instruct_total": len(ins), "n_thinking_total": len(thi),
           "n_paired": len(shared), "n_dropped_by_thinking": len(ins) - len(shared),
           "per_variant": {}}
    print(f"instruct {len(ins)} rows | thinking {len(thi)} rows | paired {len(shared)}"
          f" | dropped by thinking {len(ins) - len(shared)}\n")

    for label, m in [(INSTRUCT, ins), (THINKING, thi)]:
        rows = [m[k] for k in shared]
        pairs = [(g, p) for _, g, p in rows]
        mt, ci = metrics(pairs), bootstrap(rows, args.boot)
        ab = axis_breakdown(pairs)
        res["per_variant"][label] = {"metrics": mt, "ci": ci, "axis": ab}
        print(f"=== {label} (paired n={len(rows)}) ===")
        for k in ("Type", "Detect"):
            print(f"  {k:7s} {mt[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
        for ax in ("all", "functional", "social", "danger"):
            a = ab[ax]
            print(f"    {ax:11s} n={a['n']:4d} exact {a['exact']:.3f}  "
                  f"right-axis {a['right_axis_wrong_type']:.3f}  "
                  f"wrong-axis {a['wrong_axis']:.3f}  missed {a['missed']:.3f}")
        print()

    by_img = defaultdict(list)
    for k in shared:
        by_img[k[0]].append((ins[k][1], ins[k][2], thi[k][2]))
    imgs = sorted(by_img)
    res["paired_difference"] = paired_diff(by_img, imgs, args.boot)
    print("=== paired difference (thinking - instruct) ===")
    for k in ("Type", "Detect"):
        d = res["paired_difference"][k]
        flag = "SIGNIFICANT" if d["significant"] else "n.s. (CI spans 0)"
        print(f"  {k:7s} {d['diff']:+.3f}  [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  {flag}")

    # Does chain-of-thought specifically restore the social axis it collapsed?
    allrows = [x for i in imgs for x in by_img[i]]
    soc = [x for x in allrows if x[0] in (4, 5)]
    res["social_axis"] = {}
    print("\n=== social axis (GT codes 4,5) ===")
    for label, idx in [(INSTRUCT, 1), (THINKING, 2)]:
        emitted = sum(1 for x in allrows if x[idx] in (4, 5))
        hit = sum(1 for x in soc if x[idx] in (4, 5))
        exact = sum(1 for x in soc if x[idx] == x[0])
        res["social_axis"][label] = {"emitted_overall": emitted, "n_social_gt": len(soc),
                                     "on_axis": hit / len(soc), "exact": exact / len(soc)}
        print(f"  {label:22s} emits a social code {emitted:3d}/{len(shared)} overall | "
              f"on GT-social: on-axis {hit / len(soc):.3f}  exact {exact / len(soc):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
