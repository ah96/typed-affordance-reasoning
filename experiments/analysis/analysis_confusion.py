"""
Confusion structure and uncertainty for Experiment A, from the committed raw predictions —
self-contained, no cache, no API, no heavy deps.

Per model: the full 7x7 confusion matrix (GT rows, prediction columns), image-cluster
bootstrap CIs for mAcc-7 / mAcc-3 / Detect / Type, and an exception-axis breakdown that
splits errors on GT exceptions into: missed (predicted 0/1), wrong axis, right axis but
wrong type, and exact. Axes: functional/physical = {2,3}, social = {4,5}, danger = {6}.

    python3 analysis_confusion.py
    python3 analysis_confusion.py --boot 2000
"""
import json
import os
import glob
import random
import argparse
from collections import Counter, defaultdict

REL7 = ["Positive", "FirmlyNegative", "ObjectNonFunctional", "PhysicalObstacle",
        "SociallyAwkward", "SociallyForbidden", "Dangerous"]
AXIS = {2: "functional", 3: "functional", 4: "social", 5: "social", 6: "danger"}
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "experiment_a", "results")


def to3(c):
    return 0 if c == 0 else (1 if c == 1 else 2)


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r["gt"] is None or not (0 <= r["pred"] <= 6):
            continue
        rows.append((r["image"], r["gt"], r["pred"]))
    return rows


def macc(pairs, K, collapse=None):
    accs = []
    for c in range(K):
        idx = [(g, p) for g, p in pairs if (collapse(g) if collapse else g) == c]
        if idx:
            accs.append(sum((collapse(p) if collapse else p) == c for _, p in idx) / len(idx))
    return sum(accs) / max(1, len(accs))


def metrics(pairs):
    exc = [(g, p) for g, p in pairs if 2 <= g <= 6]
    return {
        "mAcc7": macc(pairs, 7),
        "mAcc3": macc(pairs, 3, collapse=to3),
        "Detect": macc(exc, 3, collapse=to3) if exc else None,
        "Type": macc(exc, 7) if exc else None,
    }


def bootstrap(rows, nboot, seed=0):
    by_img = defaultdict(list)
    for img, g, p in rows:
        by_img[img].append((g, p))
    imgs = sorted(by_img)
    rng = random.Random(seed)
    dists = defaultdict(list)
    for _ in range(nboot):
        sample = [pair for img in rng.choices(imgs, k=len(imgs)) for pair in by_img[img]]
        for k, v in metrics(sample).items():
            if v is not None:
                dists[k].append(v)
    ci = {}
    for k, d in dists.items():
        d.sort()
        ci[k] = (d[int(0.025 * len(d))], d[int(0.975 * len(d))])
    return ci


def axis_breakdown(pairs):
    """On GT exceptions: exact / right axis wrong type / wrong axis / missed (pred in 0,1)."""
    out = {}
    for name, codes in [("all", (2, 3, 4, 5, 6)), ("functional", (2, 3)),
                        ("social", (4, 5)), ("danger", (6,))]:
        sel = [(g, p) for g, p in pairs if g in codes]
        if not sel:
            continue
        n = len(sel)
        exact = sum(p == g for g, p in sel)
        missed = sum(p <= 1 for g, p in sel)
        right_axis = sum(p >= 2 and p != g and AXIS[p] == AXIS[g] for g, p in sel)
        wrong_axis = sum(p >= 2 and AXIS[p] != AXIS[g] for g, p in sel)
        out[name] = {"n": n, "exact": exact / n, "right_axis_wrong_type": right_axis / n,
                     "wrong_axis": wrong_axis / n, "missed": missed / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "confusion.json"))
    args = ap.parse_args()

    results = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "raw_*.jsonl"))):
        name = os.path.basename(f)[len("raw_"):-len(".jsonl")]
        rows = load(f)
        pairs = [(g, p) for _, g, p in rows]
        conf = [[0] * 7 for _ in range(7)]
        for g, p in pairs:
            conf[g][p] += 1
        res = {"n": len(pairs), "metrics": metrics(pairs), "confusion": conf,
               "ci": bootstrap(rows, args.boot), "axis": axis_breakdown(pairs)}
        results[name] = res

        print(f"\n=== {name}  n={res['n']} ===")
        m, ci = res["metrics"], res["ci"]
        for k in ("mAcc7", "mAcc3", "Detect", "Type"):
            if m[k] is not None:
                print(f"  {k:7s} {m[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
        gt_tot = [sum(r) for r in conf]
        print("  confusion (rows GT, % of row):")
        print("           " + " ".join(f"{n[:6]:>6s}" for n in REL7))
        for g in range(7):
            if gt_tot[g] == 0:
                continue
            row = " ".join(f"{100 * conf[g][p] / gt_tot[g]:6.1f}" for p in range(7))
            print(f"  {REL7[g][:9]:9s}{row}   (n={gt_tot[g]})")
        print("  exceptions by axis:")
        for k, v in res["axis"].items():
            print(f"    {k:10s} n={v['n']:4d}  exact {v['exact']:.3f}  right-axis {v['right_axis_wrong_type']:.3f}"
                  f"  wrong-axis {v['wrong_axis']:.3f}  missed {v['missed']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
