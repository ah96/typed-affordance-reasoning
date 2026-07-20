"""
Majority-vote ensemble vs. ground truth for Experiment A, from the committed raw
predictions — self-contained, no cache, no API, no heavy deps.

Joins the four standard models on (image, action, region_id) and asks whether the
inter-model consensus that Experiment B uses as a reliability signal actually tracks
correctness: the ensemble's mAcc-7 / mAcc-3 / Detect / Type against each single model.
Two tie policies: drop tied items, or resolve ties to the most severe code (the same
convention ADE-Affordance uses for annotator ties). Also quantifies how much more the
models agree with each other than with the labels (raw agreement and Cohen's kappa).

    python3 analysis_ensemble.py
"""
import json
import os
import argparse
from collections import Counter
from itertools import combinations

MODELS = ["gpt_5_5", "claude_sonnet_5", "gemini_3_5_flash", "llama_4_maverick"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "experiment_a", "results")


def to3(c):
    return 0 if c == 0 else (1 if c == 1 else 2)


def macc(pairs, K, collapse=None):
    accs = []
    for c in range(K):
        idx = [(g, p) for g, p in pairs if (collapse(g) if collapse else g) == c]
        if idx:
            accs.append(sum((collapse(p) if collapse else p) == c for _, p in idx) / len(idx))
    return sum(accs) / max(1, len(accs))


def metrics(pairs):
    exc = [(g, p) for g, p in pairs if 2 <= g <= 6]
    return {"mAcc7": macc(pairs, 7), "mAcc3": macc(pairs, 3, collapse=to3),
            "Detect": macc(exc, 3, collapse=to3), "Type": macc(exc, 7), "n": len(pairs)}


def cohen_kappa(a, b, ncat=7):
    N = len(a)
    po = sum(x == y for x, y in zip(a, b)) / N
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / N) * (cb[c] / N) for c in range(ncat))
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "ensemble.json"))
    args = ap.parse_args()

    joined = {}
    for m in MODELS:
        for line in open(os.path.join(args.dir, f"raw_{m}.jsonl")):
            r = json.loads(line)
            if r["gt"] is None or not (0 <= r["pred"] <= 6):
                continue
            joined.setdefault((r["image"], r["action"], r["region_id"]),
                              {"gt": r["gt"]})[m] = r["pred"]
    items = {k: v for k, v in joined.items() if len(v) == len(MODELS) + 1}
    print(f"joined items with all {len(MODELS)} models: {len(items)}")

    out = {"n_joined": len(items), "single": {}, "ensemble": {}}

    for m in MODELS:
        pairs = [(v["gt"], v[m]) for v in items.values()]
        out["single"][m] = metrics(pairs)

    # ensemble votes
    strict, severe = [], []
    n_tied = 0
    for v in items.values():
        votes = Counter(v[m] for m in MODELS)
        top = votes.most_common()
        best, cnt = top[0]
        tied = [c for c, n in top if n == cnt]
        if len(tied) == 1:
            strict.append((v["gt"], best))
            severe.append((v["gt"], best))
        else:
            n_tied += 1
            severe.append((v["gt"], max(tied)))
    out["n_tied"] = n_tied
    out["ensemble"]["plurality_drop_ties"] = metrics(strict)
    out["ensemble"]["plurality_severity_ties"] = metrics(severe)

    # model-model vs model-GT agreement
    gt = [v["gt"] for v in items.values()]
    cols = {m: [v[m] for v in items.values()] for m in MODELS}
    mm_raw, mm_k = [], []
    for a, b in combinations(MODELS, 2):
        mm_raw.append(sum(x == y for x, y in zip(cols[a], cols[b])) / len(gt))
        mm_k.append(cohen_kappa(cols[a], cols[b]))
    mg_raw = {m: sum(x == y for x, y in zip(cols[m], gt)) / len(gt) for m in MODELS}
    mg_k = {m: cohen_kappa(cols[m], gt) for m in MODELS}
    out["model_model"] = {"mean_raw_agreement": sum(mm_raw) / 6, "mean_cohen_kappa": sum(mm_k) / 6}
    out["model_gt"] = {"raw_agreement": mg_raw, "cohen_kappa": mg_k,
                       "mean_raw_agreement": sum(mg_raw.values()) / 4,
                       "mean_cohen_kappa": sum(mg_k.values()) / 4}

    hdr = f"{'':28s}{'mAcc7':>7s}{'mAcc3':>7s}{'Detect':>8s}{'Type':>7s}{'n':>7s}"
    print("\n" + hdr)
    for m in MODELS:
        s = out["single"][m]
        print(f"{m:28s}{s['mAcc7']:7.3f}{s['mAcc3']:7.3f}{s['Detect']:8.3f}{s['Type']:7.3f}{s['n']:7d}")
    for name, s in out["ensemble"].items():
        print(f"ensemble/{name:19s}{s['mAcc7']:7.3f}{s['mAcc3']:7.3f}{s['Detect']:8.3f}{s['Type']:7.3f}{s['n']:7d}")
    print(f"\ntied votes: {n_tied} of {len(items)} ({n_tied / len(items):.1%})")
    print(f"model-model  mean raw agreement {out['model_model']['mean_raw_agreement']:.3f}"
          f"   mean Cohen kappa {out['model_model']['mean_cohen_kappa']:.3f}")
    print(f"model-GT     mean raw agreement {out['model_gt']['mean_raw_agreement']:.3f}"
          f"   mean Cohen kappa {out['model_gt']['mean_cohen_kappa']:.3f}")
    for m in MODELS:
        print(f"  {m:18s} vs GT: raw {mg_raw[m]:.3f}  kappa {mg_k[m]:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
