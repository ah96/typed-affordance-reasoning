"""
Chance-corrected agreement for Experiment B, from the committed raw predictions —
self-contained, no cache, no API, no heavy deps.

Adds what the submitted paper's raw agreement rates lack: Fleiss' kappa, Krippendorff's
alpha (nominal), per-pair Cohen's kappa, a tie-free per-model typicality score (mean
agreement with the other three models, so no 2-2 ties are dropped), conditional type
agreement (given two models both flag an exception, do they name the same type?), and
per-model label distributions.

    python3 analysis_agreement.py                # both modes, K=3
    python3 analysis_agreement.py --mode sam2_area
"""
import json
import os
import argparse
from collections import Counter
from itertools import combinations

# The four paper models; --models overrides it (e.g. to add the D3 open-weight voters).
MODELS = ["gpt_5_5", "claude_sonnet_5", "gemini_3_5_flash", "llama_4_maverick"]
REL7 = ["Positive", "FirmlyNegative", "ObjectNonFunctional", "PhysicalObstacle",
        "SociallyAwkward", "SociallyForbidden", "Dangerous"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "experiment_b", "results")


def to3(c):
    return 0 if c == 0 else (1 if c == 1 else 2)


def load_items(indir, mode, K):
    """(image, region_id, action) -> {model: label}, kept only if all models answered."""
    preds = {}
    for m in MODELS:
        path = os.path.join(indir, f"{m}_{mode}_K{K}.jsonl")
        for line in open(path):
            r = json.loads(line)
            c = r.get("relationship_id")
            if c is None or not (0 <= c <= 6):
                continue
            preds.setdefault((r["image"], r["region_id"], r["action"]), {})[m] = c
    return {k: v for k, v in preds.items() if len(v) == len(MODELS)}


def fleiss_kappa(items, ncat):
    n = len(MODELS)
    Pi_sum, cat_tot = 0.0, Counter()
    for labs in items:
        cnt = Counter(labs)
        cat_tot.update(cnt)
        Pi_sum += (sum(v * v for v in cnt.values()) - n) / (n * (n - 1))
    N = len(items)
    Pbar = Pi_sum / N
    tot = sum(cat_tot.values())
    Pe = sum((cat_tot[c] / tot) ** 2 for c in range(ncat))
    return (Pbar - Pe) / (1 - Pe)


def kripp_alpha(items, ncat):
    n = len(MODELS)
    o = [[0.0] * ncat for _ in range(ncat)]
    for labs in items:
        cnt = Counter(labs)
        for c in cnt:
            for k in cnt:
                pairs = cnt[c] * cnt[k] - (cnt[c] if c == k else 0)
                o[c][k] += pairs / (n - 1)
    nc = [sum(o[c]) for c in range(ncat)]
    ntot = sum(nc)
    Do = sum(o[c][k] for c in range(ncat) for k in range(ncat) if c != k)
    De = sum(nc[c] * nc[k] for c in range(ncat) for k in range(ncat) if c != k) / (ntot - 1)
    return 1.0 - Do / De


def cohen_kappa(a, b, ncat):
    N = len(a)
    po = sum(x == y for x, y in zip(a, b)) / N
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / N) * (cb[c] / N) for c in range(ncat))
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def analyze(indir, mode, K):
    items = load_items(indir, mode, K)
    labs7 = [[v[m] for m in MODELS] for v in items.values()]
    labs3 = [[to3(x) for x in row] for row in labs7]
    N = len(labs7)
    out = {"mode": mode, "K": K, "n_items": N}

    out["fleiss_kappa_7"] = fleiss_kappa(labs7, 7)
    out["fleiss_kappa_3"] = fleiss_kappa(labs3, 3)
    out["kripp_alpha_7"] = kripp_alpha(labs7, 7)
    out["kripp_alpha_3"] = kripp_alpha(labs3, 3)

    cols7 = list(zip(*labs7))
    cols3 = list(zip(*labs3))
    out["pairwise_cohen_kappa_7"] = {}
    out["pairwise_cohen_kappa_3"] = {}
    for i, j in combinations(range(len(MODELS)), 2):
        key = f"{MODELS[i]}|{MODELS[j]}"
        out["pairwise_cohen_kappa_7"][key] = cohen_kappa(cols7[i], cols7[j], 7)
        out["pairwise_cohen_kappa_3"][key] = cohen_kappa(cols3[i], cols3[j], 3)
    npairs = len(MODELS) * (len(MODELS) - 1) // 2
    out["mean_pairwise_cohen_kappa_7"] = sum(out["pairwise_cohen_kappa_7"].values()) / npairs
    out["mean_pairwise_cohen_kappa_3"] = sum(out["pairwise_cohen_kappa_3"].values()) / npairs

    # tie-free typicality: mean agreement of each model with all the others, all items kept
    out["typicality"] = {}
    for i, m in enumerate(MODELS):
        s = sum(sum(row[i] == row[j] for j in range(len(MODELS)) if j != i) / (len(MODELS) - 1)
                for row in labs7)
        out["typicality"][m] = s / N

    # conditional type agreement: both models say exception (2-6) -> same type?
    both_exc, same_type = Counter(), Counter()
    for row in labs7:
        for i, j in combinations(range(len(MODELS)), 2):
            if row[i] >= 2 and row[j] >= 2:
                key = f"{MODELS[i]}|{MODELS[j]}"
                both_exc[key] += 1
                same_type[key] += row[i] == row[j]
    out["conditional_type_agreement"] = {
        k: {"n_both_exception": both_exc[k], "same_type_rate": same_type[k] / both_exc[k]}
        for k in both_exc}
    tot_b, tot_s = sum(both_exc.values()), sum(same_type.values())
    out["conditional_type_agreement_overall"] = {"n": tot_b, "rate": tot_s / tot_b}

    # the analogous conditional agreement on the 3-way verdict, for contrast
    agree3 = sum(sum(r3[i] == r3[j] for i, j in combinations(range(len(MODELS)), 2))
                 for r3 in labs3)
    out["pairwise_agreement_3way"] = agree3 / (npairs * N)
    agree7 = sum(sum(r7[i] == r7[j] for i, j in combinations(range(len(MODELS)), 2))
                 for r7 in labs7)
    out["pairwise_agreement_7way"] = agree7 / (npairs * N)

    out["label_distribution"] = {
        m: {REL7[c]: cnt / N for c, cnt in sorted(Counter(cols7[i]).items())}
        for i, m in enumerate(MODELS)}
    return out


def report(r):
    print(f"\n=== mode={r['mode']}  K={r['K']}  items(all {len(MODELS)} models)={r['n_items']} ===")
    print(f"raw pairwise agreement    7-way {r['pairwise_agreement_7way']:.3f}   3-way {r['pairwise_agreement_3way']:.3f}")
    print(f"Fleiss kappa              7-way {r['fleiss_kappa_7']:.3f}   3-way {r['fleiss_kappa_3']:.3f}")
    print(f"Krippendorff alpha        7-way {r['kripp_alpha_7']:.3f}   3-way {r['kripp_alpha_3']:.3f}")
    print(f"mean pairwise Cohen kappa 7-way {r['mean_pairwise_cohen_kappa_7']:.3f}   3-way {r['mean_pairwise_cohen_kappa_3']:.3f}")
    cta = r["conditional_type_agreement_overall"]
    print(f"both-exception type agreement: {cta['rate']:.3f}  (n={cta['n']} model-pair events)")
    print(f"typicality (tie-free, vs other {len(MODELS) - 1}):",
          "  ".join(f"{m} {v:.3f}" for m, v in r["typicality"].items()))
    print("label distribution (per model):")
    for m, d in r["label_distribution"].items():
        row = "  ".join(f"{k[:12]:12s}{v:5.1%}" for k, v in d.items())
        print(f"  {m:18s} {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--mode", choices=["sam2_area", "sam3_concept"], default=None)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "agreement.json"))
    ap.add_argument("--models", default=None,
                    help="Comma list overriding the default 4-model pool (D3b widens it).")
    args = ap.parse_args()
    if args.models:
        global MODELS
        MODELS = args.models.split(",")
    modes = [args.mode] if args.mode else ["sam2_area", "sam3_concept"]
    results = []
    for mode in modes:
        r = analyze(args.dir, mode, args.K)
        report(r)
        results.append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
