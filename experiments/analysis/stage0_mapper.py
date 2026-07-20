"""
Stage 0 of the reason-first architecture: is the taxonomy MAPPING the fixable bottleneck?

Trains a text-only mapper (sentence embeddings + logistic regression) from ADE-Affordance
TRAIN-split explanation texts to their canonical exception codes (2-6), then re-maps the
committed frontier-VLM explanations on Experiment A: wherever a model predicted an exception
and wrote an explanation, its code is replaced by the mapper's code inferred from the model's
own words. If Type accuracy rises toward Detect levels, the wrong-code-right-reason diagnosis
is confirmed and the reason-first architecture is validated — with zero new model queries.

Also reports: the mapper's ceiling on held-out TEST-split annotator explanations, a zero-shot
centroid baseline (no training) for contrast, and per-model before/after Type accuracy.

    python3 stage0_mapper.py            # full run (CPU, ~20 min, models already cached)
"""
import json
import os
import re
import glob
import argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "..", "..", "datasets", "ADE-Affordance-flat")
RESULTS_DIR = os.path.join(HERE, "..", "experiment_a", "results")

ACTION_POS = {"sit": 0, "run": 1, "grasp": 2}
FILE2CANON = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
EXC = (2, 3, 4, 5, 6)
REL7 = ["Positive", "FirmlyNegative", "ObjectNonFunctional", "PhysicalObstacle",
        "SociallyAwkward", "SociallyForbidden", "Dangerous"]
# class descriptions for the zero-shot centroid baseline
DESC = {2: "the object is broken, damaged, depleted or otherwise not functional",
        3: "a physical obstacle or scene constraint blocks the action",
        4: "doing this would be socially awkward or contextually inappropriate",
        5: "doing this is socially forbidden, disrespectful or against rules or law",
        6: "doing this is dangerous and could cause injury or harm"}


def _aggregate(votes):
    code, n = Counter(votes).most_common(1)[0]
    return code if n >= 2 else max(votes)


def parse_relationship(path):
    """instance_id -> {action: canonical code}. Tolerates 1 annotator group (train/val)
    or 3 groups (test, majority vote with ties to most severe)."""
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
        rec = {}
        for a, p in ACTION_POS.items():
            rec[a] = FILE2CANON[_aggregate([g[p] for g in groups])]
        out[iid] = rec
    return out


def collect_split(split):
    """(text, code) training pairs from one split's exco + relationship files."""
    pairs = []
    for exco_path in glob.glob(os.path.join(FLAT, split, "*_exco.json")):
        rel_path = exco_path.replace("_exco.json", "_relationship.txt")
        if not os.path.exists(rel_path):
            continue
        rel = parse_relationship(rel_path)
        for action, by_inst in json.load(open(exco_path)).items():
            for iid_s, v in by_inst.items():
                iid = int(iid_s)
                code = rel.get(iid, {}).get(action)
                if code not in EXC:
                    continue
                texts = v.get("explanation", [])
                texts = [texts] if isinstance(texts, str) else texts
                for t in texts:
                    t = (t or "").strip()
                    if t:
                        pairs.append((t, code))
    return pairs


def macc_type(rows):
    """7-way mAcc over GT-exception rows (the paper's Type metric)."""
    accs = []
    for c in EXC:
        sel = [p for g, p in rows if g == c]
        if sel:
            accs.append(sum(p == c for p in sel) / len(sel))
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out", "stage0_mapper.json"))
    args = ap.parse_args()

    train = collect_split("train") + collect_split("validation")
    test_gt = collect_split("test")
    print(f"train pairs {len(train)}  (dist { {REL7[c]: n for c, n in sorted(Counter(c for _, c in train).items())} })")
    print(f"test  pairs {len(test_gt)}")

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    st = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    def embed(texts):
        return st.encode(texts, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=True)

    X_tr = embed([t for t, _ in train])
    y_tr = np.array([c for _, c in train])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    # mapper ceiling on held-out annotator text
    X_te = embed([t for t, _ in test_gt])
    y_te = np.array([c for _, c in test_gt])
    acc_gt = float((clf.predict(X_te) == y_te).mean())

    # zero-shot centroid baseline on the same held-out text
    C_emb = embed([DESC[c] for c in EXC])
    zs = [EXC[i] for i in np.argmax(X_te @ C_emb.T, axis=1)]
    acc_zs = float((np.array(zs) == y_te).mean())
    print(f"\nmapper on held-out GT explanations: {acc_gt:.3f}  (zero-shot centroid {acc_zs:.3f}, chance 0.2)")

    out = {"n_train": len(train), "n_test_gt": len(test_gt),
           "mapper_acc_gt_text": acc_gt, "zeroshot_acc_gt_text": acc_zs, "models": {}}

    # the experiment: re-map committed model explanations
    print(f"\n{'model':20s}{'Type before':>12s}{'Type after':>12s}{'Detect':>9s}{'n remap':>9s}")
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_*.jsonl"))):
        name = os.path.basename(f)[len("raw_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(f)]
        rows = [r for r in rows if r["gt"] is not None and 0 <= r["pred"] <= 6 and 2 <= r["gt"] <= 6]
        det = [r for r in rows if 2 <= r["pred"] <= 6 and (r.get("explanation") or "").strip()]
        if det:
            X = embed([r["explanation"].strip() for r in det])
            remap = clf.predict(X)
            for r, c in zip(det, remap):
                r["_remap"] = int(c)
        before = macc_type([(r["gt"], r["pred"]) for r in rows])
        after = macc_type([(r["gt"], r.get("_remap", r["pred"])) for r in rows])
        detect3 = sum(2 <= r["pred"] <= 6 for r in rows)
        det_macc = macc_type([(r["gt"], r["gt"] if 2 <= r["pred"] <= 6 else r["pred"]) for r in rows])
        out["models"][name] = {"type_before": before, "type_after": after,
                               "type_ceiling_given_detection": det_macc,
                               "n_gt_exc": len(rows), "n_remapped": len(det)}
        print(f"{name:20s}{before:12.3f}{after:12.3f}{det_macc:9.3f}{len(det):9d}")
    print("(Detect column = Type ceiling if every detected exception were typed perfectly)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
