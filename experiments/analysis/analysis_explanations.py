"""
Semantic explanation scoring for Experiment A's 579 GT exceptions — local models only
(sentence embeddings, NLI entailment, BERTScore), no API. CPU is enough: ~2.9k texts.

The paper scored BLEU/METEOR/ROUGE over ALL GT-exception rows, so a missed exception
(prediction 0/1, empty explanation) counts as ~0 and the text metrics partly re-measure
detection. Here every metric is reported under both conventions:

  all       — paper convention, empty candidates score 0
  detected  — only rows where the model predicted an exception (2-6), i.e. wrote text

and, within detected, split by whether the predicted TYPE was exactly right, on the right
axis (functional {2,3} / social {4,5} / danger {6}), or on the wrong axis — testing whether
mistyped detections still carry the right reason in free text.

References are the 3 annotator sentences per (action, instance); scores take the max over
references. Models: sentence-transformers/all-mpnet-base-v2 (cosine), cross-encoder/
nli-deberta-v3-base (entailment prob, max over refs of mean(cand->ref, ref->cand)),
BERTScore roberta-large F1 (its own multi-ref handling).

    python3 analysis_explanations.py               # explanation field
    python3 analysis_explanations.py --field consequence
"""
import json
import os
import glob
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "experiment_a", "results")
GT_DIR = os.path.join(HERE, "..", "..", "datasets", "ADE-Affordance-flat", "test")
AXIS = {2: "functional", 3: "functional", 4: "social", 5: "social", 6: "danger"}


def load_refs(field):
    """(image, action, instance_id) -> [ref sentences]"""
    refs = {}
    for path in glob.glob(os.path.join(GT_DIR, "*_exco.json")):
        image = os.path.basename(path)[: -len("_exco.json")]
        data = json.load(open(path))
        for action, by_inst in data.items():
            for iid, v in by_inst.items():
                r = v.get(field, [])
                if isinstance(r, str):
                    r = [r]
                r = [x.strip() for x in r if x and x.strip()]
                if r:
                    refs[(image, action, int(iid))] = r
    return refs


def load_rows(refs, field):
    """model -> list of {gt, pred, cand, refs, subset flags}"""
    rows = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_*.jsonl"))):
        name = os.path.basename(f)[len("raw_"):-len(".jsonl")]
        for line in open(f):
            r = json.loads(line)
            g, p = r["gt"], r["pred"]
            if g is None or not (2 <= g <= 6) or not (0 <= p <= 6):
                continue
            key = (r["image"], r["action"], r["region_id"])
            if key not in refs:
                continue
            cand = (r.get(field) or "").strip()
            detected = 2 <= p <= 6
            rows[name].append({
                "gt": g, "pred": p, "cand": cand, "refs": refs[key],
                "detected": detected,
                "type_match": "exact" if p == g else (
                    "right_axis" if detected and AXIS[p] == AXIS[g] else (
                        "wrong_axis" if detected else "missed")),
            })
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return f"{x:.3f}" if x is not None else "  —  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=["explanation", "consequence"], default="explanation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(HERE, "out", f"semantic_{args.field}.json")

    refs = load_refs(args.field)
    rows = load_rows(refs, args.field)
    print(f"reference sets: {len(refs)}  |  models: {', '.join(rows)}")

    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    import bert_score as bs
    st = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cpu")
    ENTAIL = 1  # label order of this cross-encoder: [contradiction, entailment, neutral]

    results = {}
    for name, rs in rows.items():
        det = [r for r in rs if r["detected"] and r["cand"]]

        # embedding cosine, max over refs (batched once per model)
        cands = [r["cand"] for r in det]
        flat_refs, owners = [], []
        for i, r in enumerate(det):
            for ref in r["refs"]:
                flat_refs.append(ref)
                owners.append(i)
        if det:
            e_c = st.encode(cands, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            e_r = st.encode(flat_refs, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            best = [0.0] * len(det)
            for j, i in enumerate(owners):
                c = float(util.cos_sim(e_c[i], e_r[j]))
                best[i] = max(best[i], c)
            for r, b in zip(det, best):
                r["cos"] = b

            # NLI entailment: max over refs of mean of both directions
            pairs, powner = [], []
            for i, r in enumerate(det):
                for ref in r["refs"]:
                    pairs.append((r["cand"], ref)); powner.append((i, 0))
                    pairs.append((ref, r["cand"])); powner.append((i, 1))
            import torch
            with torch.no_grad():
                logits = nli.predict(pairs, batch_size=32, show_progress_bar=False,
                                     apply_softmax=True)
            acc = defaultdict(list)
            for (i, d), pr in zip(powner, logits):
                acc[i].append((d, float(pr[ENTAIL])))
            for i, r in enumerate(det):
                by_ref = defaultdict(dict)
                for k, (d, p) in enumerate(acc[i]):
                    by_ref[k // 2][d] = p
                r["nli"] = max((v.get(0, 0) + v.get(1, 0)) / 2 for v in by_ref.values())

            # BERTScore F1 with native multi-ref
            _, _, F = bs.score(cands, [r["refs"] for r in det], lang="en",
                               model_type="roberta-large", batch_size=32, verbose=False)
            for r, f1 in zip(det, F.tolist()):
                r["bsf1"] = f1

        n_all, n_det = len(rs), len(det)
        model_res = {"n_all": n_all, "n_detected": n_det}
        for metric in ("cos", "nli", "bsf1"):
            vals_det = [r.get(metric) for r in det]
            model_res[metric] = {
                "all": (sum(v for v in vals_det if v is not None) / n_all) if n_all else None,
                "detected": mean(vals_det),
            }
            for sub in ("exact", "right_axis", "wrong_axis"):
                model_res[metric][sub] = mean([r.get(metric) for r in det
                                               if r["type_match"] == sub])
        results[name] = model_res

        print(f"\n=== {name}  ({args.field})  GT-exc rows {n_all}, detected {n_det} ===")
        print(f"  {'metric':6s} {'all':>7s} {'detect':>7s} {'exact':>7s} {'r-axis':>7s} {'w-axis':>7s}")
        for metric, label in (("cos", "cosine"), ("nli", "NLI"), ("bsf1", "BSc-F1")):
            m = model_res[metric]
            print(f"  {label:6s} {fmt(m['all']):>7s} {fmt(m['detected']):>7s} "
                  f"{fmt(m['exact']):>7s} {fmt(m['right_axis']):>7s} {fmt(m['wrong_axis']):>7s}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
