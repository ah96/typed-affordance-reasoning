"""
LLM-judge scoring of Experiment A exception explanations, with a validity audit —
Gemini free tier (text-only), resumable per-call cache, no paid tokens.

For every GT-exception row where a model wrote an explanation (prediction 2-6), the judge
sees the action, the GT type name, the 3 reference sentences, and the model explanation,
and returns strict JSON:

  {"verdict": "same_reason" | "partially_related" | "different_reason",
   "confidence": 1-5}

The audit re-judges a stratified subset under perturbations that should not change the
verdict; how often they do measures the judge, not the models:

  --audit consistency   identical input, second pass (cache key differs)
  --audit refshuffle    reference sentences in reversed order
  --audit surface       candidate lowercased, final period stripped, "I think " prefixed

Run (from this directory, key exported in the shell):

  export GEMINI_API_KEY=...
  python3 judge_explanations.py                      # base pass, all models
  python3 judge_explanations.py --audit consistency --audit refshuffle --audit surface
  python3 judge_explanations.py --report             # tables + out/judge.json, no API

Free-tier rate limits are handled by --sleep (default 5s between calls) plus the client's
own backoff; interrupt and re-run at any time, finished calls are never re-sent.
"""
import json
import os
import sys
import glob
import time
import random
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiment_b"))
RESULTS_DIR = os.path.join(HERE, "..", "experiment_a", "results")
GT_DIR = os.path.join(HERE, "..", "..", "datasets", "ADE-Affordance-flat", "test")
CACHE = os.path.join(HERE, "cache_judge")
AUDIT_N = 150          # per model, stratified over predicted type
VERDICTS = ("same_reason", "partially_related", "different_reason")
REL7 = ["Positive", "FirmlyNegative", "ObjectNonFunctional", "PhysicalObstacle",
        "SociallyAwkward", "SociallyForbidden", "Dangerous"]

SYSTEM = (
    "You compare a model-written explanation against reference explanations written by "
    "human annotators. All texts give the reason an action is inappropriate or impossible "
    "in a scene. Decide whether the model explanation expresses the same underlying reason "
    "as ANY of the references (same_reason), overlaps only partly (partially_related), or "
    "gives a different reason (different_reason). Judge the REASON, not the wording, length, "
    "or style. Reply with strict JSON only: "
    '{"verdict": "same_reason|partially_related|different_reason", "confidence": 1-5}'
)


def user_prompt(action, gt_code, refs, cand):
    lines = [f"Action: {action}", f"Annotated constraint type: {REL7[gt_code]}", "Reference explanations:"]
    lines += [f"{i + 1}. {r}" for i, r in enumerate(refs)]
    lines += ["Model explanation:", cand]
    return "\n".join(lines)


def load_refs():
    refs = {}
    for path in glob.glob(os.path.join(GT_DIR, "*_exco.json")):
        image = os.path.basename(path)[: -len("_exco.json")]
        for action, by_inst in json.load(open(path)).items():
            for iid, v in by_inst.items():
                r = v.get("explanation", [])
                r = [r] if isinstance(r, str) else r
                r = [x.strip() for x in r if x and x.strip()]
                if r:
                    refs[(image, action, int(iid))] = r
    return refs


def load_jobs(refs):
    jobs = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "raw_*.jsonl"))):
        name = os.path.basename(f)[len("raw_"):-len(".jsonl")]
        for line in open(f):
            r = json.loads(line)
            g, p = r["gt"], r["pred"]
            cand = (r.get("explanation") or "").strip()
            if g is None or not (2 <= g <= 6) or not (2 <= p <= 6) or not cand:
                continue
            key = (r["image"], r["action"], r["region_id"])
            if key not in refs:
                continue
            jobs.append({"model": name, "image": r["image"], "action": r["action"],
                         "region_id": r["region_id"], "gt": g, "pred": p,
                         "cand": cand, "refs": refs[key]})
    return jobs


def audit_subset(jobs):
    by_model = defaultdict(list)
    for j in jobs:
        by_model[j["model"]].append(j)
    subset = []
    for model, js in by_model.items():
        by_pred = defaultdict(list)
        for j in js:
            by_pred[j["pred"]].append(j)
        rng = random.Random(7)
        want = AUDIT_N
        picked = []
        codes = sorted(by_pred)
        per = max(1, want // len(codes))
        for c in codes:
            pool = sorted(by_pred[c], key=lambda j: (j["image"], j["action"], j["region_id"]))
            picked += pool if len(pool) <= per else rng.sample(pool, per)
        subset += picked[:want]
    return subset


def variant(job, audit):
    refs, cand = job["refs"], job["cand"]
    if audit == "refshuffle":
        refs = list(reversed(refs))
    elif audit == "surface":
        cand = "I think " + cand.lower().rstrip(".")
    return refs, cand


def cache_path(job, audit):
    tag = "base" if not audit else audit
    return os.path.join(CACHE, tag, job["model"],
                        f"{job['image']}_{job['action']}_{job['region_id']}.json")


def run(jobs, audits, sleep):
    from vision_llm_clients import make_client, LLMConfig
    cfgs = json.load(open(os.path.join(HERE, "..", "experiment_b", "configs", "llms.json")))
    gem = next(c for c in cfgs if c["name"] == "gemini_3_5_flash")
    client = make_client(LLMConfig(**{k: v for k, v in gem.items() if not k.startswith("_")}))

    todo = [(j, a) for a in ([None] + audits) for j in (jobs if a is None else audit_subset(jobs))
            if not os.path.exists(cache_path(j, a))]
    print(f"pending judge calls: {len(todo)}")
    for n, (job, audit) in enumerate(todo, 1):
        refs, cand = variant(job, audit)
        try:
            out = client.complete_json(system=SYSTEM,
                                       user=user_prompt(job["action"], job["gt"], refs, cand),
                                       images_b64png=[])
            v = out.get("verdict")
            if v not in VERDICTS:
                raise ValueError(f"bad verdict {v!r}")
        except Exception as e:
            print(f"[{n}/{len(todo)}] ERROR {e} — skipped, will retry on next run")
            time.sleep(sleep)
            continue
        p = cache_path(job, audit)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        rec = dict(job, verdict=v, confidence=out.get("confidence"), audit=audit or "base")
        json.dump(rec, open(p, "w"))
        if n % 25 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)}] ok")
        time.sleep(sleep)


def report(jobs, audits):
    def read(tag):
        recs = []
        for p in glob.glob(os.path.join(CACHE, tag, "*", "*.json")):
            recs.append(json.load(open(p)))
        return recs

    out = {"base": {}, "audit": {}}
    base = read("base")
    by_model = defaultdict(list)
    for r in base:
        by_model[r["model"]].append(r)
    print(f"\n=== judge verdicts (base pass, n={len(base)}) ===")
    print(f"{'model':20s}{'n':>6s}{'same':>8s}{'partial':>9s}{'diff':>7s}   same by: exact / w-axis")
    for m, rs in sorted(by_model.items()):
        n = len(rs)
        frac = {v: sum(r["verdict"] == v for r in rs) / n for v in VERDICTS}
        exact = [r for r in rs if r["pred"] == r["gt"]]
        wax = [r for r in rs if r["pred"] != r["gt"]]
        se = sum(r["verdict"] == "same_reason" for r in exact) / len(exact) if exact else 0
        sw = sum(r["verdict"] == "same_reason" for r in wax) / len(wax) if wax else 0
        out["base"][m] = {"n": n, **frac, "same_given_exact_type": se, "same_given_wrong_type": sw}
        print(f"{m:20s}{n:6d}{frac['same_reason']:8.3f}{frac['partially_related']:9.3f}"
              f"{frac['different_reason']:7.3f}   {se:.3f} / {sw:.3f}")

    base_ix = {(r["model"], r["image"], r["action"], r["region_id"]): r["verdict"] for r in base}
    for a in audits:
        recs = read(a)
        pairs = [(base_ix.get((r["model"], r["image"], r["action"], r["region_id"])), r["verdict"])
                 for r in recs]
        pairs = [(b, v) for b, v in pairs if b]
        if not pairs:
            continue
        agree = sum(b == v for b, v in pairs) / len(pairs)
        out["audit"][a] = {"n": len(pairs), "verdict_stability": agree}
        print(f"audit {a:12s} n={len(pairs):5d}  verdict stability {agree:.3f}")

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "out", "judge.json"), "w"), indent=1)
    print("wrote out/judge.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="append", default=[],
                    choices=["consistency", "refshuffle", "surface"])
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    refs = load_refs()
    jobs = load_jobs(refs)
    print(f"judgeable rows (model wrote an explanation): {len(jobs)}")
    if args.report:
        report(jobs, ["consistency", "refshuffle", "surface"])
    else:
        run(jobs, args.audit, args.sleep)


if __name__ == "__main__":
    main()
