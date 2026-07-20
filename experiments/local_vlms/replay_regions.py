"""
Replay Experiment B's exact (image, region, action) queries against new models — extends the
agreement pool without re-running SAM. Region lists (with bboxes) come from the committed
results files, images from experiment_b_bundle/, and the system prompt / user prompt / crop /
resize logic is imported from experiment_b_run_v2.py, so new predictions are directly
comparable to the four submitted models: score them together with experiment_b_agreement.py.

Runs against any entry in a llms.json-style config — a local vLLM server (see serve_vllm.sh)
or a free API tier. Per-call cache, resumable, atomic per-image flush, same output schema
<name>_<mode>_K<K>.jsonl as the original runner.

  python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct --mode sam2_area
  python3 replay_regions.py --dry_run --mode sam2_area          # no server/keys, mock client

With --regions, jobs come from an explicit jsonl (e.g. ooal_grounding/rank_regions.py output)
instead of the committed results; --mode then only names the output files:

  python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct \
      --mode sam2_ooal --regions ../ooal_grounding/regions_sam2_ooal_K3.jsonl
"""
import json
import os
import sys
import glob
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_B = os.path.join(HERE, "..", "experiment_b")
sys.path.insert(0, EXP_B)
from experiment_b_run_v2 import SYSTEM_PROMPT, user_prompt, b64png, crop_bbox  # noqa: E402
from vision_llm_clients import load_llms, make_client  # noqa: E402

BUNDLE = os.path.join(HERE, "..", "experiment_b_bundle", "images")
SOURCE = os.path.join(EXP_B, "results")


class MockClient:
    name = "mock"

    def complete_json(self, *, system, user, images_b64png):
        return {"relationship_id": 0, "explanation": "", "consequence": ""}


def load_region_jobs(mode, K, regions=None):
    """Union of (image, region_id, bbox, action) across the four committed results files,
    or an explicit regions jsonl when given."""
    jobs = {}
    files = [regions] if regions else glob.glob(os.path.join(SOURCE, f"*_{mode}_K{K}.jsonl"))
    for f in files:
        for line in open(f):
            r = json.loads(line)
            jobs[(r["image"], r["region_id"], r["action"])] = tuple(r["bbox"])
    return [{"image": im, "region_id": rid, "action": a, "bbox": bb}
            for (im, rid, a), bb in sorted(jobs.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llms", default=os.path.join(HERE, "llms_local.json"))
    ap.add_argument("--models", default=None, help="comma-separated names from the config")
    ap.add_argument("--mode", required=True,
                    help="sam2_area / sam3_concept (committed regions) or a new name with --regions")
    ap.add_argument("--regions", default=None,
                    help="explicit regions jsonl (image, region_id, bbox, action) overriding the committed files")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--full_px", type=int, default=1024)
    ap.add_argument("--crop_px", type=int, default=768)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "cache_replay"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        clients = {"mock": MockClient()}
    else:
        cfgs = load_llms(args.llms)
        if args.models:
            want = {m.strip() for m in args.models.split(",") if m.strip()}
            cfgs = [c for c in cfgs if c.name in want]
        clients = {c.name: make_client(c) for c in cfgs}
    if not clients:
        raise SystemExit("no models selected")

    jobs = load_region_jobs(args.mode, args.K, args.regions)
    by_image = defaultdict(list)
    for j in jobs:
        by_image[j["image"]].append(j)
    print(f"mode={args.mode} K={args.K}: {len(jobs)} (region, action) queries "
          f"over {len(by_image)} images, models: {', '.join(clients)}")

    os.makedirs(args.outdir, exist_ok=True)
    for name, client in clients.items():
        out_path = os.path.join(args.outdir, f"{name}_{args.mode}_K{args.K}.jsonl")
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path):
                r = json.loads(line)
                done.add((r["image"], r["region_id"], r["action"]))
        n_err = 0
        for image_id, ijobs in tqdm(sorted(by_image.items()), desc=name):
            todo = [j for j in ijobs if (j["image"], j["region_id"], j["action"]) not in done]
            if not todo:
                continue
            img = Image.open(os.path.join(BUNDLE, image_id))
            full_b64 = b64png(img, args.full_px)
            crops = {}
            for j in todo:
                if j["region_id"] not in crops:
                    crops[j["region_id"]] = b64png(crop_bbox(img, j["bbox"]), args.crop_px)

            def call(j):
                cpath = os.path.join(args.cache_dir, name, image_id,
                                     f"{j['action']}_{j['region_id']}.json")
                if os.path.exists(cpath):
                    return j, json.load(open(cpath)), None
                try:
                    pred = client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(j["action"]),
                                                images_b64png=[full_b64, crops[j["region_id"]]])
                    rid = int(pred.get("relationship_id", -1))
                    if not (0 <= rid <= 6):
                        raise ValueError(f"bad relationship_id {rid}")
                except Exception as e:
                    return j, None, str(e)
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                json.dump(pred, open(cpath, "w"))
                return j, pred, None

            rows = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for j, pred, err in ex.map(call, todo):
                    if err:
                        n_err += 1
                        continue
                    rows.append({"image": j["image"], "region_id": j["region_id"],
                                 "bbox": list(j["bbox"]), "action": j["action"],
                                 "relationship_id": int(pred["relationship_id"]),
                                 "explanation": pred.get("explanation", "") or "",
                                 "consequence": pred.get("consequence", "") or ""})
            if rows:
                with open(out_path, "a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
        print(f"{name}: wrote {out_path}  (errors this run: {n_err})")


if __name__ == "__main__":
    main()
