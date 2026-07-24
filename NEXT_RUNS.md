# Next lab-PC runs — ordered runbook (written 2026-07-24)

D3a and D3b are done and written into the paper.
This file lists what remains, in the order that maximizes what each hour of GPU time buys.
Every command assumes the repo root is `~/amar/git/typed-affordance-reasoning` and that
`git pull` is current.

Environment notes that already bit us once, all handled inside `serve_vllm.sh`: FP8
checkpoints (bf16 8B does not fit 16 GB), PIECEWISE CUDA graphs, `--gpu-memory-utilization
0.85`, `--max-num-seqs 4`, and `VLLM_USE_FLASHINFER_SAMPLER=0`.
Do not raise the memory flags without re-testing a full multi-image run.

---

## Priority 1 — InternVL3-8B (~1-2 h)

**Why first.** The widened agreement pool in the paper's agreement section currently adds two
voters that share a base model, so they are not independent and the paper says they are.
InternVL3 has a non-Qwen vision tower and fixes that claim.
It also tests whether the collapsed social axis is a Qwen quirk or a property of 8B VLMs,
which changes how strongly that finding can be stated.

```bash
# terminal 1 — serve (first launch downloads ~16 GB and fp8-quantizes at load time)
cd ~/amar/git/typed-affordance-reasoning/experiments/local_vlms
./serve_vllm.sh internvl3
```

This is the first non-Qwen model through the serve path, so watch the first minute.
`--trust-remote-code` and `--quantization fp8` are already set.
If it OOMs at load, drop `--gpu-memory-utilization` to 0.82 in `serve_vllm.sh` for this model
only, and if the vision tower rejects fp8, fall back to `./serve_vllm.sh qwen25`
(Qwen2.5-VL-7B) which gives architecture diversity at 7B instead.

```bash
# terminal 2 — Experiment B replay, both selection strategies
cd ~/amar/git/typed-affordance-reasoning/experiments/local_vlms
python3 replay_regions.py --llms llms_local.json --models internvl3_8b --mode sam2_area
python3 replay_regions.py --llms llms_local.json --models internvl3_8b --mode sam3_concept
cp results/*.jsonl ../experiment_b/results/

# terminal 2 — Experiment A exception subset, for the axis comparison
cd ../experiment_a
python3 eval_experiment_a_vision.py --llms ../local_vlms/llms_local.json \
    --models internvl3_8b --gt_exceptions_only --limit_images 200 \
    --cache_dir cache_a_local --workers 8
python3 export_raw_results.py --cache_dir cache_a_local --out_dir results
```

Then commit `experiment_a/results/raw_internvl3_8b.jsonl` and the two
`experiment_b/results/internvl3_8b_*.jsonl`, and push.

---

## Priority 2 — D4, OOAL grounding

Both USB inputs are on the lab PC, so this is unblocked.
Full detail in `experiments/ooal_grounding/HOW_TO_RUN_D4.md`; this is the short path.

### Step 1 — checkpoint sanity (a GATE, not a formality)

```bash
cd ~/amar/git/typed-affordance-reasoning/experiments/ooal_grounding
git clone https://github.com/Reagan1311/OOAL ooal_upstream
AFFS=$(ls ../../datasets/AGD20K/Seen/testset/egocentric | paste -sd,)
python3 ooal_infer.py --ckpt ../../ooal_models_amar/seen_best --ooal_repo ooal_upstream \
    --images-from-tree ../../datasets/AGD20K/Seen/testset/egocentric \
    --affordances "$AFFS" --outdir heatmaps_agd
python3 eval_selection.py --split Seen --metrics --heatmaps heatmaps_agd
```

Expected KLD / SIM / NSS in the ballpark of **1.07 / 0.46 / 1.14**.
If they are far off, **stop and report the output**: it means `build_model()` or `saliency()`
in `ooal_infer.py` needs adapting to the revision GitHub actually served, and every
downstream number would be meaningless.
That adapter has never run on a GPU, so this is the most likely place for the day to derail.

### Step 2 — GT selection benchmark (no VLMs, cheap)

```bash
# flatten the nested test tree once (stems are unique)
mkdir -p flat_seen && find ../../datasets/AGD20K/Seen/testset/egocentric -name '*.jpg' \
    -exec ln -sf {} flat_seen/ \;
python3 rank_regions.py --images flat_seen --heatmaps heatmaps_agd --K 3 \
    --out regions_agd_ooal_K3.jsonl
python3 rank_regions.py --images flat_seen --heatmaps heatmaps_agd --rank_by area --K 3 \
    --out regions_agd_area_K3.jsonl
python3 eval_selection.py --split Seen --recall \
    --regions_area regions_agd_area_K3.jsonl --regions_ooal regions_agd_ooal_K3.jsonl
```

### Step 3 — third selection strategy for Experiment B

```bash
python3 ooal_infer.py --ckpt ../../ooal_models_amar/unseen_best --ooal_repo ooal_upstream \
    --images ../experiment_b_bundle/images \
    --affordances sit_on,hold,carry,cut,throw,ride --outdir heatmaps_expb
python3 rank_regions.py --images ../experiment_b_bundle/images --heatmaps heatmaps_expb \
    --K 3 --out regions_sam2_ooal_K3.jsonl
```

`unseen_best` is correct here because ADE objects are out of AGD20K's domain.

**Shortcut vs. the runbook.** HOW_TO_RUN_D4 step 3 suggests the pool
`qwen3_vl_8b_instruct, internvl3_8b, qwen2_5_vl_7b`, but that pool has no `sam2_area` or
`sam3_concept` replays, so it would mean three models times three modes from scratch.
Use the two Qwen variants that are already done on the other two modes instead, and you get
the three-strategy comparison for one mode's worth of new calls:

```bash
cd ../local_vlms
# serve each model in turn, then:
python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct \
    --mode sam2_ooal --regions ../ooal_grounding/regions_sam2_ooal_K3.jsonl
python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_thinking \
    --mode sam2_ooal --regions ../ooal_grounding/regions_sam2_ooal_K3.jsonl
cp results/*.jsonl ../experiment_b/results/

cd ../experiment_b
for MODE in sam2_area sam3_concept sam2_ooal; do
  python3 experiment_b_agreement.py --outdir results --mode $MODE --K 3 \
      --models qwen3_vl_8b_instruct,qwen3_vl_8b_thinking \
      --out results/agree_${MODE}_K3_d4pool.json
done
```

Add `internvl3_8b` to both the replay list and the `--models` list if Priority 1 is done.

---

## Priority 3 — Full Experiment A for the Qwen pair (overnight)

Both variants currently have only the 579-exception subset, while the four frontier models
have all 13,512 pairs in the paper's main table.
Dropping `--gt_exceptions_only` puts them in that table.
The exception calls are already cached, so this resumes rather than repeating them.

```bash
# terminal 1
cd ~/amar/git/typed-affordance-reasoning/experiments/local_vlms
./serve_vllm.sh qwen3_instruct

# terminal 2  (~2-3 h)
cd ../experiment_a
python3 eval_experiment_a_vision.py --llms ../local_vlms/llms_local.json \
    --models qwen3_vl_8b_instruct --limit_images 200 \
    --cache_dir cache_a_local --workers 8
```

Then the same with `qwen3_thinking` / `qwen3_vl_8b_thinking`.
The Thinking variant generates a reasoning trace on every call, so budget several times
longer and start it before leaving.
Export and push as usual.

---

## After any run

```bash
cd ~/amar/git/typed-affordance-reasoning/experiments/experiment_a
python3 export_raw_results.py --cache_dir cache_a_local --out_dir results
git add -A && git commit -m "<what ran>" && git push
```

The per-call caches are git-ignored, so **a manual eval leaves nothing to commit until you
export**.
Analysis all happens on the laptop afterwards: `analysis_confusion.py`,
`analysis_ablation.py`, `analysis_agreement.py --models <pool>`, `analysis_explanations.py`.
