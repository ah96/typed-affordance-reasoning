# D3 — Open-weight VLMs on the lab PC (RTX 4080)

Two goals, zero API spend:

1. **Same-weights reasoning ablation** — Qwen3-VL-8B *Instruct* vs *Thinking* are the same
   base model with and without chain-of-thought. Running both on Experiment A's exception
   subset isolates what o4-mini could not: the effect of reasoning with model identity held
   fixed.
2. **Grow the agreement pool** — new models judge the *identical* SAM regions committed in
   `experiment_b/results/` (replayed from their stored bboxes; SAM never runs again). More
   voters means fewer 2-2 ties and a stronger majority signal, and open weights are frozen
   snapshots, answering the closed-API drift limitation.

## Setup (once)

```bash
source ~/venvamar/bin/activate          # or a fresh env with torch+CUDA
pip install "vllm>=0.11" "openai>=2.47"   # old openai breaks vllm import (NamespaceTool)
python3 -m nltk.downloader wordnet omw-1.4  # METEOR scoring needs WordNet
```

First launch of each model downloads its checkpoint (~9 GB) and torch-compiles
(~10 min total); later launches reuse both caches and are ready in ~2–4 min.
`serve_vllm.sh` also exports `VLLM_USE_FLASHINFER_SAMPLER=0` — the lab PC's system
nvcc predates the 4080, so FlashInfer's JIT sampling kernel cannot build there.

Everything below runs from this directory. Models are served one at a time on
`http://localhost:8000/v1`; the runners talk to whatever is being served.

## 1. Reasoning ablation (Experiment A, exception subset first)

Needs `experiment_a_bundle/` (images + masks + GT). If the lab PC doesn't have it, regenerate
with `experiment_a/build_instance_masks.py` (downloads via the HF mirror) or copy it over.

```bash
# terminal 1                                # terminal 2
./serve_vllm.sh qwen3_instruct              cd ../experiment_a
                                            python3 eval_experiment_a_vision.py \
                                              --llms ../local_vlms/llms_local.json \
                                              --models qwen3_vl_8b_instruct \
                                              --gt_exceptions_only --limit_images 200 \
                                              --cache_dir cache_a_local --workers 8

# Ctrl-C terminal 1, then the same with:    qwen3_thinking / qwen3_vl_8b_thinking
```

Then the full 13,512-pair run for the Instruct model (drop `--gt_exceptions_only`; roughly a
long overnight run at 8 workers). Run the Thinking variant on the full set too if time allows —
otherwise the exception subset already gives the ablation table. Afterwards export and score
exactly like the paper models:

```bash
python3 export_raw_results.py --cache_dir cache_a_local     # -> results/raw_<name>.jsonl
cd results && python3 score_from_raw.py --exceptions_only
cd ../../analysis && python3 analysis_confusion.py && python3 analysis_ensemble.py
```

## 2. Agreement-pool replay (Experiment B regions)

No bundle prep needed — images are committed in `experiment_b_bundle/`, regions come from the
committed results files.

```bash
# with a model being served (or an OpenRouter :free entry in llms_local.json):
python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct --mode sam2_area
python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct --mode sam3_concept
# repeat per model; then score the widened pool (old four + new) together:
cp results/*.jsonl ../experiment_b/results/
cd ../experiment_b
python3 experiment_b_agreement.py --outdir results --mode sam2_area --K 3 \
  --models gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick,qwen3_vl_8b_instruct,internvl3_8b
cd ../analysis && python3 analysis_agreement.py     # extend MODELS list at the top first
```

Smoke tests (laptop, no GPU/keys): `python3 replay_regions.py --dry_run --mode sam2_area`, and
for Exp A the existing `--dry_run` flag of `eval_experiment_a_vision.py`.

## Model pool

| config name | serve command | role |
|---|---|---|
| qwen3_vl_8b_instruct | `./serve_vllm.sh qwen3_instruct` | ablation pair, no CoT |
| qwen3_vl_8b_thinking | `./serve_vllm.sh qwen3_thinking` | ablation pair, CoT |
| internvl3_8b | `./serve_vllm.sh internvl3` | architecture diversity |
| qwen2_5_vl_7b | `./serve_vllm.sh qwen25` | fallback if vLLM lacks Qwen3-VL |
| or_free_example | — (OpenRouter `:free`) | larger open models, rate-limited |

16 GB does not fit 7-8B in bf16 (~17 GB of weights), so `serve_vllm.sh` serves the official
FP8 checkpoints for the Qwen3-VL pair and fp8-quantizes the others at load time; anything
larger goes through the OpenRouter `:free` tier (verify the current free list first — it drifts).
The memory flags in `serve_vllm.sh` (util 0.85, `--max-num-seqs 4`, PIECEWISE CUDA graphs)
are each pinned by an observed OOM on the 4080 — see the comment above COMMON before raising
them. The server caps concurrency at 4, so eval `--workers 8` just queues — that is fine.

## Priorities if GPU time is short

1. Ablation pair on the 579-exception subset (both variants) — the headline table.
2. Instruct model, full Experiment A — comparability with the four paper models.
3. Replay on `sam3_concept` (1,215 queries — cheap), then `sam2_area` (3,342).
