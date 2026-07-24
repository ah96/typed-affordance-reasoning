# D5 Stage 1 — reason-first typed affordance model (lab PC, RTX 4080)

Stage 0 (`analysis/stage0_mapper.py`) produced the diagnosis this stage is built on.
Models often perceive the right reason and file it under the wrong code, and a text-only
mapper from reason to code saturates at **0.374** even on held-out *human* explanations, so
the reason text alone under-determines the type.
Stage 1 therefore learns reason and code **jointly from pixels**.

**Baselines to beat (already on disk, free):** text-only mapper ceiling **0.374**,
Claude + mapper **0.350**, best zero-shot Type in the paper (Claude) **0.256**,
Qwen3-VL-8B-Instruct zero-shot **0.190** — the last is the honest like-for-like number,
since it is the same base model this run fine-tunes.

**The experiment is the ordering ablation**, not just "does fine-tuning help":

| `--order` | target | reads as |
|---|---|---|
| `reason_first` | `{"explanation", "consequence", "relationship_id"}` | articulate the reason, then commit to a code |
| `label_first` | `{"relationship_id", "explanation", "consequence"}` | commit to a code, then justify it |

Both targets contain identical tokens and differ only in key order, so a gap between them
isolates the effect of reasoning before labelling.
`label_first` is the control and reproduces the schema every evaluated VLM was prompted with.

---

## Step 1 — training pixels (one-time, ~1-3 h, needs internet)

The records come from ADE-Affordance, but its instance IDs index the **full-release** ADE20K
segmentations, not the downsized SceneParse150 copy in `datasets/ADE20K` — crops taken in
the latter's coordinates would misalign.
Reuse Experiment A's fetcher, pointed at the train split:

```bash
cd ~/amar/git/typed-affordance-reasoning/experiments/experiment_a
python3 build_instance_masks.py \
    --labels_dir ../../datasets/ADE-Affordance-flat/train \
    --out_dir    ../d5_reason_first/data/instance_seg_train \
    --img_out    ../d5_reason_first/data/images_train
```

This streams the HF mirror and stops once every requested mask is saved.
It is resumable — re-running skips what is already on disk — so a dropped connection is not
a restart.
Repeat with `--labels_dir ../../datasets/ADE-Affordance-flat/validation` to add the
validation scenes to the same two output directories.

## Step 2 — build the dataset

```bash
cd ../d5_reason_first
python3 build_dataset.py --max_per_class 3000 --out data/train_balanced.jsonl \
    --images_dir data/images_train --seg_dir data/instance_seg_train --verify_pixels
```

Raw ADE-Affordance is **87% FirmlyNegative and only 3.5% exceptions**, so training on it
unbalanced would teach the label prior rather than the task.
`--max_per_class 3000` caps every code and yields roughly 19.7k records over ~7.6k scenes,
which matches the macro-averaged metric the paper reports.
The pixel check drops any record whose image or mask did not download.

Sanity-check what the model will actually be trained on before committing GPU hours:

```bash
python3 train_qlora.py --dry_run --data data/train_balanced.jsonl \
    --images_dir data/images_train --seg_dir data/instance_seg_train
```

## Step 3 — train both orderings

```bash
python3 train_qlora.py --data data/train_balanced.jsonl --order reason_first \
    --images_dir data/images_train --seg_dir data/instance_seg_train
python3 train_qlora.py --data data/train_balanced.jsonl --order label_first \
    --images_dir data/images_train --seg_dir data/instance_seg_train
```

Nothing else may differ between the two runs — same data, same seed, same steps — or the
ablation is not an ablation.

16 GB notes: the base loads in 4-bit NF4 with bf16 compute, the **vision tower is frozen**
(Stage 0 located the failure in the reason-to-code mapping, not in perception, and freezing
it is also what keeps this inside 16 GB), gradient checkpointing is on, and training images
are capped below the eval resolution because training holds activations inference does not.
The step log prints peak GPU memory every 10 steps — watch the first hundred.
If it OOMs, lower `--full_px` (512 → 448), then `--crop_px` (384 → 320), then raise
`--grad_accum` while keeping `--batch_size 1`.
Checkpoints land in `runs/<order>/step<N>` every 500 steps, so a crash costs at most 500.

**Before the full run, smoke-test with `--max_steps 20`** and confirm the loss is finite and
peak memory has headroom.

## Step 4 — merge and evaluate

Evaluation deliberately reuses the paper's own script, so the tuned model is scored exactly
like every other model rather than by a bespoke harness:

```bash
python3 merge_lora.py --adapter runs/reason_first/final --out merged/reason_first

# terminal 1
vllm serve merged/reason_first --served-model-name d5_reason_first \
    --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
    --limit-mm-per-prompt '{"image": 2}' --quantization fp8 \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

Add an entry to `../local_vlms/llms_local.json` with `"model": "d5_reason_first"` and the
usual `http://localhost:8000/v1` base URL, then:

```bash
# terminal 2
cd ../experiment_a
python3 eval_experiment_a_vision.py --llms ../local_vlms/llms_local.json \
    --models d5_reason_first --gt_exceptions_only --limit_images 200 \
    --cache_dir cache_a_local --workers 8
python3 export_raw_results.py --cache_dir cache_a_local --out_dir results
```

Repeat for `label_first`.
Then, on the laptop, `analysis_confusion.py` gives Type/Detect with CIs and the axis
breakdown, and `analysis_ablation.py` can be pointed at the two orderings for a paired
comparison.

**The test scenes were never trained on**: training pools the `train` and `validation`
splits, Experiment A scores the `test` split.

## What to look for

1. **Type vs. the baselines.** Beating 0.374 is the claim that Stage 0 could not make, and
   beating the 0.190 zero-shot of the same base model is the minimum bar.
2. **reason_first vs. label_first Type.** The hypothesis is that ordering matters. A null
   here is publishable and interesting, because it would say the gain is from supervision
   rather than from reasoning order.
3. **The axis breakdown.** D3a showed chain-of-thought rebalancing the label prior across
   axes rather than sharpening discrimination. If fine-tuning instead lifts *exact* typing
   on the social axis, that separates "learning the taxonomy" from "reallocating the prior".
4. **Explanation quality.** Run `analysis_explanations.py`; if Type rises while explanation
   cosine collapses, the model is fitting codes and abandoning reasons, which would undercut
   the reason-first framing.

## Known risks

- **ADE label noise.** Spot-checked training targets include an `ObjectNonFunctional` record
  whose explanation is about social impropriety. The taxonomy is annotator-applied and noisy,
  which bounds achievable Type accuracy and is worth a sentence in the paper.
- **Domain shift.** Training and test are both ADE scenes, so a gain here does not
  demonstrate transfer to AGD20K or elsewhere.
- **Frozen vision tower.** If Stage 0's diagnosis is wrong and the bottleneck is perception
  rather than mapping, this design cannot fix it. That is itself a testable outcome.
