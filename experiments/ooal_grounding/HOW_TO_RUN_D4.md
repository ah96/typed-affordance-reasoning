# D4 — OOAL grounding on the lab PC (RTX 4080)

Revives the ablations cut from the submission: the `ooal_models_amar/{seen_best,unseen_best}`
checkpoints become (1) a sanity-checked saliency model, (2) a third region-selection strategy
for Experiment B, and (3) a GT-grounded selection-quality benchmark on AGD20K — the spatial
anchor the submitted paper lacked. No API spend anywhere: VLM queries, where needed, go
through the D3 local pool (`../local_vlms/`).

## Setup (once)

```bash
cd experiments/ooal_grounding
git clone https://github.com/Reagan1311/OOAL ooal_upstream   # checkout the release matching Jul 2024
# env: the exp-B GPU env + the OOAL repo's requirements (DINOv2 + CLIP deps)
```

`ooal_infer.py` isolates every upstream-version-sensitive assumption in `build_model()` and
`saliency()`; if the cloned revision's constructor or forward signature differs, those two
functions (and nothing else) need a small adaptation — compare with the repo's own `test.py`,
and run `probe_ooal.py` to print the signature, the checkpoint shapes and how upstream builds
the model. The current adapter matches the revision whose `Net.__init__(args, input_dim,
out_dim)` reads `args.class_names`, and calls it as their `test.py` does: `Net(args, 768, 512)`.

Three things follow from that revision and are wired in already:

- **The affordance vocabulary is closed.** The prompt learner is trained per class, so
  `seen_best` only knows the 36 `SEEN_AFF` names and `unseen_best` only the 25 `UNSEEN_AFF`
  names (read from the clone's `data/agd20k_ego.py`). The adapter picks the list by the
  checkpoint's class dimension and skips any requested affordance outside it, loudly. Running
  `unseen_best` over the *Seen* tree therefore drops 11 affordances (`beat`, `boxing`,
  `brush_with`, `drag`, `lift`, `look_out`, `pack`, `stir`, `talk_on`, `text_on`, `write`) —
  pair each checkpoint with its own split.
- **Inference runs at 224 px** with a plain resize and ImageNet stats, matching their
  `TestData` transform (`--input_px` overrides).
- **CUDA is required**: the upstream constructor and forward hardcode `.cuda()`. The first run
  also downloads DINOv2 via `torch.hub` and CLIP ViT-B/16, so the lab PC needs network access
  once.

## Step 1 — checkpoint sanity (do this first)

Dump saliency maps for the Seen egocentric test images and score against GT:

```bash
python3 ooal_infer.py --ckpt ../../ooal_models_amar/seen_best --ooal_repo ooal_upstream \
    --tree ../../datasets/AGD20K/Seen/testset/egocentric --outdir heatmaps_agd
python3 eval_selection.py --split Seen --metrics --heatmaps heatmaps_agd
```

`--tree` walks `<affordance>/<object>/*.jpg` and scores each image for its own affordance on
a single model load, so it needs no `--affordances` list. Use `--images <flat dir>` with an
explicit `--affordances` only when every image should be scored for every affordance.

Expected: KLD/SIM/NSS in the ballpark of the OOAL paper (printed by the script). If they are
far off, stop and fix the adapter before anything downstream. Then repeat for the other split:

```bash
python3 ooal_infer.py --ckpt ../../ooal_models_amar/unseen_best --ooal_repo ooal_upstream \
    --tree ../../datasets/AGD20K/Unseen/testset/egocentric --outdir heatmaps_agd_unseen
python3 eval_selection.py --split Unseen --metrics --heatmaps heatmaps_agd_unseen
```

(1710 Seen and 540 Unseen test images; metrics here are computed at the GT's native size,
whereas upstream scores at 224 px, so small offsets from the published numbers are expected.)

## Step 2 — GT selection benchmark (AGD20K, no VLMs)

Does OOAL-ranked selection surface the right regions better than area-ranked?

```bash
python3 rank_regions.py --images <flat dir of Seen test images> --heatmaps heatmaps_agd \
    --K 3 --out regions_agd_ooal_K3.jsonl
python3 rank_regions.py --images <same dir> --heatmaps heatmaps_agd --rank_by area \
    --K 3 --out regions_agd_area_K3.jsonl        # size-ranked baseline, same masks
python3 eval_selection.py --split Seen --recall \
    --regions_area regions_agd_area_K3.jsonl --regions_ooal regions_agd_ooal_K3.jsonl
```

## Step 3 — third selection strategy for Experiment B

OOAL heatmaps for the 200 committed ADE scenes, using the paper's 6 actions:

```bash
python3 ooal_infer.py --ckpt ../../ooal_models_amar/unseen_best --ooal_repo ooal_upstream \
    --images ../experiment_b_bundle/images \
    --affordances sit_on,hold,carry,cut,throw,ride --outdir heatmaps_expb
python3 rank_regions.py --images ../experiment_b_bundle/images --heatmaps heatmaps_expb \
    --K 3 --out regions_sam2_ooal_K3.jsonl
```

(`unseen_best` is the right checkpoint here: ADE objects are out-of-domain for AGD20K.)

Then query the D3 local pool on those regions and score all three strategies with the same
models:

```bash
cd ../local_vlms
python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct \
    --mode sam2_ooal --regions ../ooal_grounding/regions_sam2_ooal_K3.jsonl
# ...repeat per local model, plus sam2_area and sam3_concept replays for the same models...
cp results/*.jsonl ../experiment_b/results/
cd ../experiment_b
for MODE in sam2_area sam3_concept sam2_ooal; do
  python3 experiment_b_agreement.py --outdir results --mode $MODE --K 3 \
      --models qwen3_vl_8b_instruct,internvl3_8b,qwen2_5_vl_7b
done
```

This yields the paper's Table-4-style comparison (agreement, exception rate) with
**sam2_ooal** as the new row — selection strategies compared within one held-fixed model
pool.

## What each step feeds in the paper

| step | claim it supports |
|---|---|
| 1 | checkpoints valid; specialist baseline numbers (KLD/SIM/NSS vs GT) |
| 2 | selection quality measured against real spatial GT, not proxies |
| 3 | affordance-aware selection vs area vs concept, same models, same scenes |
