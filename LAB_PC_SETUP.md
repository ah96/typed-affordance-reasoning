# Lab-PC setup & run guide (RTX 4080)

How to get the GPU experiments (D3, D4) running on the lab PC after cloning this repo.
The laptop-side work (D1/D2/D5 reanalysis, the LLM judge) does **not** belong here — it runs
on the laptop and needs no GPU.

---

## 1. Get the code

```bash
git clone <this-repo-url> typed-affordance-reasoning
cd typed-affordance-reasoning
git pull            # if already cloned
```

Everything code-side comes with the clone, **including the Experiment A bundle**
(`experiments/experiment_a_bundle/`, 257M) and the 200 Experiment B scenes
(`experiments/experiment_b_bundle/images/`). No download or USB needed for those.

---

## 2. USB-transfer these (git-ignored, too big for GitHub)

| Copy from the laptop | Size | Put it at (on the lab PC) | Needed for |
|---|---|---|---|
| `ooal_models_amar/` | 1.1G | `./ooal_models_amar/` (repo root) | D4 |
| `datasets/AGD20K/` | 6.9G | `./datasets/AGD20K/` (repo root) | D4 |

That's it. **Not** needed on the lab PC: the Exp A bundle (now in git), the ADE-Affordance
labels (inside the bundle), `sam_vit_h_4b8939.pth` (unused — the pipeline auto-downloads
`sam2.1_l.pt`), and `datasets/ADE20K` (the vLLM/HF and SAM downloads cover everything else).

Internet is used on first run for: the OOAL repo clone, vLLM model pulls (Qwen/InternVL from
Hugging Face), and the ultralytics SAM weight auto-download.

---

## 3. Install

```bash
python3 -m venv .venv && source .venv/bin/activate     # or your existing GPU env
pip install "vllm>=0.11" ultralytics
# D4 also needs the OOAL repo's own deps (DINOv2 / CLIP) — see its requirements after cloning.
```

---

## 4. Run — D3 (open-weight VLMs)

All from `experiments/local_vlms/`. Each driver serves a model, runs, and shuts the server
down; everything is cached/resumable (re-run to continue after an interruption).

```bash
cd experiments/local_vlms

# D3a — HEADLINE: same-weights reasoning ablation (Qwen3-VL Instruct vs Thinking),
#       Experiment A exception subset. ~30-60 min.
./run_d3_reasoning_ablation.sh

# D3b — widen the Experiment B agreement pool (fewer 2-2 ties, frozen open snapshots).
./run_d3_agreement_replay.sh qwen3_vl_8b_instruct internvl3_8b
```

Results land in `experiments/experiment_a/results/raw_qwen3_vl_8b_*.jsonl` and
`experiments/experiment_b/results/`. Richer breakdowns:
`cd ../analysis && python3 analysis_confusion.py` (Exp A) and `analysis_agreement.py` (Exp B).

Priority if GPU time is short: **D3a first** (it's the headline table). For the full
13,512-pair Exp A run, remove `--gt_exceptions_only` from the eval line in the driver.

---

## 5. Run — D4 (OOAL grounding)

From `experiments/ooal_grounding/`. Needs the two USB transfers from §2 in place.

```bash
cd experiments/ooal_grounding
git clone https://github.com/Reagan1311/OOAL ooal_upstream    # checkout the Jul-2024 release
./run_d4_grounding.sh
```

**Stop after Step 1** if the printed KLD/SIM/NSS are far from the OOAL paper
(~1.07 / 0.46 / 1.14 on Seen): that means the upstream adapter in `ooal_infer.py`
(`build_model` / `saliency`) needs a small fix for the cloned revision before the rest is
trustworthy. Steps 2-3 then give the GT-grounded selection benchmark and the third
selection strategy (`sam2_ooal`) for Experiment B.

To query VLMs on the new `sam2_ooal` regions and compare all three selection strategies in
one model pool, from `experiments/local_vlms/`:

```bash
python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct \
    --mode sam2_ooal --regions ../ooal_grounding/regions_sam2_ooal_K3.jsonl
```

---

## 6. Which needs the GPU?

| Task | GPU? |
|---|---|
| D3a `run_d3_reasoning_ablation.sh`, D3b (local models), D4, `serve_vllm.sh` | **Yes** |
| D3b with an OpenRouter `:free` model | No (remote API) |
| D1/D2/D5 in `experiments/analysis/`, the LLM judge, all scorers | No (laptop) |

Rule of thumb: the GPU is needed only to run a model's weights locally.

---

## 7. After a run

Commit the new `results/*.jsonl` and push; the laptop side folds the numbers into the paper.
The big USB'd inputs (`ooal_models_amar/`, `datasets/AGD20K/`) stay git-ignored — don't commit
them. Detailed notes live in
[`experiments/local_vlms/HOW_TO_RUN_D3.md`](experiments/local_vlms/HOW_TO_RUN_D3.md) and
[`experiments/ooal_grounding/HOW_TO_RUN_D4.md`](experiments/ooal_grounding/HOW_TO_RUN_D4.md).
