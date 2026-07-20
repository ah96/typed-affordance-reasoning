# Experiment A — data bundle (for lab PC transfer)

Self-contained data for Experiment A (typed affordance reasoning on ADE-Affordance GT).
Copy this whole folder to the lab PC via USB.

## Contents
- `images_full/` — 1,000 full-resolution ADE20K RGB images (`ADE_train_XXXXXXXX.jpg`).
  **Use these** (they match the segmentation resolution).
- `instance_seg/` — 1,000 object segmentations (`ADE_train_XXXXXXXX_seg.png`). The **blue
  channel** encodes the ADE-Affordance instance id: instance `v`'s mask = `seg[:,:,2]==v`.
- `ade_affordance_test/` — ground-truth labels:
  - `*_relationship.txt` — 3 annotators × 3 actions [sit, run, grasp] codes (0–6).
  - `*_exco.json` — per-annotator explanation + consequence lists for exceptions.
- `objectInfo150.txt` — ADE20K class-id → name lookup (optional).

## How the masks were obtained (blocker, now resolved)
ADE-Affordance ships no masks; its instance ids are the original MIT ADE20K ids. The MIT
download site is currently **broken** (registration/captcha fails). We instead pulled the
full release from the HuggingFace mirror **`1aurent/ADE20K`** (BSD, no registration) — see
`../experiment_a/build_instance_masks.py`. Verified: relationship ids exactly match the
seg blue channel.

To rebuild / resume the masks (streams the HF train shards, skips what's already saved):
```
cd ../experiment_a
python3 build_instance_masks.py    # writes into images_full/ and instance_seg/
```

## Ground-truth decoding (verified)
- `relationship.txt` line: `<iid> # s # r # g | # s # r # g | # s # r # g`
  → 3 groups are 3 **annotators**; positions are actions **[sit, run, grasp]**.
  → GT code per (instance, action) = **majority vote** over the 3 annotators
    (3-way tie → most severe). Implemented in `eval_experiment_a_vision.py`.

## Running Experiment A (on the lab PC)
- **No GPU needed** — pure VLM API calls. (GPU is only for Experiment B later.)
- Needs internet + API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
  `TOGETHER_API_KEY`.
```
cd ../experiment_a
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
export GEMINI_API_KEY=...        # start with the free one to smoke-test
python3 eval_experiment_a_vision.py --llms configs/llms.json --limit_images 200
```
Responses are cached under `experiment_a/cache_a_vision/`, so reruns are free.
