#!/usr/bin/env bash
# ============================================================================
# D4 — OOAL grounding on the lab PC (RTX 4080). Revives the cut ablations.
#
# Prereqs (checked below):
#   - OOAL upstream repo cloned here as ./ooal_upstream
#       git clone https://github.com/Reagan1311/OOAL ooal_upstream
#   - OOAL checkpoints present at ../../ooal_models_amar/{seen_best,unseen_best}
#       (USB-transfer these; they are git-ignored)
#   - AGD20K present at ../../datasets/AGD20K  (USB-transfer; git-ignored)
#   - env with torch+CUDA, ultralytics, the OOAL repo's deps
#
# Run from experiments/ooal_grounding/ :   ./run_d4_grounding.sh
#
# STOP after Step 1 if the sanity metrics are far from the OOAL paper's
# (~KLD 1.07 / SIM 0.46 / NSS 1.14 Seen): that means the upstream adapter in
# ooal_infer.py needs a small fix (build_model / saliency) before trusting
# anything downstream. Each step is resumable (skips existing outputs).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

REPO="ooal_upstream"
CKPT_SEEN="../../ooal_models_amar/seen_best"
CKPT_UNSEEN="../../ooal_models_amar/unseen_best"
AGD="../../datasets/AGD20K"
EXPB_IMAGES="../experiment_b_bundle/images"

miss=0
[ -d "$REPO" ]            || { echo "!! missing $REPO (git clone the OOAL repo here)"; miss=1; }
[ -f "$CKPT_SEEN" ]       || { echo "!! missing $CKPT_SEEN (USB-transfer ooal_models_amar/)"; miss=1; }
[ -d "$AGD/Seen/testset" ]|| { echo "!! missing $AGD (USB-transfer datasets/AGD20K/)"; miss=1; }
[ "$miss" = 0 ] || exit 1

echo "############ Step 1 — checkpoint sanity vs AGD20K GT (Seen) ############"
python3 ooal_infer.py --ckpt "$CKPT_SEEN" --ooal_repo "$REPO" \
    --tree "$AGD/Seen/testset/egocentric" --outdir heatmaps_seen
python3 eval_selection.py --split Seen --metrics --heatmaps heatmaps_seen
echo ">> If KLD/SIM/NSS are far from the paper, FIX ooal_infer.py before continuing."
echo

echo "############ Step 2 — GT-grounded selection quality (Seen) ############"
# Flat dir of Seen egocentric test images for the ranker (names stay unique).
FLAT=agd_seen_flat
if [ ! -d "$FLAT" ]; then
  mkdir -p "$FLAT"
  find "$AGD/Seen/testset/egocentric" -type f \( -name '*.jpg' -o -name '*.png' \) \
    -exec ln -sf {} "$FLAT/" \;
fi
python3 rank_regions.py --images "$FLAT" --heatmaps heatmaps_seen --rank_by ooal \
    --K 3 --out regions_agd_ooal_K3.jsonl
python3 rank_regions.py --images "$FLAT" --heatmaps heatmaps_seen --rank_by area \
    --K 3 --out regions_agd_area_K3.jsonl
python3 eval_selection.py --split Seen --recall \
    --regions_area regions_agd_area_K3.jsonl --regions_ooal regions_agd_ooal_K3.jsonl
echo

echo "###### Step 3 — third selection strategy on the Exp B scenes ######"
# ADE objects are out-of-domain for AGD20K, so use the Unseen checkpoint here.
python3 ooal_infer.py --ckpt "$CKPT_UNSEEN" --ooal_repo "$REPO" \
    --images "$EXPB_IMAGES" --affordances sit_on,hold,carry,cut,throw,ride \
    --outdir heatmaps_expb
python3 rank_regions.py --images "$EXPB_IMAGES" --heatmaps heatmaps_expb --rank_by ooal \
    --K 3 --out regions_sam2_ooal_K3.jsonl
echo
echo "Done Steps 1-3. To query VLMs on the new sam2_ooal regions and compare all"
echo "three selection strategies within one model pool, run from ../local_vlms/:"
echo "    python3 replay_regions.py --llms llms_local.json --models qwen3_vl_8b_instruct \\"
echo "        --mode sam2_ooal --regions ../ooal_grounding/regions_sam2_ooal_K3.jsonl"
echo "then score sam2_area / sam3_concept / sam2_ooal with experiment_b_agreement.py."
