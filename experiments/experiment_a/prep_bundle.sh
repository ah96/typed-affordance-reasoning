#!/usr/bin/env bash
# ============================================================================
# Rebuild the Experiment A bundle on a fresh machine (needed for the D3
# reasoning ablation). The bundle is git-ignored (regenerable), so:
#
#   1. USB-transfer the ADE-Affordance test labels — datasets/ADE-Affordance-flat/test
#      (~8 MB) — to the lab PC (anywhere), and pass its path as arg 1.
#   2. This script copies the labels into place and calls build_instance_masks.py,
#      which downloads the matching ADE20K images/segs from the HF mirror
#      1aurent/ADE20K (internet + `pip install datasets`).
#
#   ./prep_bundle.sh /path/to/ADE-Affordance-flat/test
#   ./prep_bundle.sh                     # defaults to ../../datasets/ADE-Affordance-flat/test
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

LABELS_SRC="${1:-../../datasets/ADE-Affordance-flat/test}"
BUNDLE=../experiment_a_bundle
DST="$BUNDLE/ade_affordance_test"

mkdir -p "$DST"
if [ -z "$(ls -A "$DST" 2>/dev/null)" ]; then
  if [ ! -d "$LABELS_SRC" ]; then
    echo "!! ADE-Affordance test labels not found at: $LABELS_SRC"
    echo "   USB-transfer datasets/ADE-Affordance-flat/test to the lab PC and pass its path:"
    echo "       ./prep_bundle.sh /path/to/ADE-Affordance-flat/test"
    exit 1
  fi
  echo "copying labels from $LABELS_SRC"
  cp "$LABELS_SRC"/*_relationship.txt "$DST"/ 2>/dev/null
  cp "$LABELS_SRC"/*_exco.json         "$DST"/ 2>/dev/null
fi

echo "downloading ADE20K images/segs for the test IDs (HF mirror 1aurent/ADE20K) ..."
python3 build_instance_masks.py

echo "bundle ready:  $BUNDLE  (images_full/, instance_seg/, ade_affordance_test/)"
