#!/usr/bin/env bash
# ============================================================================
# D3 — widen the Experiment B agreement pool with open-weight models.
#
# For each model key given, serves it, replays the EXACT committed Exp B
# regions (both sam2_area and sam3_concept — SAM never re-runs), stops it.
# Then copies the new predictions next to the four frontier models and scores
# the widened pool. More voters => fewer 2-2 ties, stronger majority, and
# frozen open-weight snapshots answer the closed-API-drift limitation.
#
# Run from experiments/local_vlms/ :
#     ./run_d3_agreement_replay.sh qwen3_vl_8b_instruct internvl3_8b
# Defaults to qwen3_vl_8b_instruct if no args. Cached/resumable.
#
# Server-key mapping (config name -> serve_vllm.sh key):
#   qwen3_vl_8b_instruct -> qwen3_instruct   internvl3_8b -> internvl3
#   qwen3_vl_8b_thinking -> qwen3_thinking   qwen2_5_vl_7b -> qwen25
# OpenRouter :free entries (e.g. or_free_example) need no server — this script
# skips serving for any name it has no key for and calls the API directly.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

LLMS="$PWD/llms_local.json"
EXP_B="$(cd ../experiment_b && pwd)"
PORT=8000
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen3_vl_8b_instruct)

serve_key_for() {   # config name -> serve_vllm.sh key ("" = no local server)
  case "$1" in
    qwen3_vl_8b_instruct) echo qwen3_instruct ;;
    qwen3_vl_8b_thinking) echo qwen3_thinking ;;
    internvl3_8b)         echo internvl3 ;;
    qwen2_5_vl_7b)        echo qwen25 ;;
    *)                    echo "" ;;          # API model: no local server
  esac
}

wait_ready() {
  echo "   waiting for vLLM ..."
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "   ready."; return 0; }
    kill -0 "$1" 2>/dev/null || { echo "!! vLLM exited early — see $2"; return 1; }
    sleep 5
  done
  echo "!! vLLM not ready in 15 min — see $2"; return 1
}

for name in "${MODELS[@]}"; do
  key="$(serve_key_for "$name")"
  pid=""
  if [ -n "$key" ]; then
    log="vllm_$key.log"
    echo "== [$name] serving ($key), log: $log =="
    ./serve_vllm.sh "$key" >"$log" 2>&1 &
    pid=$!
    wait_ready "$pid" "$log" || { kill "$pid" 2>/dev/null; continue; }
  else
    echo "== [$name] API model — no local server =="
  fi

  for mode in sam3_concept sam2_area; do   # concept first: cheaper (1,215 vs 3,342)
    echo "== [$name] replay $mode =="
    python3 replay_regions.py --llms "$LLMS" --models "$name" --mode "$mode"
  done

  if [ -n "$pid" ]; then
    echo "== [$name] stopping server =="; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true; sleep 3
  fi
done

echo "== scoring the widened pool =="
cp results/*.jsonl "$EXP_B/results/" 2>/dev/null || true
POOL="gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick,$(IFS=,; echo "${MODELS[*]}")"
for mode in sam2_area sam3_concept; do
  echo "-- $mode, pool: $POOL --"
  ( cd "$EXP_B" && python3 experiment_b_agreement.py --outdir results --mode "$mode" --K 3 --models "$POOL" )
done
echo
echo "Chance-corrected view (edit MODELS at the top of analysis_agreement.py to add the new names):"
echo "    cd ../analysis && python3 analysis_agreement.py"
