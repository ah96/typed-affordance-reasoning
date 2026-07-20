#!/usr/bin/env bash
# ============================================================================
# D3 HEADLINE — same-weights reasoning ablation on the lab PC (RTX 4080).
#
# Serves Qwen3-VL-8B *Instruct*, runs the Experiment A exception subset (579
# GT exceptions), stops the server, then does the same for the *Thinking*
# variant, and prints Type/Detect for both. Same weights, CoT off vs on — the
# clean isolation the o4-mini comparison in the paper could not give.
#
# Run from experiments/local_vlms/ :   ./run_d3_reasoning_ablation.sh
# Needs: vllm (pip install "vllm>=0.11"), a GPU, and the Experiment A bundle.
# Everything is cached/resumable — re-run to resume after an interruption.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

LLMS="$PWD/llms_local.json"
EXP_A="$(cd ../experiment_a && pwd)"
BUNDLE="$(cd .. && pwd)/experiment_a_bundle"
CACHE="$EXP_A/cache_a_local"
PORT=8000

if [ ! -d "$BUNDLE/images_full" ]; then
  echo "!! Experiment A bundle missing at $BUNDLE"
  echo "   It normally ships in the repo (git pull brings it). If it is absent,"
  echo "   regenerate it:  cd $EXP_A && ./prep_bundle.sh"
  exit 1
fi

wait_ready() {   # <server_pid> <logfile>
  echo "   waiting for vLLM to load the model ..."
  for _ in $(seq 1 180); do        # up to 15 min
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      echo "   server ready."; return 0
    fi
    kill -0 "$1" 2>/dev/null || { echo "!! vLLM exited early — see $2"; return 1; }
    sleep 5
  done
  echo "!! vLLM did not become ready within 15 min — see $2"; return 1
}

run_one() {      # <serve_key> <model_name_in_config>
  local key="$1" name="$2" log="vllm_$1.log"
  echo "== [$name] serving ($key), log: $log =="
  ./serve_vllm.sh "$key" >"$log" 2>&1 &
  local pid=$!
  if wait_ready "$pid" "$log"; then
    echo "== [$name] Experiment A, exception subset =="
    ( cd "$EXP_A" && python3 eval_experiment_a_vision.py \
        --llms "$LLMS" --models "$name" --bundle "$BUNDLE" \
        --gt_exceptions_only --limit_images 200 \
        --cache_dir "$CACHE" --workers 8 )
  fi
  echo "== [$name] stopping server =="
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  sleep 3
}

run_one qwen3_instruct qwen3_vl_8b_instruct
run_one qwen3_thinking qwen3_vl_8b_thinking

echo "== exporting raw predictions and scoring the ablation =="
( cd "$EXP_A" && python3 export_raw_results.py --cache_dir "$CACHE" --out_dir results )
( cd "$EXP_A/results" && python3 score_from_raw.py --exceptions_only )

echo
echo "Done. New raw predictions:  $EXP_A/results/raw_qwen3_vl_8b_*.jsonl"
echo "For the richer breakdown (7x7, axis, CIs):"
echo "    cd ../analysis && python3 analysis_confusion.py"
echo "For the FULL 13,512-pair run, re-run this script after removing"
echo "--gt_exceptions_only from the eval line (long, ~overnight at 8 workers)."
