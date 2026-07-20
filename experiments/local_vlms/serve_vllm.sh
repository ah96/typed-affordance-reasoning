#!/usr/bin/env bash
# Serve one local VLM at a time on the lab PC (RTX 4080, 16 GB) behind an OpenAI-compatible
# endpoint at http://localhost:8000/v1. The experiment runners then treat it like any cloud
# model via llms_local.json. Needs: pip install "vllm>=0.11".
#
#   ./serve_vllm.sh qwen3_instruct     # then run the experiment against it, Ctrl-C, next model
#   ./serve_vllm.sh qwen3_thinking
#   ./serve_vllm.sh internvl3
#   ./serve_vllm.sh qwen25
#
# 16 GB notes: 8B models in bf16 fit with the reduced context below (two ~1024px images plus
# the prompt stay well under 8k tokens). If vLLM OOMs on startup, first lower
# --gpu-memory-utilization to 0.88, then --max-model-len to 6144.

set -euo pipefail

COMMON=(--port 8000 --max-model-len 8192 --gpu-memory-utilization 0.92
        --limit-mm-per-prompt '{"image": 2}')

case "${1:-}" in
  qwen3_instruct)
    vllm serve Qwen/Qwen3-VL-8B-Instruct "${COMMON[@]}"
    ;;
  qwen3_thinking)
    # The reasoning parser routes the <think> block to reasoning_content, so the client's
    # message.content is only the final JSON answer.
    vllm serve Qwen/Qwen3-VL-8B-Thinking "${COMMON[@]}" --reasoning-parser qwen3
    ;;
  internvl3)
    vllm serve OpenGVLab/InternVL3-8B "${COMMON[@]}" --trust-remote-code
    ;;
  qwen25)
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct "${COMMON[@]}"
    ;;
  *)
    echo "usage: $0 {qwen3_instruct|qwen3_thinking|internvl3|qwen25}" >&2
    exit 1
    ;;
esac
