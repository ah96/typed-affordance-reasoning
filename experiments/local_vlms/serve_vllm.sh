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
# 16 GB notes: 8B models do NOT fit in bf16 (~17 GB of weights alone), so the Qwen3-VL pair
# uses the official FP8 checkpoints (~9 GB) and the others quantize to fp8 at load time —
# the 4080 (Ada) runs fp8 natively in vLLM. The reduced context below keeps two ~1024px
# images plus the prompt well under 8k tokens. Memory flags are tuned from observed OOMs;
# see the comment above COMMON before changing them.

set -euo pipefail

# The lab PC's system nvcc predates Ada (compute_89), so FlashInfer's JIT sampling
# kernel fails to build at warmup — use vLLM's native torch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Reduce fragmentation so transient vision-encoder peaks can use reserved-but-free memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 16 GB tuning, from where each higher setting OOM'd with Qwen3-VL-8B-FP8:
#   util 0.92 -> OOM capturing FULL CUDA graphs (hence PIECEWISE-only);
#   util 0.90 -> runtime OOM when 3 multi-image prefills hit the ViT encoder at once
#   (startup profiling under-counts that peak), hence 0.85 + --max-num-seqs 4.
# Client workers beyond 4 just queue. If it still OOMs: util 0.82, then --max-model-len 6144.
COMMON=(--port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85
        --max-num-seqs 4
        --limit-mm-per-prompt '{"image": 2}'
        --compilation-config '{"cudagraph_mode": "PIECEWISE"}')

case "${1:-}" in
  qwen3_instruct)
    vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 "${COMMON[@]}"
    ;;
  qwen3_thinking)
    # The reasoning parser routes the <think> block to reasoning_content, so the client's
    # message.content is only the final JSON answer.
    vllm serve Qwen/Qwen3-VL-8B-Thinking-FP8 "${COMMON[@]}" --reasoning-parser qwen3
    ;;
  internvl3)
    # No official FP8 release — quantize the bf16 weights to fp8 at load time.
    vllm serve OpenGVLab/InternVL3-8B "${COMMON[@]}" --trust-remote-code --quantization fp8
    ;;
  qwen25)
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct "${COMMON[@]}" --quantization fp8
    ;;
  *)
    echo "usage: $0 {qwen3_instruct|qwen3_thinking|internvl3|qwen25}" >&2
    exit 1
    ;;
esac
