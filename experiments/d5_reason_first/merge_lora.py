"""
Merge a D5 LoRA adapter into the base weights and save a standalone model.

Training runs in 4-bit, but merging into a 4-bit base would bake in quantization error, so
the base is reloaded in bf16 here and the adapter is merged into that. The result is a
plain Qwen3-VL checkpoint that serve_vllm.sh can host with on-the-fly fp8, which means the
tuned model is scored by exactly the same eval script as every other model in the paper.

Needs enough CPU RAM to hold the bf16 model (~17 GB); it never touches the GPU.

    python3 merge_lora.py --adapter runs/reason_first/final --out merged/reason_first
"""
import os
import argparse

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=os.path.join(HERE, "runs", "reason_first", "final"))
    ap.add_argument("--out", default=os.path.join(HERE, "merged", "reason_first"))
    args = ap.parse_args()

    print(f"loading base in bf16 on CPU: {args.base}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu")
    print(f"applying adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model -> {args.out}")
    print("serve it with:\n"
          f"  vllm serve {args.out} --served-model-name d5_reason_first \\\n"
          "    --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 \\\n"
          "    --max-num-seqs 4 --limit-mm-per-prompt '{\"image\": 2}' \\\n"
          "    --quantization fp8 --compilation-config '{\"cudagraph_mode\": \"PIECEWISE\"}'")


if __name__ == "__main__":
    main()
