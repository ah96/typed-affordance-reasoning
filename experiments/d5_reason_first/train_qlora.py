"""
D5 Stage 1 — QLoRA fine-tune of Qwen3-VL-8B for reason-first typed affordance prediction.

Stage 0 established the diagnosis this training run is built on. Models frequently perceive
the correct reason and then file it under the wrong code, and a text-only mapper from reason
to code saturates at 0.374 even on human explanations, so the reason alone under-determines
the type. Stage 1 therefore learns the reason and the code together, conditioned on pixels.

The ordering ablation is the point of the experiment, not a detail:

    --order reason_first   target = {"explanation", "consequence", "relationship_id"}
    --order label_first    target = {"relationship_id", "explanation", "consequence"}

Both targets contain identical tokens and differ only in order, so any gap isolates the
effect of committing to a code before or after articulating the reason. label_first
reproduces the schema the evaluated VLMs were prompted with, and is the control.

The prompt, the two-image layout, and the crop policy are copied from
experiment_a/eval_experiment_a_vision.py so that a tuned checkpoint can be served with vLLM
and scored by the *same* eval script as every other model in the paper.

16 GB notes. The base is loaded in 4-bit NF4 with bf16 compute, the vision tower is frozen
(we are teaching the reason-to-code mapping, not re-teaching perception), gradient
checkpointing is on, and images are capped well below the eval's resolution by default
because training holds activations that inference does not. If it OOMs, lower --full_px
first, then --crop_px, then raise --grad_accum while keeping --batch_size 1.

    python3 train_qlora.py --data data/train_balanced.jsonl --order reason_first
    python3 train_qlora.py --data data/train_balanced.jsonl --order label_first \
        --out runs/label_first
"""
import os
import io
import json
import math
import random
import argparse

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# Copied verbatim from experiment_a/eval_experiment_a_vision.py so training and evaluation
# see byte-identical instructions. Do not paraphrase.
SYSTEM_PROMPT = """You are evaluating affordances in images under a CLOSED ontology.
You must follow the label taxonomy EXACTLY and output STRICT JSON only.

Relationship label ids:
0: Positive
1: FirmlyNegative
2: ObjectNonFunctional
3: PhysicalObstacle
4: SociallyAwkward
5: SociallyForbidden
6: Dangerous

Return schema:
{"relationship_id": <int 0..6>, "explanation": <string>, "consequence": <string>}

Rules:
- Always output relationship_id.
- If relationship_id is 0 or 1, explanation and consequence must be empty strings.
- If relationship_id is 2..6, explanation and consequence must be ONE short sentence each.
"""

ACTION_PHRASE = {"sit": "sit on", "run": "run on", "grasp": "grasp"}


def build_user_prompt(action: str) -> str:
    phrase = ACTION_PHRASE.get(action, action)
    return (
        f"The first image is the full scene. The second image is a crop of ONE target object.\n"
        f"Question: for the target object, is the action \"{phrase}\" appropriate?\n"
        f"Classify the relationship (0..6) and, for exception categories (2..6), give a "
        f"one-sentence explanation and consequence. Output STRICT JSON only."
    )


def target_json(rec, order):
    """The string the model must produce. Same content, two orderings."""
    code, expl, cons = rec["code"], rec["explanation"], rec["consequence"]
    if code in (0, 1):
        expl = cons = ""
    if order == "reason_first":
        obj = [("explanation", expl), ("consequence", cons), ("relationship_id", code)]
    else:
        obj = [("relationship_id", code), ("explanation", expl), ("consequence", cons)]
    body = ", ".join(f'"{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in obj)
    return "{" + body + "}"


def shrink(img, max_px):
    w, h = img.size
    s = max_px / max(w, h)
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
    return img.convert("RGB")


def crop_instance(img, mask, pad=0.12):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    px, py = int(w * pad), int(h * pad)
    W, H = img.size
    return img.crop((max(0, x1 - px), max(0, y1 - py),
                     min(W, x2 + px + 1), min(H, y2 + py + 1)))


class AffordanceDataset:
    """Yields (messages, target_text) pairs. Pixels are resolved lazily per item."""

    def __init__(self, path, images_dir, seg_dir, order, full_px, crop_px):
        self.records = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.images_dir, self.seg_dir = images_dir, seg_dir
        self.order, self.full_px, self.crop_px = order, full_px, crop_px
        self._seg_cache = (None, None)

    def __len__(self):
        return len(self.records)

    def _seg(self, image):
        if self._seg_cache[0] != image:
            arr = np.array(Image.open(os.path.join(self.seg_dir, f"{image}_seg.png")))
            self._seg_cache = (image, arr[:, :, 2])       # blue channel = instance id
        return self._seg_cache[1]

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(os.path.join(self.images_dir, f"{r['image']}.jpg"))
        crop = crop_instance(img, self._seg(r["image"]) == r["instance_id"])
        if crop is None:
            return None
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": shrink(img, self.full_px)},
                {"type": "image", "image": shrink(crop, self.crop_px)},
                {"type": "text", "text": build_user_prompt(r["action"])},
            ]},
        ]
        return messages, target_json(r, self.order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "train_balanced.jsonl"))
    ap.add_argument("--images_dir", default=os.path.join(HERE, "data", "images_train"))
    ap.add_argument("--seg_dir", default=os.path.join(HERE, "data", "instance_seg_train"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--order", choices=["reason_first", "label_first"], default="reason_first")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--full_px", type=int, default=512,
                    help="Training-time cap on the full image (eval uses 1024).")
    ap.add_argument("--crop_px", type=int, default=384,
                    help="Training-time cap on the instance crop (eval uses 768).")
    ap.add_argument("--max_steps", type=int, default=None, help="Debug: stop early.")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="Build a few samples and print them. No model, no GPU.")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(HERE, "runs", args.order)

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = AffordanceDataset(args.data, args.images_dir, args.seg_dir,
                           args.order, args.full_px, args.crop_px)
    print(f"dataset {len(ds)} records | order={args.order} | out={out_dir}")

    if args.dry_run:
        for i in range(min(3, len(ds))):
            item = ds[i]
            if item is None:
                print(f"[{i}] no mask pixels, skipped")
                continue
            messages, target = item
            sizes = [c["image"].size for c in messages[1]["content"] if c["type"] == "image"]
            print(f"\n--- sample {i} --- images {sizes}")
            print(messages[1]["content"][-1]["text"])
            print("TARGET:", target)
        return

    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig, get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Adapt the language model only. The vision tower stays frozen because Stage 0 located
    # the failure in the reason-to-code mapping, not in perception, and freezing it is also
    # what keeps this run inside 16 GB.
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*(visual|vision_tower|vision_model).*",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        texts, image_lists = [], []
        for messages, target in batch:
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            texts.append(prompt + target + processor.tokenizer.eos_token)
            image_lists.append([c["image"] for c in messages[1]["content"]
                                if c["type"] == "image"])
        enc = processor(text=texts, images=image_lists, return_tensors="pt", padding=True)

        # Supervise the answer only. Everything up to the generation prompt is context, so
        # its labels are masked out and the loss lands on the JSON the model must produce.
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, (messages, target) in enumerate(batch):
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            n_prompt = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            n_pad = int((enc["attention_mask"][i] == 0).sum()) \
                if processor.tokenizer.padding_side == "left" else 0
            labels[i, : n_pad + n_prompt] = -100
        enc["labels"] = labels
        return enc

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total_steps), total_steps)
    print(f"steps/epoch {steps_per_epoch} | total optimizer steps {total_steps}")

    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=1)

    model.train()
    step, run_loss, seen = 0, 0.0, 0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for micro, enc in enumerate(loader):
            if enc is None:
                continue
            enc = {k: v.to("cuda") for k, v in enc.items()}
            loss = model(**enc).loss / args.grad_accum
            loss.backward()
            run_loss += loss.item() * args.grad_accum
            seen += 1
            if (micro + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"loss {run_loss / max(1, seen):.4f} peak {mem:.1f} GiB", flush=True)
                run_loss, seen = 0.0, 0
            if step % args.save_every == 0 or step == total_steps:
                model.save_pretrained(os.path.join(out_dir, f"step{step}"))
                print(f"  saved step{step}", flush=True)
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(os.path.join(out_dir, "final"))
    print(f"done -> {os.path.join(out_dir, 'final')}")


if __name__ == "__main__":
    main()
