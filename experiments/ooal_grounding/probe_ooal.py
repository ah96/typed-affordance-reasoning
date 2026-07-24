"""
Diagnostic for the OOAL adapter — run this when ooal_infer.py fails to construct the model.

It prints everything needed to adapt build_model() to whatever revision the clone provided:
the Net constructor signature, the arg fields the constructor actually reads, the checkpoint's
tensor shapes (which pin input_dim / out_dim), and how the upstream test.py builds the model.

Reads only. Nothing is loaded onto the GPU.

    python3 probe_ooal.py --ckpt ../../ooal_models_amar/seen_best --ooal_repo ooal_upstream
"""
import os
import re
import sys
import ast
import inspect
import argparse
import importlib

import torch


def find_net_class(repo):
    """Locate the model class the way ooal_infer.py does, but report every candidate."""
    sys.path.insert(0, os.path.abspath(repo))
    found = []
    for mod_name in ("models.ooal", "model.ooal", "models.model", "models.net", "model.net"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for cls_name in ("Net", "OOAL", "Model"):
            cls = getattr(mod, cls_name, None)
            if cls is not None and inspect.isclass(cls):
                found.append((mod_name, cls_name, cls))
    return found


def arg_fields_used(cls):
    """Which attributes of `args` does __init__ actually touch? (drives the stub we build)"""
    try:
        src = inspect.getsource(cls.__init__)
    except Exception:
        return []
    tree = ast.parse(src.lstrip())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "args":
            names.add(node.attr)
    return sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ooal_repo", default="ooal_upstream")
    args = ap.parse_args()

    print("=" * 70)
    print("1. MODEL CLASS")
    print("=" * 70)
    cands = find_net_class(args.ooal_repo)
    if not cands:
        print("no Net/OOAL/Model class found in the usual modules; listing model files:")
        for root, _, files in os.walk(args.ooal_repo):
            if os.sep + ".git" in root:
                continue
            for f in files:
                if f.endswith(".py") and ("model" in f or "net" in f or "ooal" in f):
                    print("   ", os.path.join(root, f))
    for mod_name, cls_name, cls in cands:
        print(f"\n{mod_name}.{cls_name}")
        try:
            print("  signature:", inspect.signature(cls.__init__))
        except Exception as e:
            print("  signature: <unavailable>", e)
        used = arg_fields_used(cls)
        print("  reads args.*:", used or "(none detected)")

    print()
    print("=" * 70)
    print("2. CHECKPOINT")
    print("=" * 70)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        print("top-level keys:", list(ckpt.keys())[:12])
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt \
        else ckpt
    if not isinstance(state, dict):
        print("unexpected checkpoint layout:", type(state))
        return
    print(f"tensors: {len(state)}")
    print("\nfirst 15 keys with shapes:")
    for k in list(state)[:15]:
        v = state[k]
        print(f"   {k:60s} {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
    print("\nlast 15 keys with shapes (the head usually pins out_dim):")
    for k in list(state)[-15:]:
        v = state[k]
        print(f"   {k:60s} {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")

    # Small dimensions that repeat across the head are the class-count candidates.
    small = {}
    for k, v in state.items():
        if hasattr(v, "shape"):
            for d in v.shape:
                if 2 <= int(d) <= 128:
                    small.setdefault(int(d), []).append(k)
    print("\ncandidate out_dim values (small dims, with an example tensor):")
    for d in sorted(small, key=lambda x: -len(small[x]))[:8]:
        print(f"   {d:4d}  seen in {len(small[d]):3d} tensors, e.g. {small[d][0]}")

    print()
    print("=" * 70)
    print("3. HOW UPSTREAM BUILDS IT (test.py / train.py)")
    print("=" * 70)
    pat = re.compile(r"(Net\s*\(|OOAL\s*\(|Model\s*\(|input_dim|out_dim|load_state_dict"
                     r"|add_argument\(\s*['\"]--)")
    for name in ("test.py", "train.py", "main.py"):
        path = os.path.join(args.ooal_repo, name)
        if not os.path.exists(path):
            continue
        print(f"\n--- {name} ---")
        for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
            if pat.search(line):
                print(f"  {i:4d}: {line.rstrip()[:150]}")


if __name__ == "__main__":
    main()
