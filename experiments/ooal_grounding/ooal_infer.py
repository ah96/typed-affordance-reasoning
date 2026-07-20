"""
OOAL checkpoint adapter — turns ooal_models_amar/{seen_best,unseen_best} into per-action
saliency maps. Lab PC only (CUDA). Requires the upstream OOAL codebase cloned somewhere:

    git clone https://github.com/Reagan1311/OOAL ooal_upstream

The checkpoints hold {iter, model_state_dict, merge_weight} with module prefixes
dino_model / aff_text_encoder / seg_decoder / embedder / prompt_learner / lln_* / linear_cls,
matching the upstream Net. All upstream-specific construction is isolated in build_model();
if the cloned revision's constructor differs, adapt only that function (or pass
--model_module/--model_class).

CLI dumps one float32 .npy heatmap (image size) per image x affordance:

    python3 ooal_infer.py --ckpt ../../ooal_models_amar/unseen_best \
        --ooal_repo ooal_upstream --images <dir with images> \
        --affordances sit_on,hold,carry,cut,throw,ride --outdir heatmaps/
"""
import os
import sys
import glob
import argparse
import importlib

import numpy as np
import torch
from PIL import Image


def build_model(ooal_repo, ckpt_path, device, model_module=None, model_class=None):
    """Construct the upstream OOAL net and load our checkpoint. Isolated on purpose:
    everything version-sensitive lives here."""
    sys.path.insert(0, os.path.abspath(ooal_repo))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    candidates = ([(model_module, model_class)] if model_module else
                  [("models.ooal", "Net"), ("model.ooal", "Net"), ("models.model", "Net")])
    last_err = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            break
        except Exception as e:
            last_err = e
    else:
        raise ImportError(
            f"Could not import the OOAL Net from {ooal_repo} (tried {candidates}, last error: "
            f"{last_err}). Find the model class in the cloned repo (their test.py shows it) and "
            f"pass --model_module/--model_class.")

    # Upstream Net(args)-style constructors want the training args; defaults matching the
    # released AGD20K configs. If construction fails, compare with the cloned repo's test.py.
    try:
        model = cls()
    except TypeError:
        import types
        args = types.SimpleNamespace(divide="Unseen" if "unseen" in ckpt_path.lower() else "Seen")
        model = cls(args)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 5 or (missing and len(missing) > 5):
        raise RuntimeError(
            f"State dict mismatch: {len(missing)} missing / {len(unexpected)} unexpected keys "
            f"(first missing: {missing[:3]}, first unexpected: {unexpected[:3]}). The cloned OOAL "
            f"revision does not match the checkpoints — check out the release matching Jul 2024.")
    model.merge_weight = ckpt.get("merge_weight")
    return model.to(device).eval()


@torch.no_grad()
def saliency(model, image, affordance, device, input_px=448):
    """One affordance text -> HxW float map in [0,1], resized to the image size.
    Assumes the upstream forward accepts (image_tensor, [affordance_text]) and returns a
    dense prediction; adapt here if the cloned revision's signature differs."""
    w, h = image.size
    im = image.convert("RGB").resize((input_px, input_px))
    x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device)
    out = model(x, [affordance.replace("_", " ")])
    if isinstance(out, (tuple, list)):
        out = out[0]
    m = out.squeeze().float().cpu().numpy()
    if m.ndim != 2:
        raise RuntimeError(f"Expected a dense HxW map from the model, got shape {m.shape}.")
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h))) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ooal_repo", default="ooal_upstream")
    ap.add_argument("--model_module", default=None)
    ap.add_argument("--model_class", default="Net")
    ap.add_argument("--images", default=None,
                    help="flat directory of images; each is scored for every --affordances entry")
    ap.add_argument("--tree", default=None,
                    help="AGD20K egocentric root <...>/testset/egocentric; walks <aff>/<obj>/*.jpg "
                         "and scores each image for its own affordance (one model load)")
    ap.add_argument("--affordances", default=None, help="comma-separated (required with --images)")
    ap.add_argument("--outdir", default="heatmaps")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.images and not args.tree:
        ap.error("pass --images <dir> --affordances a,b,c  OR  --tree <egocentric_root>")

    model = build_model(args.ooal_repo, args.ckpt, args.device,
                        args.model_module, args.model_class)
    os.makedirs(args.outdir, exist_ok=True)

    # build the (image_path, affordance) work list
    jobs = []
    if args.tree:
        for aff in sorted(os.listdir(args.tree)):
            adir = os.path.join(args.tree, aff)
            if not os.path.isdir(adir):
                continue
            for e in ("*.jpg", "*.png"):
                for p in glob.glob(os.path.join(adir, "*", e)):
                    jobs.append((p, aff))
    else:
        actions = [a.strip() for a in (args.affordances or "").split(",") if a.strip()]
        if not actions:
            ap.error("--images requires --affordances")
        for e in ("*.jpg", "*.png"):
            for p in sorted(glob.glob(os.path.join(args.images, e))):
                jobs += [(p, a) for a in actions]

    print(f"{len(jobs)} (image, affordance) heatmaps to compute")
    for p, a in jobs:
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.outdir, f"{stem}__{a}.npy")
        if os.path.exists(out):
            continue
        np.save(out, saliency(model, Image.open(p), a, args.device).astype(np.float32))
    print(f"wrote heatmaps to {args.outdir}")


if __name__ == "__main__":
    main()
