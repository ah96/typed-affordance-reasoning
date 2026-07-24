"""
OOAL checkpoint adapter — turns ooal_models_amar/{seen_best,unseen_best} into per-action
saliency maps. Lab PC only (CUDA). Requires the upstream OOAL codebase cloned somewhere:

    git clone https://github.com/Reagan1311/OOAL ooal_upstream

The checkpoints hold {iter, model_state_dict, optimizer_state_dict} with module prefixes
dino_model / aff_text_encoder / seg_decoder / embedder / prompt_learner / lln_* / linear_cls,
matching the upstream Net. All upstream-specific construction is isolated in build_model();
if the cloned revision's constructor differs, adapt only that function (or pass
--model_module/--model_class).

The affordance vocabulary is CLOSED: the prompt learner is trained per class, so a checkpoint
only knows its own list (seen_best = the 36 SEEN_AFF names, unseen_best = the 25 UNSEEN_AFF
names, both read from the cloned repo's data/agd20k_ego.py). Requested affordances outside a
checkpoint's list are skipped with a warning, not silently mapped.

CLI dumps one float32 .npy heatmap (image size) per image x affordance:

    python3 ooal_infer.py --ckpt ../../ooal_models_amar/unseen_best \
        --ooal_repo ooal_upstream --images <dir with images> \
        --affordances sit_on,hold,carry,cut,throw,ride --outdir heatmaps/
"""
import os
import sys
import glob
import types
import inspect
import argparse
import importlib

import numpy as np
import torch
from PIL import Image


def class_names(ooal_repo, state):
    """The checkpoint's affordance list, pinned by the prompt learner's class dimension.
    Names must match training order exactly or prompt_learner/token_* will not load."""
    n = None
    for k in ("prompt_learner.token_prefix", "prompt_learner.token_suffix"):
        if k in state:
            n = int(state[k].shape[0])
            break
    if n is None:
        raise RuntimeError("checkpoint has no prompt_learner.token_* tensor to pin the "
                           "affordance count — run probe_ooal.py")
    try:
        ego = importlib.import_module("data.agd20k_ego")
        lists = [ego.SEEN_AFF, ego.UNSEEN_AFF]
    except Exception as e:
        raise ImportError(f"could not import data.agd20k_ego from {ooal_repo} to get the "
                          f"affordance names ({e})")
    for names in lists:
        if len(names) == n:
            return list(names)
    raise RuntimeError(f"checkpoint expects {n} affordances but the repo's lists have "
                       f"{[len(x) for x in lists]} — wrong OOAL revision?")


def _construct(cls, state, names):
    """Upstream constructors vary by revision: Net(), Net(args) and Net(args, input_dim,
    out_dim) all exist in the wild. Read the signature and fill each required parameter from
    the checkpoint, which is the only authoritative source for the dimensions."""
    args = types.SimpleNamespace(class_names=names, crop_size=224, resize_size=256)
    supply = {"args": args,
              "input_dim": int(state["embedder.fc1.weight"].shape[1]),   # 768, DINOv2 ViT-B
              "out_dim": int(state["linear_cls.weight"].shape[0])}       # 512, CLIP text width
    required = [p for p in list(inspect.signature(cls.__init__).parameters.values())[1:]
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    unknown = [p.name for p in required if p.name not in supply]
    if unknown:
        raise RuntimeError(f"{cls.__name__}.__init__ requires {unknown}, which this adapter "
                           f"cannot fill from the checkpoint — run probe_ooal.py and extend "
                           f"_construct()")
    return cls(*[supply[p.name] for p in required])


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

    # The upstream Net builds its prompt learner from args.class_names, so the vocabulary has
    # to come from the checkpoint before construction. Run probe_ooal.py if this still fails.
    names = class_names(ooal_repo, state)
    model = _construct(cls, state, names)
    model.class_names = names

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 5 or (missing and len(missing) > 5):
        raise RuntimeError(
            f"State dict mismatch: {len(missing)} missing / {len(unexpected)} unexpected keys "
            f"(first missing: {missing[:3]}, first unexpected: {unexpected[:3]}). The cloned OOAL "
            f"revision does not match the checkpoints — check out the release matching Jul 2024.")
    return model.to(device).eval()


@torch.no_grad()
def saliency(model, image, affordance, device, input_px=224):
    """One affordance -> HxW float map in [0,1], resized to the image size. Preprocessing
    mirrors the upstream TestData transform (plain resize to crop_size, ImageNet stats) and
    the forward is called the way their test.py does it, with the affordance's class index."""
    idx = model.class_names.index(affordance)
    w, h = image.size
    im = image.convert("RGB").resize((input_px, input_px))
    x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device)
    out = model(x, gt_aff=[idx])
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out is None:
        raise RuntimeError("the model returned None — this revision's forward does not select "
                           "a class from gt_aff in eval mode; compare with its test.py")
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
    ap.add_argument("--input_px", type=int, default=224, help="upstream test crop_size")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.images and not args.tree:
        ap.error("pass --images <dir> --affordances a,b,c  OR  --tree <egocentric_root>")
    if not torch.cuda.is_available():
        ap.error("the upstream Net hardcodes .cuda() in its constructor and forward — this "
                 "needs a CUDA machine (lab PC)")

    model = build_model(args.ooal_repo, args.ckpt, args.device,
                        args.model_module, args.model_class)
    os.makedirs(args.outdir, exist_ok=True)

    # build the (image_path, affordance) work list, dropping affordances this checkpoint was
    # not trained on (seen_best knows 36, unseen_best only 25 — mixing them is a silent error)
    known = set(model.class_names)
    jobs, skipped = [], set()
    if args.tree:
        for aff in sorted(os.listdir(args.tree)):
            adir = os.path.join(args.tree, aff)
            if not os.path.isdir(adir):
                continue
            if aff not in known:
                skipped.add(aff)
                continue
            for e in ("*.jpg", "*.png"):
                for p in glob.glob(os.path.join(adir, "*", e)):
                    jobs.append((p, aff))
    else:
        actions = [a.strip() for a in (args.affordances or "").split(",") if a.strip()]
        if not actions:
            ap.error("--images requires --affordances")
        skipped = {a for a in actions if a not in known}
        actions = [a for a in actions if a in known]
        for e in ("*.jpg", "*.png"):
            for p in sorted(glob.glob(os.path.join(args.images, e))):
                jobs += [(p, a) for a in actions]
    if skipped:
        print(f"skipping {len(skipped)} affordance(s) absent from this checkpoint: "
              f"{sorted(skipped)}\n  known: {sorted(known)}")
    if not jobs:
        raise SystemExit("nothing to compute — no image matched a known affordance")

    print(f"{len(jobs)} (image, affordance) heatmaps to compute")
    for p, a in jobs:
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.outdir, f"{stem}__{a}.npy")
        if os.path.exists(out):
            continue
        np.save(out, saliency(model, Image.open(p), a, args.device,
                              input_px=args.input_px).astype(np.float32))
    print(f"wrote heatmaps to {args.outdir}")


if __name__ == "__main__":
    main()
