#!/usr/bin/env python3
"""Offline hallucination-aware detectability evaluation (Phase 1 harness).

Scores a denoiser on the task-based metrics that PSNR/SSIM are blind to: does a
known low-contrast lesion survive denoising (preserved), get washed out
(erased), or get invented where none exists (fabricated)? See ``docs/plan.md``.

The method needs no CT projector. The dataset's real paired low/full slices give
the real, spatially-correlated FBP noise for free (``n = low - full``); a
signal-present low-dose image is then literally ``low + s`` for a known lesion
``s`` (:mod:`ctdenoiser.detectability`). Present/absent ROI ensembles are pushed
through the denoiser and a Channelized Hotelling Observer reports ``d'`` for the
**input**, the **denoised** output, and the **clean** reference (the ceiling).

This is deliberately kept *out* of ``train.py:evaluate``: the CHO needs many
realizations x ROIs, far too expensive for the per-epoch loop. The cheap
PSNR/SSIM eval stays where it is.

Usage
-----
    # identity baseline (no checkpoint): the floor every model must beat
    python scripts/evaluate_detectability.py --h5-path cache.h5 --identity

    # a trained checkpoint
    python scripts/evaluate_detectability.py --h5-path cache.h5 \
        --model redcnn --checkpoint checkpoints/redcnn.pt --out det.csv

    # quick smoke test on synthetic data (no real dataset needed)
    python scripts/evaluate_detectability.py --identity --max-slices 4 \
        --contrast-hu 60 --patch-size 64
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctdenoiser.detectability import run_detectability_eval  # noqa: E402
from ctdenoiser.inference import overlapped_inference  # noqa: E402
from ctdenoiser.train import MODELS, build_loaders  # noqa: E402


def _denoiser(args, device):
    """Return a callable ``low -> denoised`` (identity, or a loaded checkpoint).

    The denoiser is model-agnostic and reuses the same ``overlapped_inference``
    path the rest of the benchmark uses, so detectability is measured on exactly
    the images a deployed model would produce.
    """
    if args.identity:
        return lambda low: low

    if args.model == "flowmatching":
        from ctdenoiser.models import FlowMatching
        model = FlowMatching(num_steps=args.flow_steps).to(device)
    else:
        model = MODELS[args.model]().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    def run(low):
        return overlapped_inference(
            model, low, patch_size=args.patch_size, margin=args.patch_size // 4
        ).clamp(0.0, 1.0)

    return run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="redcnn")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to a trained model .pt (omit with --identity)")
    p.add_argument("--identity", action="store_true",
                   help="score the noisy input directly (do-nothing floor)")
    # Data sources (mirror train.py).
    p.add_argument("--h5-path", type=str, default=None)
    p.add_argument("--dicom-root", type=str, default=None)
    p.add_argument("--anatomy", default="abdomen")
    p.add_argument("--hu-offset", type=float, default=None)
    p.add_argument("--hu-scale", type=float, default=None,
                   help="HU window width (defaults to the --anatomy preset); "
                        "lesion contrast is contrast_hu/hu_scale")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--synthetic-len", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--flow-steps", type=int, default=20)
    # Signal / ROI / observer knobs.
    p.add_argument("--contrast-hu", type=float, nargs="+", default=[40.0, 80.0, 160.0],
                   metavar="HU",
                   help="lesion contrast(s) in HU; multiple values sweep a "
                        "contrast-detail curve (det/c{hu}/*), highest = headline")
    p.add_argument("--radius-px", type=float, default=4.0)
    p.add_argument("--profile", choices=["disk", "gaussian"], default="gaussian")
    p.add_argument("--roi-size", type=int, default=32)
    p.add_argument("--sites-per-slice", type=int, default=6)
    p.add_argument("--var-quantile", type=float, default=0.3)
    p.add_argument("--n-channels", type=int, default=10)
    p.add_argument("--max-slices", type=int, default=0,
                   help="cap slices evaluated (0 = all val slices)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="append a row to this CSV")
    args = p.parse_args(argv)

    if not args.identity and not args.checkpoint:
        p.error("provide --checkpoint MODEL.pt or use --identity")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Resolve the HU window like train.py so --dicom-root normalisation and the
    # contrast_hu/hu_scale conversion agree (h5 caches override this from attrs).
    from ctdenoiser.data.dataset import window_for_anatomy
    _offset, _scale = window_for_anatomy(args.anatomy)
    args.hu_offset = _offset if args.hu_offset is None else args.hu_offset
    args.hu_scale = _scale if args.hu_scale is None else args.hu_scale
    train_loader, val_loader, _ = build_loaders(args)

    denoise = _denoiser(args, device)
    method = "identity" if args.identity else args.model
    print(f"device={device}  method={method}  contrasts={args.contrast_hu} HU")

    res = run_detectability_eval(
        denoise, val_loader, device, hu_scale=args.hu_scale,
        contrast_hu=args.contrast_hu, radius_px=args.radius_px, profile=args.profile,
        roi_size=args.roi_size, sites_per_slice=args.sites_per_slice,
        var_quantile=args.var_quantile, n_channels=args.n_channels,
        max_slices=args.max_slices, seed=args.seed,
    )
    res = {"method": method, **res}

    print(
        f"slices={res['n_slices']}  "
        f"d'(input)={res['d_prime_input']:.3f}  "
        f"d'(denoised)={res['d_prime_denoised']:.3f}  "
        f"d'(clean)={res['d_prime_clean']:.3f}  "
        f"preserved={res['detectability_preserved']:.3f}  "
        f"fabrication d'={res['d_prime_fabrication']:.3f} "
        f"(floor {res['d_prime_fabrication_input']:.3f})  "
        f"nps_freq {res['nps_mean_freq_input']:.3f}->{res['nps_mean_freq_denoised']:.3f}"
    )

    if args.out:
        out = Path(args.out)
        write_header = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res.keys()))
            if write_header:
                w.writeheader()
            w.writerow(res)
        print(f"appended -> {out}")
    return res


if __name__ == "__main__":
    main()
