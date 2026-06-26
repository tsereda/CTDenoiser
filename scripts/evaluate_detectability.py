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

from ctdenoiser.detectability import (  # noqa: E402
    cho_detectability,
    extract_rois,
    insert_signal,
    sample_flat_locations,
    signal_template,
)
from ctdenoiser.inference import overlapped_inference  # noqa: E402
from ctdenoiser.metrics import uniform_nps  # noqa: E402
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


def evaluate_detectability(denoise, loader, device, args):
    """Accumulate present/absent ROI ensembles over a loader and score the CHO.

    For each slice we sample flat lesion sites on the clean reference, insert the
    same known lesion at every site (sites are kept >= ``roi_size`` apart so the
    nonlinear denoiser barely couples them), and build:

        low_present = low + s,  low_absent = low                 (input stage)
        denoised(low_present), denoised(low_absent)              (denoised stage)
        clean + s,   clean                                       (clean ceiling)

    ROIs are extracted at the known sites and pooled across all slices, then a
    single CHO ``d'`` is computed per stage. ``d'_denoised / d'_input`` answers
    "is detectability preserved"; the clean ceiling bounds what is achievable.
    """
    roi = args.roi_size
    # The SKE template: the lesion as it sits at the centre of an ROI.
    template = signal_template(
        (roi, roi), (roi // 2, roi // 2), args.radius_px,
        args.contrast_hu, args.hu_scale, profile=args.profile,
        device=device, dtype=torch.float32,
    )

    stages = ("input", "denoised", "clean")
    present = {s: [] for s in stages}
    absent = {s: [] for s in stages}
    nps_in, nps_out = [], []

    n_slices = 0
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        clean = full[0, 0]
        sites = sample_flat_locations(
            clean, args.sites_per_slice, roi, var_quantile=args.var_quantile,
            seed=args.seed + n_slices,
        )
        if not sites:
            continue

        # One signal map with a lesion at every (well-separated) site.
        s_map = torch.zeros_like(clean)
        for cy, cx in sites:
            s_map = s_map + signal_template(
                clean.shape, (cy, cx), args.radius_px, args.contrast_hu,
                args.hu_scale, profile=args.profile,
                device=device, dtype=clean.dtype,
            )
        s_map = s_map[None, None]

        low_present, low_absent = low + s_map, low
        clean_present, clean_absent = full + s_map, full
        den_present = denoise(low_present)
        den_absent = denoise(low_absent)

        imgs = {
            "input": (low_present, low_absent),
            "denoised": (den_present, den_absent),
            "clean": (clean_present, clean_absent),
        }
        for st, (pi, ai) in imgs.items():
            present[st].append(extract_rois(pi[0, 0], sites, roi))
            absent[st].append(extract_rois(ai[0, 0], sites, roi))

        # Reference-free NPS: input noise vs denoised-output noise on flat ROIs.
        nps_in.append(uniform_nps(low_absent[0, 0], sites, roi_size=roi))
        nps_out.append(uniform_nps(den_absent[0, 0], sites, roi_size=roi))

        n_slices += 1
        if args.max_slices and n_slices >= args.max_slices:
            break

    if n_slices == 0:
        raise RuntimeError("no slices yielded usable flat ROIs")

    results = {"n_slices": n_slices}
    for st in stages:
        p = torch.cat(present[st], 0)
        a = torch.cat(absent[st], 0)
        cho = cho_detectability(p, a, signal=template, n_channels=args.n_channels)
        results[f"d_prime_{st}"] = cho["d_prime"]
        results[f"auc_{st}"] = cho["auc"]
        results[f"n_rois_{st}"] = cho["n_present"]

    d_in = results["d_prime_input"]
    results["detectability_preserved"] = (
        results["d_prime_denoised"] / d_in if d_in > 0 else float("nan")
    )
    # Mean noise spectral centroid: a drop after denoising = blotchier texture.
    results["nps_mean_freq_input"] = sum(x["mean_freq"] for x in nps_in) / n_slices
    results["nps_mean_freq_denoised"] = sum(x["mean_freq"] for x in nps_out) / n_slices
    results["noise_power_input"] = sum(x["total_power"] for x in nps_in) / n_slices
    results["noise_power_denoised"] = sum(x["total_power"] for x in nps_out) / n_slices
    return results


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
    p.add_argument("--contrast-hu", type=float, default=12.0,
                   help="lesion contrast in HU (keep low so noise dominates)")
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
    print(f"device={device}  method={method}  contrast={args.contrast_hu} HU")

    res = evaluate_detectability(denoise, val_loader, device, args)
    res = {"method": method, **res}

    print(
        f"slices={res['n_slices']}  "
        f"d'(input)={res['d_prime_input']:.3f}  "
        f"d'(denoised)={res['d_prime_denoised']:.3f}  "
        f"d'(clean)={res['d_prime_clean']:.3f}  "
        f"preserved={res['detectability_preserved']:.3f}  "
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
