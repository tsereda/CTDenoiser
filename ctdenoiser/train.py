"""Training / benchmarking entry point.

Examples
--------
    # synthetic smoke run
    python -m ctdenoiser.train --model ctformer --epochs 1

    # DICOM series directories (patient-split, full-slice overlapped-inference eval)
    python -m ctdenoiser.train --model ctformer \
        --dicom-root /data/ldct_dicom --epochs 50 --batch-size 16
"""

import argparse
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from .data.dataset import (
    ANATOMY_WINDOWS,
    DICOMCTDataset,
    HDF5CTDataset,
    SyntheticCTDataset,
    window_for_anatomy,
)
from .inference import overlapped_inference
from .metrics import gmsd, nps_ratio, psnr, rmse, ssim
from .models import CTformer, DnCNN, FlowMatching, REDCNN, SelfSupervisedFlow, UNet
from .selfsupervised import n2sim_training_step, n2v_training_step
from .zeroshot import denoise_image as zsn2n_denoise_image
from .zeroshot import denoise_image_f2n as f2n_denoise_image

MODELS = {
    "ctformer": CTformer,
    "dncnn": DnCNN,
    "flowmatching": FlowMatching,
    "redcnn": REDCNN,
    "ssflow": SelfSupervisedFlow,
    "unet": UNet,
}


def model_stats(model):
    """Parameter count and on-disk size of a model.

    The efficiency axis of the benchmark: a denoiser is only useful if its
    quality justifies its size / compute, so every run logs these alongside the
    image-quality metrics.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4 bytes/param (float32) -> MB; matches a saved state_dict closely enough.
    return {
        "param_count": total,
        "trainable_params": trainable,
        "model_size_mb": total * 4 / 1e6,
    }


def _git_sha():
    """Short git commit of the working tree, or None outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.SubprocessError, OSError):
        return None


def provenance(args, device):
    """Reproducibility metadata: code version, environment, resolved window."""
    info = {
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "device": str(device),
        "anatomy": getattr(args, "anatomy", None),
        "hu_offset": getattr(args, "hu_offset", None),
        "hu_scale": getattr(args, "hu_scale", None),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


def dataset_provenance(train_loader, val_loader, full_slice):
    """Sizes and the exact val patient IDs, so a run's split is reproducible."""
    info = {
        "n_train_slices": len(train_loader.dataset),
        "n_val_slices": len(val_loader.dataset),
    }
    if full_slice:
        val_pids = sorted(getattr(val_loader.dataset, "low_volumes", {}).keys())
        train_pids = getattr(train_loader.dataset, "low_volumes", {})
        info["n_train_patients"] = len(train_pids)
        info["n_val_patients"] = len(val_pids)
        info["val_patient_ids"] = ",".join(val_pids)
    return info


def build_loaders(args):
    """Return (train_loader, val_loader, full_slice_eval)."""
    if getattr(args, "h5_path", None):
        # The window is baked into the cache; reflect its real anatomy/window in
        # the logged provenance rather than the (ignored) --anatomy default.
        try:
            import h5py
            with h5py.File(args.h5_path, "r") as f:
                args.anatomy = str(f.attrs.get("anatomy", args.anatomy))
                args.hu_offset = float(f.attrs.get("hu_offset", args.hu_offset))
                args.hu_scale = float(f.attrs.get("hu_scale", args.hu_scale))
        except (OSError, KeyError):
            pass
        train_p, val_p = HDF5CTDataset.split_patients(
            args.h5_path, val_fraction=args.val_fraction, seed=args.seed
        )
        print(
            f"HDF5: {len(train_p)} train / {len(val_p)} val patients "
            f"({train_p[:3]}... | {val_p})"
        )
        train_ds = HDF5CTDataset(
            args.h5_path, train_p, patch_size=args.patch_size, train=True
        )
        val_ds = HDF5CTDataset(
            args.h5_path, val_p, patch_size=args.patch_size, train=False
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        return train_loader, val_loader, True

    if args.dicom_root:
        train_p, val_p = DICOMCTDataset.split_patients(
            args.dicom_root, val_fraction=args.val_fraction, seed=args.seed
        )
        print(
            f"DICOM root: {len(train_p)} train / {len(val_p)} val patients "
            f"({train_p[:3]}... | {val_p})"
        )
        train_ds = DICOMCTDataset(
            args.dicom_root, train_p, patch_size=args.patch_size, train=True,
            hu_offset=args.hu_offset, hu_scale=args.hu_scale,
        )
        val_ds = DICOMCTDataset(
            args.dicom_root, val_p, patch_size=args.patch_size, train=False,
            hu_offset=args.hu_offset, hu_scale=args.hu_scale,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        return train_loader, val_loader, True

    print("No --h5-path or --dicom-root; using SyntheticCTDataset.")
    ds = SyntheticCTDataset(
        length=args.synthetic_len, patch_size=args.patch_size
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    return loader, loader, False


_METRIC_FNS = {"psnr": psnr, "ssim": ssim, "rmse": rmse, "gmsd": gmsd, "nps_ratio": nps_ratio}


def _summarize(per_sample):
    """Mean and per-sample std for each metric in ``per_sample``.

    ``per_sample`` maps a metric name to a list of per-slice scalar values.
    Returns ``{metric}`` (mean) and ``{metric}_std`` (spread across slices) so
    every eval path produces identically-shaped error-bar-ready dicts.
    """
    import numpy as np

    out = {}
    for k, vals in per_sample.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[k] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std())
    return out


@torch.no_grad()
def evaluate(model, loader, device, full_slice, patch_size, eval_steps=None):
    """Mean + std of each metric plus inference latency / peak memory.

    Returns mean ``{metric}`` and spread ``{metric}_std`` (error bars across
    eval samples), and timing: ``latency_ms`` (per slice) and ``peak_mem_mb``
    (CUDA only). Latency is wall-clock around the forward path only, so it is
    the deployable inference cost, independent of dataset size.
    """
    model.eval()
    _orig_steps = getattr(model, "num_steps", None)
    if _orig_steps is not None and eval_steps is not None:
        model.num_steps = eval_steps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    per_sample = {k: [] for k in _METRIC_FNS}
    infer_s, n = 0.0, 0
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if full_slice:
            pred = overlapped_inference(
                model, low, patch_size=patch_size, margin=patch_size // 4
            ).clamp(0.0, 1.0)
        else:
            pred = model(low).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_s += time.perf_counter() - t0
        bs = low.size(0)
        for k, fn in _METRIC_FNS.items():
            per_sample[k].append(fn(pred, full))
        n += bs

    if _orig_steps is not None:
        model.num_steps = _orig_steps

    out = _summarize(per_sample)
    out["latency_ms"] = 1000.0 * infer_s / max(n, 1)
    if device.type == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
    return out


@torch.no_grad()
def identity_baseline(loader, device):
    """Score the noisy input directly against the clean reference (no model).

    This is the "do nothing" denoiser: ``pred = low``. It establishes the floor
    every trained model must beat and is the reference against which the
    per-image zsn2n results should be read (a near-identity output scores high
    when ``low`` is already close to ``full``). Returns mean and per-slice
    ``{metric}_std`` keys, matching :func:`evaluate`.
    """
    per_sample = {k: [] for k in _METRIC_FNS}
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        pred = low.clamp(0.0, 1.0)
        for k, fn in _METRIC_FNS.items():
            per_sample[k].append(fn(pred, full))
    return _summarize(per_sample)


def run_zsn2n_eval(loader, device, args):
    """Per-image Zero-Shot Noise2Noise evaluation.

    For each noisy slice in ``loader``, a fresh tiny network is trained from
    scratch on that image alone (no shared weights, no checkpoint) and the
    denoised result is scored against the clean ``full`` reference. Returns mean
    and per-slice ``{metric}_std`` keys, matching :func:`evaluate` so results are
    directly comparable to the supervised / N2V models.
    """
    per_sample = {k: [] for k in _METRIC_FNS}
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        pred = zsn2n_denoise_image(
            low,
            num_iters=args.zsn2n_iters,
            lr=args.zsn2n_lr,
            num_channels=args.zsn2n_channels,
            device=device,
            seed=args.seed,
        ).clamp(0.0, 1.0)
        # denoise_image may crop odd dims to even; align the reference to match.
        full = full[..., : pred.shape[-2], : pred.shape[-1]]
        for k, fn in _METRIC_FNS.items():
            per_sample[k].append(fn(pred, full))
    return _summarize(per_sample)


def run_f2n_eval(loader, device, args):
    """Per-image Filter2Noise evaluation.

    For each noisy slice in ``loader``, a fresh attention-guided bilateral
    filter stack is trained from scratch on that image alone (no shared weights,
    no checkpoint) and the denoised result is scored against the clean ``full``
    reference. Returns mean and per-slice ``{metric}_std`` keys, matching
    :func:`evaluate` so results are directly comparable to the other methods.
    """
    per_sample = {k: [] for k in _METRIC_FNS}
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        pred = f2n_denoise_image(
            low,
            num_iters=args.f2n_iters,
            lr=args.f2n_lr,
            num_layers=args.f2n_layers,
            radius=args.f2n_radius,
            num_channels=args.f2n_channels,
            device=device,
            seed=args.seed,
        ).clamp(0.0, 1.0)
        # denoise_image_f2n may crop odd dims to even; align the reference.
        full = full[..., : pred.shape[-2], : pred.shape[-1]]
        for k, fn in _METRIC_FNS.items():
            per_sample[k].append(fn(pred, full))
    return _summarize(per_sample)


@torch.no_grad()
def log_sample_images(model, loader, device, full_slice, patch_size, wb, n=4, epoch=None):
    """Log a [low-dose | predicted | full-dose | |diff|] panel grid to W&B."""
    if not _MPL_AVAILABLE:
        return
    import io
    import numpy as np
    model.eval()
    panels = []
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        if full_slice:
            pred = overlapped_inference(
                model, low, patch_size=patch_size, margin=patch_size // 4
            ).clamp(0.0, 1.0)
        else:
            pred = model(low).clamp(0.0, 1.0)
        for i in range(min(low.size(0), n - len(panels))):
            l_img = low[i, 0].cpu().numpy()
            p_img = pred[i, 0].cpu().numpy()
            f_img = full[i, 0].cpu().numpy()
            diff = np.abs(p_img - f_img)

            fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), dpi=100)
            titles = ["Low-dose input", "Denoised (pred)", "Full-dose ref", "|Pred − Ref|"]
            imgs = [l_img, p_img, f_img, diff]
            vmaxes = [1.0, 1.0, 1.0, max(diff.max(), 1e-6)]
            cmaps = ["gray", "gray", "gray", "hot"]
            for ax, img, title, vmax, cmap in zip(axes, imgs, titles, vmaxes, cmaps):
                im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            epoch_str = f" — epoch {epoch}" if epoch is not None else ""
            fig.suptitle(f"Sample {len(panels) + 1}{epoch_str}", fontsize=10, y=1.02)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            from PIL import Image as _PILImage  # noqa: PLC0415 (PIL is a wandb dep)
            panels.append(_wandb.Image(_PILImage.open(buf), caption=f"sample_{len(panels)+1}"))
        if len(panels) >= n:
            break
    if panels:
        wb.log({"val/images": panels})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a CT denoiser.")
    parser.add_argument("--model", choices=MODELS, default="ctformer")
    parser.add_argument(
        "--training-mode",
        choices=["supervised", "n2v", "n2sim", "zsn2n", "f2n", "ssflow"],
        default="supervised",
        help="supervised: clean-target MSE/flow loss (default). "
             "n2v: Noise2Void blind-spot self-supervision (clean target ignored). "
             "n2sim: Noise2Sim similarity-based self-supervision (clean target ignored). "
             "zsn2n: per-image zero-shot test-time training (no shared model/checkpoint). "
             "f2n: per-image Filter2Noise attention-guided bilateral filtering (no checkpoint). "
             "ssflow: self-supervised rectified flow on manufactured noisy pairs "
             "(requires --model ssflow; clean target ignored).",
    )
    parser.add_argument("--n2v-mask-fraction", type=float, default=0.02,
                        help="fraction of pixels masked as blind spots (n2v)")
    parser.add_argument("--n2v-neighbor-radius", type=int, default=2,
                        help="neighbour window radius for blind-spot replacement (n2v)")
    parser.add_argument("--n2sim-search-radius", type=int, default=4,
                        help="window radius searched for a similar pixel (n2sim)")
    parser.add_argument("--n2sim-patch-radius", type=int, default=1,
                        help="patch radius used to score pixel similarity (n2sim)")
    parser.add_argument("--n2sim-num-similar", type=int, default=1,
                        help="number of best matches averaged into the target (n2sim)")
    parser.add_argument("--ssflow-pairing", choices=["similarity", "downsample"],
                        default="similarity",
                        help="noisy-pair construction for ssflow: similarity "
                             "(Noise2Sim-style non-local, correlated-noise-aware, v2) "
                             "or downsample (Neighbor2Neighbor/ZS-N2N half-res, v1)")
    parser.add_argument("--ssflow-search-radius", type=int, default=4,
                        help="window radius searched for a similar patch (ssflow similarity)")
    parser.add_argument("--ssflow-patch-radius", type=int, default=1,
                        help="patch radius used to score similarity (ssflow)")
    parser.add_argument("--ssflow-num-similar", type=int, default=1,
                        help="number of best matches averaged into the paired view (ssflow)")
    parser.add_argument("--ssflow-exclude-radius", type=int, default=2,
                        help="exclude candidate offsets within this radius so the paired "
                             "noise is decorrelated (ssflow; the correlated-noise knob)")
    parser.add_argument("--zsn2n-iters", type=int, default=2000,
                        help="per-image optimisation steps (zsn2n)")
    parser.add_argument("--zsn2n-channels", type=int, default=48,
                        help="hidden width of the per-image network (zsn2n)")
    parser.add_argument("--zsn2n-lr", type=float, default=1e-3,
                        help="learning rate for per-image optimisation (zsn2n)")
    parser.add_argument("--f2n-iters", type=int, default=1500,
                        help="per-image optimisation steps (f2n)")
    parser.add_argument("--f2n-layers", type=int, default=2,
                        help="number of stacked attention bilateral filters (f2n)")
    parser.add_argument("--f2n-radius", type=int, default=3,
                        help="bilateral filter window radius (f2n)")
    parser.add_argument("--f2n-channels", type=int, default=16,
                        help="hidden width of the parameter-prediction net (f2n)")
    parser.add_argument("--f2n-lr", type=float, default=1e-3,
                        help="learning rate for per-image optimisation (f2n)")
    parser.add_argument("--dicom-root", type=str, default=None,
                        help="dir of DICOM series subdirs (SeriesInstanceUID)")
    parser.add_argument("--h5-path", type=str, default=None,
                        help="preprocessed HDF5 file (see scripts/convert_dicom_to_h5.py)")
    parser.add_argument("--anatomy", choices=sorted(ANATOMY_WINDOWS),
                        default="abdomen",
                        help="HU window preset for --dicom-root normalisation: "
                             "abdomen (soft tissue), chest (lung), or head (brain). "
                             "Ignored for --h5-path (window baked in at conversion).")
    parser.add_argument("--hu-offset", type=float, default=None,
                        help="override the --anatomy window offset (HU). "
                             "Window is [-offset, scale-offset].")
    parser.add_argument("--hu-scale", type=float, default=None,
                        help="override the --anatomy window scale (HU width).")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--flow-steps", type=int, default=20,
                        help="Euler ODE steps at inference (flowmatching only)")
    parser.add_argument("--flow-steps-eval", type=int, default=5,
                        help="Euler ODE steps during training-time validation (faster; "
                             "use --flow-steps for final quality eval)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--synthetic-len", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name; enables per-epoch metric logging")
    parser.add_argument("--log-images", type=int, default=0, metavar="N",
                        help="number of val samples to log as images each epoch (0=off)")
    parser.add_argument("--log-image-freq", type=int, default=1, metavar="FREQ",
                        help="log images every FREQ epochs (default: every epoch)")
    args = parser.parse_args(argv)

    if args.training_mode in ("n2v", "n2sim") and args.model == "flowmatching":
        parser.error(
            "flowmatching needs paired clean targets; it is incompatible with "
            f"--training-mode {args.training_mode}. Use --training-mode supervised."
        )

    if (args.model == "ssflow") != (args.training_mode == "ssflow"):
        parser.error(
            "--model ssflow and --training-mode ssflow must be used together: "
            "the self-supervised flow has its own velocity network and loss."
        )

    # Resolve the HU window: --anatomy preset, with explicit --hu-offset/--hu-scale
    # taking precedence. Stored back onto args so build_loaders / provenance see
    # the concrete numbers (and they land in the W&B config).
    _offset, _scale = window_for_anatomy(args.anatomy)
    args.hu_offset = _offset if args.hu_offset is None else args.hu_offset
    args.hu_scale = _scale if args.hu_scale is None else args.hu_scale

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # zsn2n / f2n train a fresh per-image net and ignore --model.
    model_str = (
        "per-image-net" if args.training_mode in ("zsn2n", "f2n") else args.model
    )
    print(f"device={device}  model={model_str}  mode={args.training_mode}")
    train_loader, val_loader, full_slice = build_loaders(args)

    prov = provenance(args, device)
    ds_prov = dataset_provenance(train_loader, val_loader, full_slice)
    print(
        f"window: anatomy={prov['anatomy']} "
        f"hu=[{-args.hu_offset:.0f}, {args.hu_scale - args.hu_offset:.0f}]  "
        f"data: {ds_prov.get('n_train_patients', '?')} train / "
        f"{ds_prov.get('n_val_patients', '?')} val patients, "
        f"{ds_prov['n_train_slices']}/{ds_prov['n_val_slices']} slices  "
        f"git={prov['git_sha']}"
    )

    _wb = None
    if args.wandb_project:
        if _WANDB_AVAILABLE:
            _wb = _wandb.init(
                project=args.wandb_project,
                config=vars(args),
                resume="allow",
            )
            _wb.config.update({**prov, **ds_prov}, allow_val_change=True)
            # Summarise each val metric by its *best* epoch rather than its last.
            # Val PSNR routinely peaks mid-training and then regresses (markedly
            # for the flow models), so W&B's default last-value summary understates
            # a run. Mirrors the best-by-PSNR checkpoint tracking in the loop below.
            for _m, _how in (("psnr", "max"), ("ssim", "max"), ("rmse", "min"),
                             ("gmsd", "min"), ("nps_ratio", "min")):
                _wb.define_metric(f"val/{_m}", summary=_how)
        else:
            print("wandb not installed; skipping W&B logging.")

    # Identity ("do nothing") baseline: score the noisy input against the clean
    # reference. Logged once for every run so trained models and the per-image
    # zsn2n results can be read against the floor they must beat.
    baseline = identity_baseline(val_loader, device)
    print(
        f"baseline (identity)  psnr={baseline['psnr']:.3f}  "
        f"ssim={baseline['ssim']:.4f}  rmse={baseline['rmse']:.5f}  "
        f"gmsd={baseline['gmsd']:.5f}  nps_ratio={baseline['nps_ratio']:.5f}"
    )
    if _wb:
        _wb.log({f"baseline/{k}": v for k, v in baseline.items()})

    # Zero-Shot Noise2Noise and Filter2Noise train a fresh per-image network at
    # eval time; they have no shared model, training loop, or checkpoint, so they
    # short-circuit here.
    if args.training_mode in ("zsn2n", "f2n"):
        if args.training_mode == "zsn2n":
            metrics = run_zsn2n_eval(val_loader, device, args)
        else:
            metrics = run_f2n_eval(val_loader, device, args)
        print(
            f"eval ({args.training_mode})  psnr={metrics['psnr']:.3f}  "
            f"ssim={metrics['ssim']:.4f}  rmse={metrics['rmse']:.5f}  "
            f"gmsd={metrics['gmsd']:.5f}  nps_ratio={metrics['nps_ratio']:.5f}"
        )
        if _wb:
            _wb.log({f"val/{k}": v for k, v in metrics.items()})
            _wb.finish()
        return

    if args.model == "flowmatching":
        model = FlowMatching(num_steps=args.flow_steps).to(device)
    elif args.model == "ssflow":
        model = SelfSupervisedFlow(
            num_steps=args.flow_steps,
            pairing=args.ssflow_pairing,
            search_radius=args.ssflow_search_radius,
            patch_radius=args.ssflow_patch_radius,
            num_similar=args.ssflow_num_similar,
            exclude_radius=args.ssflow_exclude_radius,
        ).to(device)
    else:
        model = MODELS[args.model]().to(device)

    stats = model_stats(model)
    print(
        f"model={args.model}  params={stats['param_count']:,}  "
        f"size={stats['model_size_mb']:.2f} MB"
    )
    if _wb:
        _wb.config.update(stats, allow_val_change=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    n_train = len(train_loader.dataset)
    best_psnr = -float("inf")
    best_metrics = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for low, full in train_loader:
            low = low.to(device)
            optimizer.zero_grad()
            if args.training_mode == "n2v":
                # Self-supervised: clean target ignored, train on noisy input only.
                loss = n2v_training_step(
                    model, low,
                    mask_fraction=args.n2v_mask_fraction,
                    neighbor_radius=args.n2v_neighbor_radius,
                )
            elif args.training_mode == "n2sim":
                # Self-supervised: similarity target from the noisy image itself.
                loss = n2sim_training_step(
                    model, low,
                    search_radius=args.n2sim_search_radius,
                    patch_radius=args.n2sim_patch_radius,
                    num_similar=args.n2sim_num_similar,
                )
            elif args.training_mode == "ssflow":
                # Self-supervised rectified flow on manufactured noisy pairs.
                loss = model.ss_flow_loss(low)
            elif hasattr(model, "flow_loss"):
                loss = model.flow_loss(low, full.to(device))
            else:
                loss = criterion(model(low), full.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item() * low.size(0)
        train_loss = running / n_train
        print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.6f}")
        if _wb:
            epoch_metrics = evaluate(model, val_loader, device, full_slice,
                                     args.patch_size, eval_steps=args.flow_steps_eval)
            _wb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                **{f"val/{k}": v for k, v in epoch_metrics.items()},
            })
            # Keep the best-by-PSNR weights: val PSNR often peaks before the final
            # epoch, so the last-epoch model is not the one to checkpoint or report.
            if epoch_metrics["psnr"] > best_psnr:
                best_psnr = epoch_metrics["psnr"]
                best_metrics = epoch_metrics
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if args.log_images > 0 and epoch % args.log_image_freq == 0:
                log_sample_images(model, val_loader, device, full_slice,
                                  args.patch_size, _wb, n=args.log_images, epoch=epoch)

    if best_metrics is not None:
        # Best epoch observed during training (per-epoch eval needs W&B logging).
        metrics, save_state = best_metrics, best_state
    else:
        # W&B disabled -> no per-epoch eval; score the final model once.
        metrics = evaluate(model, val_loader, device, full_slice,
                           args.patch_size, eval_steps=args.flow_steps_eval)
        save_state = model.state_dict()
    print(
        f"eval  psnr={metrics['psnr']:.3f}  ssim={metrics['ssim']:.4f}  "
        f"rmse={metrics['rmse']:.5f}  gmsd={metrics['gmsd']:.5f}  "
        f"nps_ratio={metrics['nps_ratio']:.5f}  "
        f"params={stats['param_count']:,}  latency={metrics['latency_ms']:.1f} ms"
    )

    if _wb:
        # Headline summary: how far the best epoch beats the do-nothing floor.
        _wb.summary["val/psnr_gain"] = metrics["psnr"] - baseline["psnr"]
        _wb.summary["val/ssim_gain"] = metrics["ssim"] - baseline["ssim"]
        _wb.finish()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{args.model}.pt"
    torch.save(save_state, ckpt)
    print(f"saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
