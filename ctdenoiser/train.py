"""Training / benchmarking entry point.

Examples
--------
    # synthetic smoke run
    python -m ctdenoiser.train --model ctformer --epochs 1

    # paired .npy directories
    python -m ctdenoiser.train --model redcnn --data-root data/ --epochs 50

    # TCIA HDF5 cache (patient-split, full-slice overlapped-inference eval)
    python -m ctdenoiser.train --model ctformer \
        --h5-cache /content/ldct_cache.h5 --epochs 50 --batch-size 16
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import HDF5CTDataset, PairedCTDataset, SyntheticCTDataset
from .inference import overlapped_inference
from .metrics import psnr, rmse, ssim
from .models import CTformer, REDCNN

MODELS = {"ctformer": CTformer, "redcnn": REDCNN}


_DRIVE_CACHE_FALLBACKS = [
    "/content/drive/MyDrive/ldct_cache.h5",
    "/content/drive/MyDrive/CTDenoiser/ldct_cache.h5",
]


def _resolve_h5(path: str) -> str:
    """Return path if it exists, else try common Colab Drive locations."""
    if Path(path).exists():
        return path
    for fb in _DRIVE_CACHE_FALLBACKS:
        if Path(fb).exists():
            print(f"Cache not found at {path!r} — using {fb!r} instead.")
            return fb
    raise FileNotFoundError(
        f"HDF5 cache not found at {path!r}.\n"
        "Options:\n"
        "  1. Copy from Drive first (faster I/O):\n"
        "       import shutil; shutil.copy('/content/drive/MyDrive/ldct_cache.h5', '/content/ldct_cache.h5')\n"
        "  2. Pass the Drive path directly:\n"
        "       --h5-cache /content/drive/MyDrive/ldct_cache.h5"
    )


def build_loaders(args):
    """Return (train_loader, val_loader, full_slice_eval)."""
    if args.h5_cache:
        args.h5_cache = _resolve_h5(args.h5_cache)
        train_p, val_p = HDF5CTDataset.split_patients(
            args.h5_cache, val_fraction=args.val_fraction, seed=args.seed
        )
        print(
            f"HDF5 cache: {len(train_p)} train / {len(val_p)} val patients "
            f"({train_p[:3]}... | {val_p})"
        )
        train_ds = HDF5CTDataset(
            args.h5_cache, train_p, patch_size=args.patch_size, train=True
        )
        val_ds = HDF5CTDataset(
            args.h5_cache, val_p, patch_size=args.patch_size, train=False
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        # Full slices vary in size -> batch_size must be 1.
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        return train_loader, val_loader, True

    if args.data_root:
        ds = PairedCTDataset(args.data_root, patch_size=args.patch_size)
    else:
        print("No --data-root / --h5-cache; using SyntheticCTDataset.")
        ds = SyntheticCTDataset(
            length=args.synthetic_len, patch_size=args.patch_size
        )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    return loader, loader, False


@torch.no_grad()
def evaluate(model, loader, device, full_slice, patch_size):
    model.eval()
    n, p, s, r = 0, 0.0, 0.0, 0.0
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        if full_slice:
            pred = overlapped_inference(
                model, low, patch_size=patch_size, margin=patch_size // 4
            ).clamp(0.0, 1.0)
        else:
            pred = model(low).clamp(0.0, 1.0)
        bs = low.size(0)
        p += psnr(pred, full) * bs
        s += ssim(pred, full) * bs
        r += rmse(pred, full) * bs
        n += bs
    return {"psnr": p / n, "ssim": s / n, "rmse": r / n}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a CT denoiser.")
    parser.add_argument("--model", choices=MODELS, default="ctformer")
    parser.add_argument("--data-root", type=str, default=None,
                        help="dir with low_dose/ and full_dose/ .npy slices")
    parser.add_argument("--h5-cache", type=str, default=None,
                        help="TCIA ldct_cache.h5 (<pid>_low / <pid>_full)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--synthetic-len", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device}  model={args.model}")
    model = MODELS[args.model]().to(device)
    train_loader, val_loader, full_slice = build_loaders(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    n_train = len(train_loader.dataset)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for low, full in train_loader:
            low, full = low.to(device), full.to(device)
            optimizer.zero_grad()
            loss = criterion(model(low), full)
            loss.backward()
            optimizer.step()
            running += loss.item() * low.size(0)
        print(f"epoch {epoch}/{args.epochs}  loss={running / n_train:.6f}")

    metrics = evaluate(model, val_loader, device, full_slice, args.patch_size)
    print(
        f"eval  psnr={metrics['psnr']:.3f}  "
        f"ssim={metrics['ssim']:.4f}  rmse={metrics['rmse']:.5f}"
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{args.model}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
