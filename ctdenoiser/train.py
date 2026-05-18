"""Training / benchmarking entry point.

Examples
--------
    python -m ctdenoiser.train --model ctformer --epochs 1
    python -m ctdenoiser.train --model redcnn --data-root data/ --epochs 50
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import PairedCTDataset, SyntheticCTDataset
from .metrics import psnr, rmse, ssim
from .models import CTformer, REDCNN

MODELS = {"ctformer": CTformer, "redcnn": REDCNN}


def build_dataset(args):
    if args.data_root:
        return PairedCTDataset(args.data_root, patch_size=args.patch_size)
    print("No --data-root given; using SyntheticCTDataset for a smoke run.")
    return SyntheticCTDataset(length=args.synthetic_len, patch_size=args.patch_size)


def evaluate(model, loader, device):
    model.eval()
    n, p, s, r = 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for low, full in loader:
            low, full = low.to(device), full.to(device)
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
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--synthetic-len", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = MODELS[args.model]().to(device)
    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for low, full in loader:
            low, full = low.to(device), full.to(device)
            optimizer.zero_grad()
            loss = criterion(model(low), full)
            loss.backward()
            optimizer.step()
            running += loss.item() * low.size(0)
        print(f"epoch {epoch}/{args.epochs}  loss={running / len(dataset):.6f}")

    metrics = evaluate(model, loader, device)
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
