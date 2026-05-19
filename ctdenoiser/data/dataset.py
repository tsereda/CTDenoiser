"""Paired low-dose / full-dose CT datasets."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _random_patch(low, full, patch_size):
    h, w = low.shape
    if patch_size is None or (h <= patch_size or w <= patch_size):
        return low, full
    top = np.random.randint(0, h - patch_size + 1)
    left = np.random.randint(0, w - patch_size + 1)
    sl = (slice(top, top + patch_size), slice(left, left + patch_size))
    return low[sl], full[sl]


class PairedCTDataset(Dataset):
    """Loads matching ``.npy`` slices from ``<root>/low_dose`` and
    ``<root>/full_dose``. Filenames must match between the two directories."""

    def __init__(self, root, patch_size=64):
        root = Path(root)
        self.low_dir = root / "low_dose"
        self.full_dir = root / "full_dose"
        self.patch_size = patch_size
        self.files = sorted(p.name for p in self.low_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy slices found under {self.low_dir}")
        missing = [f for f in self.files if not (self.full_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} low-dose slices have no full-dose match "
                f"(e.g. {missing[0]})"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        low = np.load(self.low_dir / name).astype(np.float32)
        full = np.load(self.full_dir / name).astype(np.float32)
        low, full = _random_patch(low, full, self.patch_size)
        return (
            torch.from_numpy(low).unsqueeze(0),
            torch.from_numpy(full).unsqueeze(0),
        )


class SyntheticCTDataset(Dataset):
    """Synthetic clean/noisy pairs for smoke-testing the pipeline."""

    def __init__(self, length=64, patch_size=64, noise_std=0.1, seed=0):
        self.length = length
        self.patch_size = patch_size
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        clean = torch.rand(1, self.patch_size, self.patch_size, generator=g)
        noise = torch.randn(1, self.patch_size, self.patch_size, generator=g)
        low = (clean + self.noise_std * noise).clamp(0.0, 1.0)
        return low, clean
