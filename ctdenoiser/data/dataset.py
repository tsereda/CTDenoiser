"""Paired low-dose / full-dose CT datasets."""

import random
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


class HDF5CTDataset(Dataset):
    """Reads the ``ldct_cache.h5`` produced by the TCIA preprocessing step.

    The cache stores one dataset per patient/dose, named ``<pid>_low`` and
    ``<pid>_full``, each shaped ``(num_slices, H, W)`` in raw Hounsfield
    units. HU is mapped to ``[0, 1]`` via ``clamp((hu + offset) / scale)``.

    Split by *patient* (not slice) to avoid train/val leakage; use
    :meth:`split_patients` to get disjoint patient lists.

    With ``train=True`` a random ``patch_size`` crop is returned; otherwise
    the full slice is returned (use ``batch_size=1`` for the val loader,
    slices vary in size and full-slice eval uses overlapped inference).
    """

    def __init__(
        self,
        cache_path,
        patients,
        patch_size=64,
        train=True,
        hu_offset=1000.0,
        hu_scale=2000.0,
    ):
        import h5py

        self.cache_path = str(cache_path)
        self.patch_size = patch_size
        self.train = train
        self.hu_offset = hu_offset
        self.hu_scale = hu_scale
        self._h5 = None

        # Build the (patient, slice) index now, then close — keeping a file
        # handle open across DataLoader worker forks corrupts HDF5 reads.
        with h5py.File(self.cache_path, "r") as f:
            self.samples = []
            for p in patients:
                key = f"{p}_low"
                if key not in f:
                    raise KeyError(f"{key} not found in {self.cache_path}")
                for i in range(f[key].shape[0]):
                    self.samples.append((p, i))
        if not self.samples:
            raise ValueError(
                f"No slices for patients {patients} in {self.cache_path}"
            )

    @staticmethod
    def list_patients(cache_path):
        import h5py

        with h5py.File(str(cache_path), "r") as f:
            return sorted({k.split("_")[0] for k in f.keys()})

    @classmethod
    def split_patients(cls, cache_path, val_fraction=0.2, seed=0):
        patients = cls.list_patients(cache_path)
        rng = random.Random(seed)
        rng.shuffle(patients)
        n_val = max(1, int(round(len(patients) * val_fraction)))
        return patients[n_val:], patients[:n_val]

    @property
    def h5(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, i = self.samples[idx]
        low = np.asarray(self.h5[f"{p}_low"][i], dtype=np.float32)
        full = np.asarray(self.h5[f"{p}_full"][i], dtype=np.float32)

        low = torch.from_numpy(low).unsqueeze(0)
        full = torch.from_numpy(full).unsqueeze(0)
        low = torch.clamp((low + self.hu_offset) / self.hu_scale, 0.0, 1.0)
        full = torch.clamp((full + self.hu_offset) / self.hu_scale, 0.0, 1.0)

        if self.train and self.patch_size:
            _, h, w = low.shape
            if h > self.patch_size and w > self.patch_size:
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                low = low[:, y : y + self.patch_size, x : x + self.patch_size]
                full = full[:, y : y + self.patch_size, x : x + self.patch_size]
        return low, full


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
