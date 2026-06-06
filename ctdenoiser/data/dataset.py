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


class DICOMCTDataset(Dataset):
    """Reads paired DICOM series directly from a root directory.

    Expects ``root`` to contain subdirectories named by SeriesInstanceUID,
    each holding ``.dcm`` files. Low/full dose is detected from the DICOM
    ``SeriesDescription`` header and paired by ``PatientID``.

    All volumes are loaded into memory at init time and normalised to [0, 1].
    """

    def __init__(
        self,
        root,
        patients,
        patch_size=64,
        train=True,
        hu_offset=1000.0,
        hu_scale=2000.0,
    ):
        from .dicom import read_series_hu

        self.patch_size = patch_size
        self.train = train

        mapping = self._scan_series(root)
        self.low_volumes: dict[str, np.ndarray] = {}
        self.full_volumes: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        for pid in patients:
            if pid not in mapping:
                raise KeyError(f"Patient {pid} not found in {root}")
            low_dir, full_dir = mapping[pid]["low"], mapping[pid]["full"]
            low_vol = read_series_hu(str(low_dir)).astype(np.float32)
            full_vol = read_series_hu(str(full_dir)).astype(np.float32)

            low_vol = np.clip((low_vol + hu_offset) / hu_scale, 0.0, 1.0)
            full_vol = np.clip((full_vol + hu_offset) / hu_scale, 0.0, 1.0)

            n_slices = min(low_vol.shape[0], full_vol.shape[0])
            if low_vol.shape[0] != full_vol.shape[0]:
                print(
                    f"Warning: {pid} slice count mismatch "
                    f"(low={low_vol.shape[0]}, full={full_vol.shape[0]}), "
                    f"using {n_slices}"
                )
            self.low_volumes[pid] = low_vol[:n_slices]
            self.full_volumes[pid] = full_vol[:n_slices]
            for i in range(n_slices):
                self.samples.append((pid, i))

        if not self.samples:
            raise ValueError(f"No slices for patients {patients} in {root}")

    @staticmethod
    def _scan_series(root):
        """Scan series dirs and return {patient_id: {low: Path, full: Path}}."""
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("pydicom is required: pip install pydicom") from exc

        root = Path(root)
        mapping: dict[str, dict[str, Path]] = {}
        for sdir in sorted(root.iterdir()):
            if not sdir.is_dir():
                continue
            dcm_files = list(sdir.glob("*.dcm"))
            if not dcm_files:
                continue
            hdr = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            pid = str(hdr.PatientID)
            desc = str(getattr(hdr, "SeriesDescription", "")).lower()
            if "low dose" in desc:
                dose = "low"
            elif "full dose" in desc:
                dose = "full"
            else:
                continue
            mapping.setdefault(pid, {})
            mapping[pid][dose] = sdir

        complete = {
            pid: v
            for pid, v in mapping.items()
            if "low" in v and "full" in v
        }
        if not complete:
            raise ValueError(
                f"No paired low/full dose patients found in {root}. "
                f"Detected: {mapping}"
            )
        return complete

    @staticmethod
    def list_patients(root):
        mapping = DICOMCTDataset._scan_series(root)
        return sorted(mapping.keys())

    @classmethod
    def split_patients(cls, root, val_fraction=0.2, seed=0):
        patients = cls.list_patients(root)
        rng = random.Random(seed)
        rng.shuffle(patients)
        n_val = max(1, int(round(len(patients) * val_fraction)))
        return patients[n_val:], patients[:n_val]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, i = self.samples[idx]
        low = torch.from_numpy(self.low_volumes[pid][i]).unsqueeze(0)
        full = torch.from_numpy(self.full_volumes[pid][i]).unsqueeze(0)
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
