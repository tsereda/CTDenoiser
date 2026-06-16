"""Paired low-dose / full-dose CT datasets."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Default HU window: soft-tissue / abdomen (level 40 HU, width 400 HU ->
# range [-160, +240]). This is the standard AAPM/Mayo LDCT window. A wide
# window (e.g. [-1000, +1000]) compresses the low/full dose difference down to
# ~1% of the [0, 1] range, making the "denoising" task near-trivial and the
# identity baseline beat every trained model -- see pair_noise_stats below.
HU_OFFSET = 160.0
HU_SCALE = 400.0


def normalize_hu(vol, hu_offset=HU_OFFSET, hu_scale=HU_SCALE):
    """Window a HU volume into [0, 1]: clip((vol + offset) / scale)."""
    return np.clip((vol + hu_offset) / hu_scale, 0.0, 1.0)


def pair_noise_stats(pid, low_vol, full_vol):
    """Report the low/full difference and warn on a likely pairing bug.

    Returns the mean absolute difference (in normalised [0, 1] units), which is
    the noise the denoiser is asked to remove and the floor the identity
    baseline scores against. A value near 0 means ``low`` and ``full`` are
    effectively the same image -- either a duplicated/mis-paired series or a
    window so wide the dose difference washed out.
    """
    diff = float(np.mean(np.abs(low_vol.astype(np.float64) - full_vol)))
    if diff < 1e-4:
        print(
            f"Warning: {pid} low and full are nearly identical "
            f"(mean|low-full|={diff:.2e}); the denoising task is trivial. "
            f"Check the low/full pairing and that the HU window is not too wide."
        )
    return diff


class _PairedCTBase(Dataset):
    """Shared logic for paired low/full-dose CT datasets."""

    low_volumes: dict[str, np.ndarray]
    full_volumes: dict[str, np.ndarray]
    samples: list[tuple[str, int]]
    patch_size: int
    train: bool

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

    @staticmethod
    def _split(patients, val_fraction=0.2, seed=0):
        rng = random.Random(seed)
        rng.shuffle(patients)
        n_val = max(1, int(round(len(patients) * val_fraction)))
        return patients[n_val:], patients[:n_val]


class DICOMCTDataset(_PairedCTBase):
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
        hu_offset=HU_OFFSET,
        hu_scale=HU_SCALE,
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

            low_vol = normalize_hu(low_vol, hu_offset, hu_scale)
            full_vol = normalize_hu(full_vol, hu_offset, hu_scale)
            pair_noise_stats(pid, low_vol, full_vol)

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
        return cls._split(patients, val_fraction, seed)


class HDF5CTDataset(_PairedCTBase):
    """Reads paired low/full dose CT from a preprocessed HDF5 file.

    The HDF5 file should contain ``/patients/{id}/low`` and
    ``/patients/{id}/full`` datasets (float32, already normalised to [0, 1]).
    Use ``scripts/convert_dicom_to_h5.py`` to create one from DICOM data.
    """

    def __init__(self, h5_path, patients, patch_size=64, train=True):
        import h5py

        self.patch_size = patch_size
        self.train = train
        self.low_volumes: dict[str, np.ndarray] = {}
        self.full_volumes: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        with h5py.File(h5_path, "r") as f:
            for pid in patients:
                grp = f[f"patients/{pid}"]
                low_vol = grp["low"][:]
                full_vol = grp["full"][:]

                pair_noise_stats(pid, low_vol, full_vol)

                n_slices = low_vol.shape[0]
                self.low_volumes[pid] = low_vol
                self.full_volumes[pid] = full_vol
                for i in range(n_slices):
                    self.samples.append((pid, i))

        if not self.samples:
            raise ValueError(f"No slices for patients {patients} in {h5_path}")

    @staticmethod
    def list_patients(h5_path):
        import h5py

        with h5py.File(h5_path, "r") as f:
            return sorted(f["patients"].keys())

    @classmethod
    def split_patients(cls, h5_path, val_fraction=0.2, seed=0):
        patients = cls.list_patients(h5_path)
        return cls._split(patients, val_fraction, seed)


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
