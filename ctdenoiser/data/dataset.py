"""Paired low-dose / full-dose CT datasets."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .noise import add_synthetic_noise, simulate_low_dose

# Default HU window: soft-tissue / abdomen (level 40 HU, width 400 HU ->
# range [-160, +240]). This is the standard AAPM/Mayo LDCT window. A wide
# window (e.g. [-1000, +1000]) compresses the low/full dose difference down to
# ~1% of the [0, 1] range, making the "denoising" task near-trivial and the
# identity baseline beat every trained model -- see pair_noise_stats below.
HU_OFFSET = 160.0
HU_SCALE = 400.0

# Per-anatomy HU windows as (offset, scale), where the window is
# [-offset, scale-offset] and offset/scale derive from clinical level/width:
#   offset = width/2 - level,  scale = width.
# Lung and brain content sit far outside the abdomen window, so denoising chest
# or head scans under the default soft-tissue window clips the relevant anatomy
# to 0/1 and washes the low/full difference out (the pair_noise_stats warning).
ANATOMY_WINDOWS = {
    "abdomen": (HU_OFFSET, HU_SCALE),  # soft tissue L40/W400  -> [-160, +240]
    "chest": (1350.0, 1500.0),         # lung      L-600/W1500 -> [-1350, +150]
    "head": (0.0, 80.0),               # brain     L40/W80     -> [0, +80]
}


def open_h5(h5_path, mode="r"):
    """Open an HDF5 cache, turning a corrupt/missing file into a clear error.

    h5py's native message for a truncated or partially-written cache is the
    cryptic ``OSError: Unable to synchronously open file (bad object header
    version number)``. Since the sweep agents copy the cache off a PVC, the
    usual culprit is a source file left half-written by an interrupted
    ``convert_dicom_to_h5.py`` run (or a truncated copy). Surface that.
    """
    import os

    import h5py

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"HDF5 cache {h5_path!r} does not exist. Generate it with "
            f"scripts/convert_dicom_to_h5.py and copy it into place."
        )
    size = os.path.getsize(h5_path)
    if size == 0:
        raise OSError(
            f"HDF5 cache {h5_path!r} is empty (0 bytes) -- the conversion or "
            f"the copy that produced it did not finish. Regenerate it."
        )
    try:
        return h5py.File(h5_path, mode)
    except OSError as e:
        raise OSError(
            f"Could not open HDF5 cache {h5_path!r} ({size / 1e6:.1f} MB): {e}. "
            f"The file is likely truncated or partially written -- an "
            f"interrupted scripts/convert_dicom_to_h5.py run or an incomplete "
            f"copy leaves a corrupt cache. Regenerate it and copy it again."
        ) from e


def window_for_anatomy(anatomy):
    """Return the ``(hu_offset, hu_scale)`` window preset for an anatomy name."""
    try:
        return ANATOMY_WINDOWS[anatomy]
    except KeyError:
        raise ValueError(
            f"unknown anatomy {anatomy!r}; choose from "
            f"{sorted(ANATOMY_WINDOWS)} or pass --hu-offset/--hu-scale."
        ) from None


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
        low_np = self.low_volumes[pid][i]
        full_np = self.full_volumes[pid][i]
        if self.train and self.patch_size:
            h, w = low_np.shape
            if h > self.patch_size and w > self.patch_size:
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                low_np = low_np[y : y + self.patch_size, x : x + self.patch_size]
                full_np = full_np[y : y + self.patch_size, x : x + self.patch_size]
        # Copy out of the backing store: the volumes may be read-only memmaps
        # (mmap_cache mode), which torch.from_numpy cannot safely wrap. The
        # copy is the crop only (64x64 in training), and a writable tensor is
        # required downstream by pin_memory / in-place ops anyway.
        low = torch.from_numpy(np.array(low_np)).unsqueeze(0)
        full = torch.from_numpy(np.array(full_np)).unsqueeze(0)
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
        from .dicom import scan_paired_series

        return scan_paired_series(root)

    @staticmethod
    def list_patients(root):
        mapping = DICOMCTDataset._scan_series(root)
        return sorted(mapping.keys())

    @classmethod
    def split_patients(cls, root, val_fraction=0.2, seed=0):
        patients = cls.list_patients(root)
        return cls._split(patients, val_fraction, seed)


def unpack_h5_to_npy(h5_path, cache_dir):
    """Unpack an HDF5 cache into per-patient ``.npy`` files for memory-mapping.

    Writes ``{cache_dir}/{pid}.{low,full}.npy`` for every patient in the cache.
    Idempotent and safe under concurrent callers: existing files are kept, and
    each file is written to a per-process temp name and moved into place with an
    atomic ``os.replace``, so a reader never sees a truncated array and racing
    unpackers merely do redundant work.

    Why: the gzip-chunked HDF5 cannot be memory-mapped, so every training
    process that eager-loads it holds a private full copy of the dataset in
    RAM. With several sweep agents packed onto one GPU (one pod), those copies
    multiply and push the pod past its memory request (the eviction/kill
    symptom). Plain ``.npy`` files loaded with ``mmap_mode="r"`` share a single
    page-cache copy across every agent and DataLoader worker in the pod.
    """
    import os

    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    with open_h5(h5_path, "r") as f:
        for pid in sorted(f["patients"].keys()):
            for arm in ("low", "full"):
                dst = os.path.join(cache_dir, f"{pid}.{arm}.npy")
                if os.path.exists(dst):
                    continue
                tmp = f"{dst}.tmp.{os.getpid()}"
                with open(tmp, "wb") as out:
                    np.save(out, f[f"patients/{pid}/{arm}"][:])
                os.replace(tmp, dst)
    return cache_dir


class HDF5CTDataset(_PairedCTBase):
    """Reads paired low/full dose CT from a preprocessed HDF5 file.

    The HDF5 file should contain ``/patients/{id}/low`` and
    ``/patients/{id}/full`` datasets (float32, already normalised to [0, 1]).
    Use ``scripts/convert_dicom_to_h5.py`` to create one from DICOM data.

    With ``mmap_cache`` set, the file is first unpacked (once, idempotently)
    into per-patient ``.npy`` files under that directory and the volumes are
    memory-mapped instead of copied into process RAM. All processes on the
    node then share one page-cache copy of the data -- essential when several
    sweep agents run in one pod, where eager per-process copies exceed the
    pod's memory request and get it evicted.
    """

    def __init__(self, h5_path, patients, patch_size=64, train=True,
                 mmap_cache=None):
        self.patch_size = patch_size
        self.train = train
        self.low_volumes: dict[str, np.ndarray] = {}
        self.full_volumes: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        if mmap_cache:
            import os

            cache_dir = unpack_h5_to_npy(h5_path, mmap_cache)
            for pid in patients:
                low_vol = np.load(
                    os.path.join(cache_dir, f"{pid}.low.npy"), mmap_mode="r"
                )
                full_vol = np.load(
                    os.path.join(cache_dir, f"{pid}.full.npy"), mmap_mode="r"
                )
                self.low_volumes[pid] = low_vol
                self.full_volumes[pid] = full_vol
                for i in range(low_vol.shape[0]):
                    self.samples.append((pid, i))
        else:
            with open_h5(h5_path, "r") as f:
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
        with open_h5(h5_path, "r") as f:
            return sorted(f["patients"].keys())

    @classmethod
    def split_patients(cls, h5_path, val_fraction=0.2, seed=0):
        patients = cls.list_patients(h5_path)
        return cls._split(patients, val_fraction, seed)


def _procedural_clean_image(h, w, seed):
    """A deterministic smooth-plus-structured clean image in [0, 1].

    Used as a stand-in for real natural images so the natural-image dataset (and
    its tests / smoke runs) need no download: a low-frequency base field with a
    few random bright/dark blobs and a linear gradient gives enough structure
    that denoising is non-trivial, while staying reproducible per ``seed``.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    # Sum of a few low-frequency sinusoids -> a smooth, structured base.
    img = np.zeros((h, w), dtype=np.float32)
    for _ in range(4):
        fx, fy = rng.uniform(0.5, 3.0, size=2)
        ph = rng.uniform(0, 2 * np.pi)
        img += np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
    img += 2.0 * (rng.uniform() * xx + rng.uniform() * yy)  # gradient
    # A handful of Gaussian blobs for local contrast (edges to preserve).
    for _ in range(rng.integers(3, 7)):
        cy, cx = rng.uniform(0, 1, size=2)
        r = rng.uniform(0.05, 0.2)
        amp = rng.uniform(-1.5, 1.5)
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    img -= img.min()
    img /= img.max() + 1e-8
    return img.astype(np.float32)


def _procedural_ct_volume(n_slices, h, w, seed):
    """A deterministic normalised full-dose CT-like volume in [0, 1].

    An elliptical soft-tissue "body" with a few embedded structures per slice --
    enough spatial structure and signal range for the low-dose simulator to
    produce a meaningful paired low/full example without any real DICOM data.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = (yy - h / 2) / (h / 2)
    xx = (xx - w / 2) / (w / 2)
    vol = np.zeros((n_slices, h, w), dtype=np.float32)
    for s in range(n_slices):
        body = ((xx / 0.85) ** 2 + (yy / 0.7) ** 2) <= 1.0
        img = np.where(body, 0.5, 0.0).astype(np.float32)
        for _ in range(rng.integers(3, 6)):
            cy, cx = rng.uniform(-0.5, 0.5, size=2)
            r = rng.uniform(0.08, 0.25)
            val = rng.uniform(0.2, 1.0)
            blob = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
            img = np.where(blob & body, val, img)
        vol[s] = img
    return vol


class NaturalImageDenoisingDataset(Dataset):
    """Clean natural images + synthetic noise, for general (non-CT) denoising.

    Generalises the benchmark beyond CT while keeping the exact ``(low, full)``
    tensor interface the CT datasets use, so the training / inference path is
    unchanged. ``full`` is a clean image and ``low`` is a synthetically noised
    view, under two regimes:

    * ``noise_mode='gaussian'``  -- i.i.d. white noise, and
    * ``noise_mode='correlated'`` -- spatially-blurred noise,

    with the target being either the clean image (``pair_mode='clean'``,
    supervised) or a second *independent* noisy view (``pair_mode='noisy'``,
    Noise2Noise-style decorrelated pairing).

    Clean images are read from ``root`` (any PIL-readable files, converted to
    grayscale [0, 1]); with no ``root`` a deterministic procedural set is
    generated so the dataset runs self-contained for tests and smoke runs.
    """

    def __init__(self, items, root=None, patch_size=64, train=True,
                 noise_std=0.1, noise_mode="gaussian", correlation_sigma=1.5,
                 pair_mode="clean", samples_per_image=1, seed=0,
                 image_size=128):
        if noise_mode not in ("gaussian", "correlated"):
            raise ValueError(
                f"unknown noise_mode {noise_mode!r}; choose 'gaussian' or "
                f"'correlated'."
            )
        if pair_mode not in ("clean", "noisy"):
            raise ValueError(
                f"unknown pair_mode {pair_mode!r}; choose 'clean' or 'noisy'."
            )
        self.patch_size = patch_size
        self.train = train
        self.noise_std = noise_std
        self.noise_mode = noise_mode
        self.correlation_sigma = correlation_sigma
        self.pair_mode = pair_mode
        self.seed = seed

        self.images: list[np.ndarray] = [
            self._load_item(it, root, image_size) for it in items
        ]
        if not self.images:
            raise ValueError("NaturalImageDenoisingDataset got no images.")
        reps = samples_per_image if train else 1
        self.samples = [i for i in range(len(self.images)) for _ in range(reps)]

    @staticmethod
    def _load_item(item, root, image_size):
        if isinstance(item, (int, np.integer)):
            return _procedural_clean_image(image_size, image_size, int(item))
        from PIL import Image  # noqa: PLC0415 (optional; only the root path needs it)

        import os

        path = item if root is None else os.path.join(root, item)
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx = self.samples[idx]
        clean_np = self.images[img_idx]
        h, w = clean_np.shape
        if self.train and self.patch_size and h > self.patch_size and w > self.patch_size:
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
            clean_np = clean_np[y : y + self.patch_size, x : x + self.patch_size]
        clean = torch.from_numpy(np.array(clean_np)).float()
        # Deterministic noise in eval (seeded per image) so metrics are stable;
        # fresh randomness in training.
        gen = None
        if not self.train:
            gen = torch.Generator().manual_seed(self.seed + img_idx)
        low = add_synthetic_noise(
            clean, self.noise_std, self.noise_mode, self.correlation_sigma, gen
        )
        if self.pair_mode == "noisy":
            gen2 = None
            if not self.train:
                gen2 = torch.Generator().manual_seed(self.seed + img_idx + 10_000_000)
            full = add_synthetic_noise(
                clean, self.noise_std, self.noise_mode, self.correlation_sigma, gen2
            )
        else:
            full = clean
        return low.unsqueeze(0), full.unsqueeze(0)

    @staticmethod
    def list_items(root=None, length=256):
        """Enumerate items: image filenames under ``root``, else seed integers."""
        if root:
            import os

            exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")
            names = sorted(
                f for f in os.listdir(root) if f.lower().endswith(exts)
            )
            if not names:
                raise ValueError(f"No image files found in {root}")
            return names
        return list(range(length))

    @classmethod
    def split_items(cls, root=None, length=256, val_fraction=0.2, seed=0):
        items = cls.list_items(root, length)
        rng = random.Random(seed)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction)))
        return items[n_val:], items[:n_val]


class SimulatedLowDoseCTDataset(_PairedCTBase):
    """A second, independent low-dose CT source via low-dose *simulation*.

    Where :class:`HDF5CTDataset` reads real paired scans, this manufactures the
    paired low-dose arm from full-dose CT with a physically-motivated noise
    model (see :func:`ctdenoiser.data.noise.simulate_low_dose`: correlated,
    signal-dependent). That gives a low/full source that is independent of any
    single real acquisition -- the "does the clinical result hold on other
    low-dose data?" robustness axis -- without needing external DICOM.

    Full-dose volumes come from an HDF5 cache (``/patients/{id}/full``, reusing
    :func:`scripts/convert_dicom_to_h5.py` output) or, with no source, a
    deterministic procedural phantom set so the dataset runs self-contained.
    """

    def __init__(self, patients, h5_path=None, patch_size=64, train=True,
                 base_std=0.03, signal_std=0.06, correlation_sigma=1.0,
                 pair_mode="clean", seed=0, n_slices=8, image_size=128):
        if pair_mode not in ("clean", "noisy"):
            raise ValueError(
                f"unknown pair_mode {pair_mode!r}; choose 'clean' or 'noisy'."
            )
        self.patch_size = patch_size
        self.train = train
        self.base_std = base_std
        self.signal_std = signal_std
        self.correlation_sigma = correlation_sigma
        self.pair_mode = pair_mode
        self.seed = seed
        # ``full_volumes`` holds the clean full-dose arm; the low arm is
        # simulated on the fly in __getitem__, so no ``low_volumes`` store.
        self.full_volumes: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        if h5_path:
            with open_h5(h5_path, "r") as f:
                for pid in patients:
                    full_vol = f[f"patients/{pid}/full"][:]
                    self.full_volumes[pid] = full_vol
                    for i in range(full_vol.shape[0]):
                        self.samples.append((pid, i))
        else:
            import zlib

            for pid in patients:
                # Stable per-process hash (unlike hash(), which is salted by
                # PYTHONHASHSEED) so procedural phantoms are reproducible.
                s = self.seed + (zlib.crc32(pid.encode()) & 0xFFFF)
                vol = _procedural_ct_volume(n_slices, image_size, image_size, s)
                self.full_volumes[pid] = vol
                for i in range(vol.shape[0]):
                    self.samples.append((pid, i))

        if not self.samples:
            raise ValueError(f"No slices for patients {patients}")

    def _simulate(self, clean, gen):
        return simulate_low_dose(
            clean, base_std=self.base_std, signal_std=self.signal_std,
            correlation_sigma=self.correlation_sigma, generator=gen,
        )

    def __getitem__(self, idx):
        pid, i = self.samples[idx]
        clean_np = self.full_volumes[pid][i]
        if self.train and self.patch_size:
            h, w = clean_np.shape
            if h > self.patch_size and w > self.patch_size:
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                clean_np = clean_np[y : y + self.patch_size, x : x + self.patch_size]
        clean = torch.from_numpy(np.array(clean_np)).float()
        gen = None
        if not self.train:
            gen = torch.Generator().manual_seed(self.seed + idx)
        low = self._simulate(clean, gen)
        if self.pair_mode == "noisy":
            gen2 = None
            if not self.train:
                gen2 = torch.Generator().manual_seed(self.seed + idx + 10_000_000)
            full = self._simulate(clean, gen2)
        else:
            full = clean
        return low.unsqueeze(0), full.unsqueeze(0)

    @staticmethod
    def list_patients(h5_path=None, n_patients=8):
        if h5_path:
            with open_h5(h5_path, "r") as f:
                return sorted(f["patients"].keys())
        return [f"phantom{i:03d}" for i in range(n_patients)]

    @classmethod
    def split_patients(cls, h5_path=None, n_patients=8, val_fraction=0.2, seed=0):
        patients = cls.list_patients(h5_path, n_patients)
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
