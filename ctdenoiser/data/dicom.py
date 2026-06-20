"""DICOM series utilities for TCIA LDCT data."""

from pathlib import Path

import numpy as np


def scan_paired_series(root: str) -> dict[str, dict[str, Path]]:
    """Scan a root of SeriesInstanceUID dirs into ``{pid: {low, full}}``.

    Low/full dose is read from each series' ``SeriesDescription`` header and
    paired by ``PatientID``. Only patients with *both* a low- and a full-dose
    series are returned (paired supervised training needs both).

    Raises if a single patient has two series mapped to the *same* dose level
    (e.g. two "Low Dose Images" reconstructions). That is the silent-overwrite
    footgun the old ``mapping[pid][dose] = sdir`` had: it would keep an
    arbitrary one and mis-pair the patient. Surfacing it lets the caller decide.
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required: pip install pydicom") from exc

    root_path = Path(root)
    mapping: dict[str, dict[str, Path]] = {}
    for sdir in sorted(root_path.iterdir()):
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
        series = mapping.setdefault(pid, {})
        if dose in series:
            raise ValueError(
                f"Patient {pid} has two '{dose} dose' series:\n"
                f"  {series[dose]}\n  {sdir}\n"
                "Ambiguous pairing -- keep only one series per (patient, dose), "
                "e.g. point --dicom-root at a single anatomy/reconstruction."
            )
        series[dose] = sdir

    complete = {
        pid: series
        for pid, series in mapping.items()
        if "low" in series and "full" in series
    }
    if not complete:
        raise ValueError(
            f"No paired low/full dose patients found in {root}. "
            f"Detected: {mapping}"
        )
    return complete


def read_series_hu(series_dir: str) -> np.ndarray:
    """Load all DICOM slices in a directory, sort by z-position, return HU array.

    Requires pydicom: pip install pydicom
    Returns shape (num_slices, H, W) float32 with raw Hounsfield unit values.
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required: pip install pydicom") from exc

    dcms = sorted(
        Path(series_dir).glob("*.dcm"),
        key=lambda p: float(
            pydicom.dcmread(str(p), stop_before_pixels=True).ImagePositionPatient[2]
        ),
    )
    if not dcms:
        raise ValueError(f"No .dcm files in {series_dir}")
    slices = []
    for p in dcms:
        ds = pydicom.dcmread(str(p))
        hu = (
            ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
            + float(ds.RescaleIntercept)
        )
        slices.append(hu)
    return np.stack(slices, axis=0)
