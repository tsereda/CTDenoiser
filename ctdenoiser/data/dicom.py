"""DICOM series utilities for TCIA LDCT data."""

from pathlib import Path

import numpy as np


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
