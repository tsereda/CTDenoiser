#!/usr/bin/env python
"""Convert paired DICOM series to a single preprocessed HDF5 file.

Run once on the data pod / PVC to create a compact file that sweep agents
can copy in seconds instead of minutes.

    python scripts/convert_dicom_to_h5.py --dicom-root /data/ldct_dicom
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def read_series_hu(series_dir: str) -> np.ndarray:
    """Load sorted DICOM slices from a directory and return HU volume."""
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


def scan_paired_series(root: str) -> dict[str, dict[str, Path]]:
    """Return {patient_id: {low: Path, full: Path}} for paired DICOM series."""
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
        mapping.setdefault(pid, {})
        mapping[pid][dose] = sdir

    complete = {
        pid: series
        for pid, series in mapping.items()
        if "low" in series and "full" in series
    }
    if not complete:
        raise ValueError(
            f"No paired low/full dose patients found in {root}. Detected: {mapping}"
        )
    return complete


def main():
    parser = argparse.ArgumentParser(
        description="Convert paired DICOM series to HDF5."
    )
    parser.add_argument(
        "--dicom-root", default="/data/ldct_dicom",
        help="directory containing SeriesInstanceUID subdirs",
    )
    parser.add_argument(
        "--output", default="/data/ldct_preprocessed.h5",
        help="output HDF5 path",
    )
    parser.add_argument("--hu-offset", type=float, default=1000.0)
    parser.add_argument("--hu-scale", type=float, default=2000.0)
    parser.add_argument(
        "--compression", default="gzip",
        help="HDF5 compression filter (default: gzip)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=4,
        help="compression level (default: 4)",
    )
    args = parser.parse_args()

    mapping = scan_paired_series(args.dicom_root)
    patients = sorted(mapping.keys())
    print(f"Found {len(patients)} paired patients: {patients}")

    t0 = time.time()
    with h5py.File(args.output, "w") as f:
        f.attrs["hu_offset"] = args.hu_offset
        f.attrs["hu_scale"] = args.hu_scale

        for i, pid in enumerate(patients, 1):
            t_pat = time.time()
            low_dir = str(mapping[pid]["low"])
            full_dir = str(mapping[pid]["full"])

            low_vol = read_series_hu(low_dir).astype(np.float32)
            full_vol = read_series_hu(full_dir).astype(np.float32)

            low_vol = np.clip(
                (low_vol + args.hu_offset) / args.hu_scale, 0.0, 1.0
            )
            full_vol = np.clip(
                (full_vol + args.hu_offset) / args.hu_scale, 0.0, 1.0
            )

            n_slices = min(low_vol.shape[0], full_vol.shape[0])
            if low_vol.shape[0] != full_vol.shape[0]:
                print(
                    f"  Warning: {pid} slice mismatch "
                    f"(low={low_vol.shape[0]}, full={full_vol.shape[0]}), "
                    f"using {n_slices}"
                )
            low_vol = low_vol[:n_slices]
            full_vol = full_vol[:n_slices]

            grp = f.create_group(f"patients/{pid}")
            chunks = (1, low_vol.shape[1], low_vol.shape[2])
            grp.create_dataset(
                "low", data=low_vol, chunks=chunks,
                compression=args.compression,
                compression_opts=args.compression_level,
            )
            grp.create_dataset(
                "full", data=full_vol, chunks=chunks,
                compression=args.compression,
                compression_opts=args.compression_level,
            )

            elapsed = time.time() - t_pat
            print(
                f"  [{i}/{len(patients)}] {pid}: "
                f"{n_slices} slices, shape {low_vol.shape[1:]}, "
                f"{elapsed:.1f}s"
            )

    total = time.time() - t0
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nDone: {args.output} ({size_mb:.1f} MB) in {total:.1f}s")


if __name__ == "__main__":
    main()
