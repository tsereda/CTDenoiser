#!/usr/bin/env python
"""Convert paired DICOM series to a single preprocessed HDF5 file.

Run once on the data pod / PVC to create a compact file that sweep agents
can copy in seconds instead of minutes.

    python scripts/convert_dicom_to_h5.py --dicom-root /data/ldct_dicom
"""

import argparse
import time

import h5py
import numpy as np

from ctdenoiser.data.dataset import DICOMCTDataset
from ctdenoiser.data.dicom import read_series_hu


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

    mapping = DICOMCTDataset._scan_series(args.dicom_root)
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
