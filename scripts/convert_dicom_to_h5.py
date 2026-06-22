#!/usr/bin/env python
"""Convert paired DICOM series to a single preprocessed HDF5 file.

Run once on the data pod / PVC to create a compact file that sweep agents
can copy in seconds instead of minutes.

    python scripts/convert_dicom_to_h5.py --dicom-root /data/ldct_dicom
"""

import argparse
import os
import time

import h5py
import numpy as np

from ctdenoiser.data.dataset import ANATOMY_WINDOWS, window_for_anatomy
from ctdenoiser.data.dicom import read_series_hu, scan_paired_series


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
    # The HU window is baked into the cache, so pick it to match the anatomy of
    # the scans being converted. --anatomy sets a clinical preset; --hu-offset/
    # --hu-scale override it for a custom window. Default: abdomen soft tissue
    # (level 40 HU, width 400 HU -> [-160, +240]). A wide window compresses the
    # low/full dose difference to ~1% of [0, 1] and makes the task near-trivial.
    parser.add_argument("--anatomy", choices=sorted(ANATOMY_WINDOWS),
                        default="abdomen",
                        help="HU window preset (default: abdomen)")
    parser.add_argument("--hu-offset", type=float, default=None,
                        help="override the --anatomy window offset")
    parser.add_argument("--hu-scale", type=float, default=None,
                        help="override the --anatomy window scale")
    parser.add_argument(
        "--compression", default="gzip",
        help="HDF5 compression filter (default: gzip)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=4,
        help="compression level (default: 4)",
    )
    args = parser.parse_args()

    offset, scale = window_for_anatomy(args.anatomy)
    if args.hu_offset is not None:
        offset = args.hu_offset
    if args.hu_scale is not None:
        scale = args.hu_scale

    mapping = scan_paired_series(args.dicom_root)
    patients = sorted(mapping.keys())
    print(f"Found {len(patients)} paired patients: {patients}")
    print(f"Window: anatomy={args.anatomy} offset={offset} scale={scale}")

    # Write to a temp file and rename only once the whole cache is on disk, so
    # an interrupted/crashed run never leaves a half-written .h5 in place. A
    # truncated cache fails to open later with the cryptic h5py "bad object
    # header version number" error, taking down every sweep agent that reads it.
    t0 = time.time()
    tmp_output = f"{args.output}.tmp"
    with h5py.File(tmp_output, "w") as f:
        f.attrs["hu_offset"] = offset
        f.attrs["hu_scale"] = scale
        f.attrs["anatomy"] = args.anatomy

        for i, pid in enumerate(patients, 1):
            t_pat = time.time()
            low_dir = str(mapping[pid]["low"])
            full_dir = str(mapping[pid]["full"])

            low_vol = read_series_hu(low_dir).astype(np.float32)
            full_vol = read_series_hu(full_dir).astype(np.float32)

            low_vol = np.clip((low_vol + offset) / scale, 0.0, 1.0)
            full_vol = np.clip((full_vol + offset) / scale, 0.0, 1.0)

            n_slices = min(low_vol.shape[0], full_vol.shape[0])
            if low_vol.shape[0] != full_vol.shape[0]:
                print(
                    f"  Warning: {pid} slice mismatch "
                    f"(low={low_vol.shape[0]}, full={full_vol.shape[0]}), "
                    f"using {n_slices}"
                )
            low_vol = low_vol[:n_slices]
            full_vol = full_vol[:n_slices]

            noise = float(np.mean(np.abs(low_vol.astype(np.float64) - full_vol)))
            if noise < 1e-4:
                print(
                    f"  Warning: {pid} low and full are nearly identical "
                    f"(mean|low-full|={noise:.2e}); check pairing / HU window."
                )

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
                f"noise(mean|low-full|)={noise:.4f}, {elapsed:.1f}s"
            )

    os.replace(tmp_output, args.output)

    total = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nDone: {args.output} ({size_mb:.1f} MB) in {total:.1f}s")


if __name__ == "__main__":
    main()
