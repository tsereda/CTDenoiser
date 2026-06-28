#!/usr/bin/env python3
"""
Summarise a W&B sweep CSV export by *training mode*, the axis that actually
matters for this project, and flag the zsn2n result as the artifact it is.

Why this exists (and how it differs from ``analyze_sweep.py``)
-------------------------------------------------------------
``analyze_sweep.py`` keeps the single best-PSNR run per *model*. Because the
per-image ``zsn2n`` path scores ~41 dB PSNR (vs ~28 dB for trained models) it
would win every model and hide the supervised-vs-n2v comparison. That high
score is misleading: ``zsn2n`` ignores the ``--model`` flag entirely (it trains
its own tiny per-image net) and barely changes the input, so it looks great
only when the low-dose image is already close to the full-dose reference.

This script therefore groups by ``(model, training_mode)``, averages repeated
runs, and prints the *identity baseline* ("do nothing": score the noisy input
directly against the clean reference) whenever the export contains it. The
baseline is the number to compare zsn2n against — if they match, zsn2n is a
no-op.

Usage
-----
    python scripts/sweep_report.py export.csv
    python scripts/sweep_report.py export.csv --out-dir figures/ --fmt pdf
    python scripts/sweep_report.py a.csv b.csv          # combine exports

Outputs
-------
  * a plain-text table grouped by (model, training mode), printed to stdout
  * <out-dir>/psnr_ssim_by_mode.<fmt> — grouped bars: PSNR & SSIM per model,
    one bar colour per training mode, with the identity baseline drawn in.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# metric column -> (pretty label, higher_is_better)
METRICS = {
    "val/psnr": ("PSNR (dB)", True),
    "val/ssim": ("SSIM", True),
    "val/rmse": ("RMSE", False),
    "val/gmsd": ("GMSD", False),
    "val/nps_ratio": ("NPS ratio", False),
}
# W&B exports the per-metric summary under either the bare name or its
# best-epoch aggregation (``.max`` for higher-is-better, ``.min`` otherwise).
# Some exports leave the bare column empty, so fall back to the aggregation.
METRIC_AGG = {
    "val/psnr": "val/psnr.max",
    "val/ssim": "val/ssim.max",
    "val/rmse": "val/rmse.min",
    "val/gmsd": "val/gmsd.min",
    "val/nps_ratio": "val/nps_ratio.min",
}
BASELINE_PSNR = "baseline/psnr"  # logged by ctdenoiser.train if present

MODE_ORDER = ["supervised", "n2sim", "n2v", "f2n", "zsn2n"]
MODE_COLOR = {
    "supervised": "#4C72B0",
    "n2sim":      "#DD8452",
    "n2v":        "#55A868",
    "f2n":        "#8172B2",
    "zsn2n":      "#C44E52",
}


def load(paths) -> pd.DataFrame:
    """Read one or more W&B CSV exports into a single tidy frame.

    Keeps only finished runs that actually logged ``val/psnr``, and resolves the
    training mode from whichever of the inconsistently named columns is present.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if "State" in df.columns:
        df = df[df["State"].astype(str).str.strip() == "finished"].copy()

    # Training mode lives under different headers across exports; coalesce them,
    # filling gaps from one column with the other. Handles pandas' <NA>.
    # ``method`` is the canonical name; some exports leave it blank and only
    # populate ``training-mode``/``training_mode``, so try all three.
    mode = None
    for col in ("method", "training_mode", "training-mode"):
        if col in df.columns:
            s = df[col].astype("string").str.strip().replace("", pd.NA)
            mode = s if mode is None else mode.fillna(s)
    if mode is None:
        mode = pd.Series(pd.NA, index=df.index, dtype="string")
    # A blank/missing mode means the older default: plain supervised training.
    df["mode"] = mode.fillna("supervised").astype(str)

    df["model"] = df["model"].astype("string").str.strip().str.lower().astype(str)

    for col in METRICS:
        s = pd.to_numeric(df.get(col), errors="coerce")
        if not isinstance(s, pd.Series):  # column absent -> all-NaN
            s = pd.Series(np.nan, index=df.index)
        alt = METRIC_AGG.get(col)
        if alt and alt in df.columns:
            s = s.fillna(pd.to_numeric(df[alt], errors="coerce"))
        df[col] = s
    # Drop runs with no PSNR (crashed / never evaluated).
    df = df[df["val/psnr"].notna()].copy()
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per (model, mode), with a run count."""
    g = df.groupby(["model", "mode"])
    out = g[list(METRICS)].mean()
    out["n_runs"] = g.size()
    return out.reset_index()


def baseline_value(df: pd.DataFrame):
    """Mean identity-baseline PSNR if the export logged it, else None."""
    if BASELINE_PSNR in df.columns:
        vals = pd.to_numeric(df[BASELINE_PSNR], errors="coerce").dropna()
        if len(vals):
            return float(vals.mean())
    return None


def print_table(summary: pd.DataFrame, baseline):
    cols = list(METRICS)
    labels = [METRICS[c][0] for c in cols]
    header = f"{'model':<14}{'mode':<12}{'runs':>5}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))

    # Sort by mode (supervised, n2v, zsn2n) then model for readability.
    summary = summary.assign(
        _o=summary["mode"].map(lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 99)
    ).sort_values(["_o", "model"])

    for _, r in summary.iterrows():
        line = f"{r['model']:<14}{r['mode']:<12}{int(r['n_runs']):>5}"
        for c in cols:
            v = r[c]
            cell = "—" if pd.isna(v) else (f"{v:.2f}" if "psnr" in c else f"{v:.4f}")
            line += f"{cell:>12}"
        print(line)

    print()
    if baseline is not None:
        print(f"Identity baseline (do-nothing) PSNR: {baseline:.2f} dB")
        zs = summary.loc[summary["mode"] == "zsn2n", "val/psnr"]
        if len(zs) and abs(zs.mean() - baseline) < 1.0:
            print("  ⚠ zsn2n PSNR ≈ baseline → zsn2n is barely denoising (near no-op).")
    else:
        print("Identity baseline not in this export (re-run training to log "
              "baseline/* and compare it against zsn2n).")
    if (summary["mode"] == "zsn2n").any():
        print("Note: zsn2n ignores --model (it trains a tiny per-image net), so "
              "its score is not comparable to the trained models — read it "
              "against the identity baseline.")


def plot(summary: pd.DataFrame, baseline, out: Path, fmt: str):
    models = sorted(summary["model"].unique())
    modes = [m for m in MODE_ORDER if m in summary["mode"].values]
    x = np.arange(len(models))
    width = 0.8 / max(len(modes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (col, (label, _)) in zip(axes, [("val/psnr", METRICS["val/psnr"]),
                                            ("val/ssim", METRICS["val/ssim"])]):
        for i, mode in enumerate(modes):
            vals = [
                summary.loc[(summary["model"] == m) & (summary["mode"] == mode), col].mean()
                for m in models
            ]
            ax.bar(x + i * width, vals, width, label=mode, color=MODE_COLOR.get(mode))
        if col == "val/psnr" and baseline is not None:
            ax.axhline(baseline, ls="--", color="grey", lw=1.2,
                       label=f"identity baseline ({baseline:.1f})")
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by model & training mode")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"psnr_ssim_by_mode.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nsaved → {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="+", help="W&B exported CSV file(s)")
    ap.add_argument("--out-dir", default="figures", help="output dir (default: figures/)")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    df = load(args.csv)
    if df.empty:
        print("ERROR: no finished runs with metrics found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} finished runs "
          f"({df['model'].nunique()} models, modes: {sorted(df['mode'].unique())})\n")
    summary = summarise(df)
    base = baseline_value(df)
    print_table(summary, base)
    plot(summary, base, Path(args.out_dir), args.fmt)


if __name__ == "__main__":
    main()
