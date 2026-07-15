#!/usr/bin/env python3
"""Task-based detectability figure (paper Section~\\ref{sec:detect}, Table tab:detect).

Reads a W&B export of the main benchmark sweep -- the one that already carries the
``det/*`` columns -- and produces the three-panel detectability figure:

  (a) CHO detectability *preserved*  d'_den / d'_in  vs lesion contrast (40/80/160 HU),
      one line per supervision regime, with the 1.0 "unchanged" reference. Shows the
      erosion concentrating at the high-contrast operating point.
  (b) NPS radial-centroid shift  input -> denoised  per regime: the low-frequency
      ("waxy") drift that PSNR/SSIM miss and radiologists distrust.
  (c) Fabrication d' per regime against the signal-absent input floor: erosion
      *without* invented structure.

No new compute -- the detectability metrics are already logged in the sweep export.

    python scripts/figure_detectability.py sweep_export.csv --out-dir figures --fmt pdf

By default aggregates the three CNN backbones (RED-CNN, U-Net, DnCNN) x seeds to
match Table tab:detect; pass --models '' to use every finished run with det/* data.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── aesthetics (match scripts/figures.py) ─────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

REGIMES = ["supervised", "n2sim", "n2v"]
REGIME_LABEL = {"supervised": "Supervised", "n2sim": "Noise2Sim", "n2v": "Noise2Void"}
REGIME_COLOR = {"supervised": "#4C72B0", "n2sim": "#55A868", "n2v": "#DD8452"}
CNN_MODELS = ["redcnn", "unet", "dncnn"]
CONTRASTS = [40, 80, 160]


def col(df: pd.DataFrame, *names) -> pd.Series:
    """First present column among candidates, numeric-coerced (dashed vs underscore)."""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def load(path: str, models) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["State"] == "finished"].copy()
    df["_mode"] = df["training_mode"].fillna(df.get("training-mode"))
    df["_model"] = df["model"]
    df = df[col(df, "det/c160/detectability_preserved").notna()]
    if models:
        df = df[df["_model"].isin(models)]
    return df


def aggregate(df: pd.DataFrame):
    stats = {}
    for r in REGIMES:
        sub = df[df["_mode"] == r]
        if sub.empty:
            continue
        d = {c: (col(sub, f"det/c{c}/detectability_preserved").mean(),
                 col(sub, f"det/c{c}/detectability_preserved").std())
             for c in CONTRASTS}
        d["nps_den"] = col(sub, "det/nps_mean_freq_denoised").mean()
        fab = col(sub, "det/d_prime_fabrication")
        d["fab"] = (fab.mean(), fab.std())
        stats[r] = d
    nps_in = col(df, "det/nps_mean_freq_input").mean()
    fab_in = col(df, "det/d_prime_fabrication_input").mean()
    return stats, nps_in, fab_in


def make_fig(df: pd.DataFrame, out_dir: str, fmt: str) -> Path:
    stats, nps_in, fab_in = aggregate(df)
    present = [r for r in REGIMES if r in stats]
    x = np.arange(len(present))
    colors = [REGIME_COLOR[r] for r in present]
    labels = [REGIME_LABEL[r] for r in present]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # (a) preserved d' vs contrast
    ax = axes[0]
    for r in present:
        ys = [stats[r][c][0] for c in CONTRASTS]
        es = [stats[r][c][1] for c in CONTRASTS]
        ax.errorbar(CONTRASTS, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                    color=REGIME_COLOR[r], label=REGIME_LABEL[r])
    ax.axhline(1.0, ls="--", color="#7f7f7f", lw=1)
    ax.text(CONTRASTS[-1], 1.002, "unchanged", color="#7f7f7f", fontsize=8,
            ha="right", va="bottom")
    ax.set_xticks(CONTRASTS)
    ax.set_xlabel("lesion contrast (HU)")
    ax.set_ylabel(r"detectability preserved  $d'_{\mathrm{den}}/d'_{\mathrm{in}}$")
    ax.set_title("(a) Detectability eroded at high contrast")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # (b) NPS radial centroid: input reference + per-regime denoised
    ax = axes[1]
    ax.axhline(nps_in, ls="--", color="#7f7f7f", lw=1)
    ax.text(x[-1] + 0.45, nps_in, "input", color="#7f7f7f", fontsize=8, va="center")
    ax.bar(x, [stats[r]["nps_den"] for r in present], color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("NPS radial centroid (cyc/px)")
    ax.set_title('(b) Noise spectrum shifts low ("waxy")')

    # (c) fabrication d' vs signal-absent input floor
    ax = axes[2]
    ax.axhline(fab_in, ls="--", color="#7f7f7f", lw=1)
    ax.text(x[-1] + 0.45, fab_in, "input\nfloor", color="#7f7f7f", fontsize=8, va="center")
    ax.bar(x, [stats[r]["fab"][0] for r in present],
           yerr=[stats[r]["fab"][1] for r in present], capsize=3,
           color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"fabrication $d'$")
    ax.set_title("(c) No fabricated structure")

    fig.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"detectability.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="W&B export of the benchmark sweep (with det/* columns)")
    ap.add_argument("--out-dir", default="figures", help="output dir (default: figures/)")
    ap.add_argument("--fmt", default="pdf", choices=["png", "pdf", "svg"])
    ap.add_argument("--models", default=",".join(CNN_MODELS),
                    help="comma-separated model filter (default: the 3 CNNs; "
                         "pass empty string for all models)")
    a = ap.parse_args()

    models = [m for m in a.models.split(",") if m] if a.models else None
    df = load(a.csv, models)
    if df.empty:
        raise SystemExit("no finished runs with det/* columns found in export")
    print(f"{len(df)} runs with detectability; regimes: {sorted(df['_mode'].unique())}")
    path = make_fig(df, a.out_dir, a.fmt)
    print("wrote", path)


if __name__ == "__main__":
    main()
