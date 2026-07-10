#!/usr/bin/env python3
"""
Paper figures for the self-supervised rectified-flow (``ssflow``) ablation.

Why this exists (and how it differs from the other scripts)
-----------------------------------------------------------
``analyze_sweep.py`` keeps the best run *per model* and ``sweep_report.py``
groups *per training mode* — neither understands the two axes that the dedicated
``sweep_hallucination.yml`` grid (which supersedes the retired ``sweep_ssflow.yml``)
actually varies:

  * ``ssflow-exclude-radius`` ∈ {1, 2, 3}  — the correlated-noise knob. r=1 is
    the Noise2Sim target (self-exclusion only); r=2,3 push matched pixels beyond
    the FBP noise-correlation length so the manufactured pair is decorrelated.
  * ``flow-steps-eval`` ∈ {1, 4}           — one-step posterior mean (== a
    similarity-based Noise2Noise regression) vs multi-step Euler refinement.

This script reads the ssflow sweep export and produces three figures:

  ssflow_ablation.{fmt}            — PSNR & SSIM vs exclude-radius, one line per
                                     eval-step setting (the 3x2 grid). Each point
                                     is annotated with its logged epoch, because
                                     a running sweep has runs at different epochs
                                     and the 4-step eval is ~4x slower.
  ssflow_leaderboard.{fmt}         — PSNR gain over the identity baseline for the
                                     best ssflow run placed against the published
                                     benchmark (Table 1 in paper/main.tex),
                                     coloured by tier. Shows where the label-free
                                     flow lands relative to Noise2Sim / N2V /
                                     zero-shot and the supervised models.
  ssflow_fidelity_vs_texture.{fmt} — PSNR vs GMSD scatter. ssflow tops the pixel
                                     metrics yet trails on the gradient/texture
                                     metric — the tension the paper is about.

The benchmark numbers are baked in from paper/main.tex (Mayo TCIA abdomen, same
val patients and identity baseline as the sweep) so the comparison needs only
the ssflow CSV. Edit ``BENCHMARK`` if the leaderboard is re-run.

Usage
-----
    python scripts/figures.py ssflow_export.csv
    python scripts/figures.py ssflow_export.csv --out-dir figures/ --fmt pdf

Unlike the other scripts this keeps ``running`` runs (the ssflow sweep is often
read mid-flight); it filters to rows whose model is ``ssflow`` and that have a
logged ``val/psnr``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── aesthetics (match analyze_sweep.py) ───────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Tier → colour, shared across the leaderboard and the scatter.
TIER_COLOR = {
    "supervised": "#4C72B0",
    "noise2sim":  "#55A868",
    "noise2void": "#DD8452",
    "zeroshot":   "#937860",
    "ssflow":     "#C44E52",   # ours — highlighted
    "baseline":   "#7f7f7f",
}
TIER_LABEL = {
    "supervised": "Supervised",
    "noise2sim":  "Noise2Sim (self-sup.)",
    "noise2void": "Noise2Void (self-sup.)",
    "zeroshot":   "Zero-shot",
    "ssflow":     "SSFlow (ours)",
    "baseline":   "LDCT input (no denoising)",
}

# Published benchmark — paper/main.tex Table 1 (Mayo TCIA abdomen).
# (label, tier, psnr, ssim, rmse, gmsd, nps).  Same identity baseline and val
# patients as the ssflow sweep, so the rows are directly comparable.
BENCHMARK = [
    ("LDCT input",         "baseline",   31.44, 0.893, 0.0274, 0.166, 0.0079),
    ("RED-CNN",            "supervised", 35.48, 0.940, 0.0171, 0.166, 0.0031),
    ("U-Net",              "supervised", 35.47, 0.940, 0.0171, 0.178, 0.0030),
    ("DnCNN",              "supervised", 35.40, 0.938, 0.0172, 0.184, 0.0031),
    ("CTformer",           "supervised", 34.78, 0.917, 0.0184, 0.263, 0.0036),
    ("CFM",                "supervised", 33.87, 0.923, 0.0206, 0.248, 0.0044),
    ("U-Net + N2Sim",      "noise2sim",  32.61, 0.910, 0.0238, 0.169, 0.0060),
    ("DnCNN + N2Sim",      "noise2sim",  32.14, 0.891, 0.0251, 0.178, 0.0067),
    ("RED-CNN + N2Sim",    "noise2sim",  31.94, 0.904, 0.0257, 0.162, 0.0070),
    ("CTformer + N2Sim",   "noise2sim",  31.09, 0.885, 0.0282, 0.247, 0.0083),
    ("RED-CNN + N2V",      "noise2void", 31.52, 0.895, 0.0271, 0.175, 0.0077),
    ("U-Net + N2V",        "noise2void", 31.49, 0.893, 0.0271, 0.188, 0.0078),
    ("CTformer + N2V",     "noise2void", 31.28, 0.894, 0.0277, 0.223, 0.0081),
    ("DnCNN + N2V",        "noise2void", 30.98, 0.885, 0.0288, 0.198, 0.0087),
    ("Filter2Noise",       "zeroshot",   31.58, 0.896, 0.0269, 0.176, 0.0076),
    ("ZS-N2N",             "zeroshot",   31.17, 0.892, 0.0281, 0.182, 0.0083),
]
BENCHMARK_COLS = ["label", "tier", "val/psnr", "val/ssim", "val/rmse",
                  "val/gmsd", "val/nps_ratio"]

BASELINE_PSNR_FALLBACK = 31.44  # Table-1 LDCT input, used if CSV lacks baseline/*


# ── load ──────────────────────────────────────────────────────────────────────

def _coalesce(df: pd.DataFrame, names) -> pd.Series:
    """First non-null value across the candidate columns (dashed vs underscore).

    W&B exports the same hyper-parameter under both ``flow-steps-eval`` and
    ``flow_steps_eval`` depending on how it was set; merge whichever is present.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for n in names:
        if n in df.columns:
            out = out.fillna(pd.to_numeric(df[n], errors="coerce"))
    return out


def load(path: str) -> tuple[pd.DataFrame, float]:
    """Return (tidy ssflow runs, identity-baseline PSNR)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    model = df.get("model", pd.Series("", index=df.index)).astype("string").str.strip().str.lower()
    df = df[model == "ssflow"].copy()

    for col in ("val/psnr", "val/ssim", "val/rmse", "val/gmsd", "val/nps_ratio", "epoch"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df[df["val/psnr"].notna()].copy()

    df["exclude_radius"] = _coalesce(df, ["ssflow_exclude_radius", "ssflow-exclude-radius"])
    df["eval_steps"] = _coalesce(df, ["flow_steps_eval", "flow-steps-eval"])
    df["name"] = df.get("Name", pd.Series("", index=df.index)).astype(str)

    # Identity baseline: mean of logged baseline/psnr if present, else Table 1.
    base = pd.to_numeric(df.get("baseline/psnr"), errors="coerce").dropna()
    baseline = float(base.mean()) if len(base) else BASELINE_PSNR_FALLBACK
    return df, baseline


def ssflow_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (eval_steps, exclude_radius) — the furthest-trained run."""
    return (
        df.sort_values("epoch")
          .drop_duplicates(subset=["eval_steps", "exclude_radius"], keep="last")
          .sort_values(["eval_steps", "exclude_radius"])
    )


# ── figure 1: the ablation grid ───────────────────────────────────────────────

def fig_ablation(grid: pd.DataFrame, baseline: float, out: Path, fmt: str):
    radii = sorted(grid["exclude_radius"].dropna().unique())
    steps = sorted(grid["eval_steps"].dropna().unique())
    step_style = {s: sty for s, sty in zip(steps, [("o-", "#C44E52"), ("s--", "#4C72B0")])}

    panels = [("val/psnr", "PSNR (dB)  ↑", baseline),
              ("val/ssim", "SSIM  ↑", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.2))

    for ax, (col, label, base) in zip(axes, panels):
        for s in steps:
            sub = grid[grid["eval_steps"] == s].sort_values("exclude_radius")
            marker, color = step_style[s]
            ax.plot(sub["exclude_radius"], sub[col], marker, color=color,
                    label=f"{int(s)}-step eval", linewidth=1.8, markersize=7,
                    zorder=3)
            for _, r in sub.iterrows():  # annotate epoch — runs differ mid-sweep
                ax.annotate(f"ep{int(r['epoch'])}", (r["exclude_radius"], r[col]),
                            fontsize=6, color="grey", xytext=(4, 4),
                            textcoords="offset points")
        if base is not None:
            ax.axhline(base, ls=":", color=TIER_COLOR["baseline"], lw=1.2,
                       label=f"LDCT input ({base:.2f})")
        ax.set_xticks(radii)
        ax.set_xlabel("ssflow exclude-radius  (decorrelation knob →)")
        ax.set_ylabel(label)
        ax.set_title(label.split()[0])
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("ssflow ablation: decorrelation radius × eval steps", y=1.01)
    _save(fig, out / f"ssflow_ablation.{fmt}")


# ── figure 2: leaderboard (PSNR gain over baseline) ───────────────────────────

def fig_leaderboard(grid: pd.DataFrame, baseline: float, out: Path, fmt: str):
    bench = pd.DataFrame(BENCHMARK, columns=BENCHMARK_COLS)
    bench = bench[bench["tier"] != "baseline"]  # baseline is the zero line

    best = grid.sort_values("val/psnr").iloc[-1]
    ours = pd.DataFrame([{
        "label": f"SSFlow {int(best['eval_steps'])}-step r{int(best['exclude_radius'])} (ours)",
        "tier": "ssflow", "val/psnr": best["val/psnr"],
    }])
    rows = pd.concat([bench[["label", "tier", "val/psnr"]], ours], ignore_index=True)
    rows["gain"] = rows["val/psnr"] - baseline
    rows = rows.sort_values("gain")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    y = np.arange(len(rows))
    colors = [TIER_COLOR[t] for t in rows["tier"]]
    edges = ["black" if t == "ssflow" else "white" for t in rows["tier"]]
    widths = [2.0 if t == "ssflow" else 0.5 for t in rows["tier"]]
    ax.barh(y, rows["gain"], color=colors, edgecolor=edges, linewidth=widths, zorder=3)

    for yi, (g, p) in enumerate(zip(rows["gain"], rows["val/psnr"])):
        ax.text(g + (0.04 if g >= 0 else -0.04), yi, f"{g:+.2f}",
                va="center", ha="left" if g >= 0 else "right", fontsize=7)

    ax.axvline(0, color=TIER_COLOR["baseline"], lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=8)
    ax.set_xlabel("PSNR gain over LDCT input (dB)  ↑")
    ax.set_title("Where ssflow lands on the benchmark", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in
               ["supervised", "noise2sim", "noise2void", "zeroshot", "ssflow"]]
    ax.legend(handles, [TIER_LABEL[t] for t in
              ["supervised", "noise2sim", "noise2void", "zeroshot", "ssflow"]],
              fontsize=7.5, frameon=False, loc="lower right")
    _save(fig, out / f"ssflow_leaderboard.{fmt}")


# ── figure 3: fidelity (PSNR) vs texture (GMSD) ───────────────────────────────

def fig_fidelity_vs_texture(grid: pd.DataFrame, out: Path, fmt: str):
    bench = pd.DataFrame(BENCHMARK, columns=BENCHMARK_COLS)

    fig, ax = plt.subplots(figsize=(7, 5))
    # Benchmark methods as points, coloured by tier.
    for tier, sub in bench.groupby("tier"):
        ax.scatter(sub["val/psnr"], sub["val/gmsd"], color=TIER_COLOR[tier],
                   s=60, zorder=3, edgecolors="white", linewidths=0.5,
                   label=TIER_LABEL[tier])
        for _, r in sub.iterrows():
            ax.annotate(r["label"], (r["val/psnr"], r["val/gmsd"]), fontsize=6,
                        color="grey", xytext=(4, 2), textcoords="offset points")

    # Every ssflow run (the whole grid, not just the best).
    ax.scatter(grid["val/psnr"], grid["val/gmsd"], color=TIER_COLOR["ssflow"],
               s=90, marker="*", zorder=4, edgecolors="black", linewidths=0.6,
               label=TIER_LABEL["ssflow"])
    for _, r in grid.iterrows():
        ax.annotate(f"{int(r['eval_steps'])}st·r{int(r['exclude_radius'])}",
                    (r["val/psnr"], r["val/gmsd"]), fontsize=6,
                    color=TIER_COLOR["ssflow"], xytext=(4, -8),
                    textcoords="offset points")

    ax.set_xlabel("PSNR (dB)  →  better fidelity")
    ax.set_ylabel("GMSD  →  worse texture (↓ better)")
    ax.invert_yaxis()  # so up = better on both axes
    ax.set_title("Pixel fidelity vs gradient/texture similarity", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _save(fig, out / f"ssflow_fidelity_vs_texture.{fmt}")


# ── util / main ───────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="W&B export of the ssflow sweep")
    ap.add_argument("--out-dir", default="figures", help="output dir (default: figures/)")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    df, baseline = load(args.csv)
    if df.empty:
        print("ERROR: no ssflow runs with a logged val/psnr in this CSV.",
              file=sys.stderr)
        sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = ssflow_grid(df)

    n_running = (grid["epoch"].max() - grid["epoch"].min())
    print(f"Loaded {len(df)} ssflow runs; identity baseline = {baseline:.2f} dB")
    print(f"Grid: eval_steps={sorted(grid['eval_steps'].unique())}, "
          f"exclude_radius={sorted(grid['exclude_radius'].unique())}")
    if n_running > 1:
        print(f"  ⚠ runs span epochs {int(grid['epoch'].min())}–{int(grid['epoch'].max())}"
              " — comparisons across eval-steps are not yet at equal training.")

    best = grid.sort_values("val/psnr").iloc[-1]
    print(f"Best so far: {best['name']} "
          f"({int(best['eval_steps'])}-step, r={int(best['exclude_radius'])}) "
          f"PSNR {best['val/psnr']:.2f} ({best['val/psnr'] - baseline:+.2f} dB), "
          f"SSIM {best['val/ssim']:.3f}, GMSD {best['val/gmsd']:.3f}\n")

    print("Generating figures…")
    fig_ablation(grid, baseline, out, args.fmt)
    fig_leaderboard(grid, baseline, out, args.fmt)
    fig_fidelity_vs_texture(grid, out, args.fmt)


if __name__ == "__main__":
    main()
