#!/usr/bin/env python3
"""
Analyze W&B sweep CSV export and produce publication-ready figures.

Usage
-----
    python scripts/analyze_sweep.py results.csv
    python scripts/analyze_sweep.py results.csv --out-dir figures/ --fmt pdf

Figures produced
----------------
  01_metrics_bar.{fmt}    — grouped bar chart, one panel per metric
  02_radar.{fmt}          — radar chart comparing best-of-model runs
  03_scatter_psnr_ssim.{fmt} — PSNR vs SSIM scatter
  04_heatmap.{fmt}        — model × metric heat-map (z-scored)

A LaTeX-ready summary table is printed to stdout.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

MODEL_ORDER = ["dncnn", "redcnn", "unet", "ctformer", "flowmatching"]
MODEL_LABELS = {
    "dncnn": "DnCNN",
    "redcnn": "RED-CNN",
    "unet": "U-Net",
    "ctformer": "CTformer",
    "flowmatching": "Flow\nMatching",
}
PALETTE = {
    "dncnn":        "#4C72B0",
    "redcnn":       "#DD8452",
    "unet":         "#55A868",
    "ctformer":     "#C44E52",
    "flowmatching": "#8172B2",
}

# metric display config: (column, label, higher_is_better)
METRICS = [
    ("val/psnr",      "PSNR (dB)",  True),
    ("val/ssim",      "SSIM",       True),
    ("val/rmse",      "RMSE",       False),
    ("val/gmsd",      "GMSD",       False),
    ("val/nps_ratio", "NPS ratio",  False),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["State"] == "finished"].copy()
    for col, *_ in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model"] = df["model"].str.strip().str.lower()
    return df


def best_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per model — the run with highest PSNR."""
    return (
        df.sort_values("val/psnr", ascending=False)
          .drop_duplicates(subset="model")
          .set_index("model")
          .reindex([m for m in MODEL_ORDER if m in df["model"].values])
    )


def fmt_val(col, val):
    if "psnr" in col:
        return f"{val:.2f}"
    return f"{val:.4f}"


# ── figure 1: grouped bar chart ───────────────────────────────────────────────

def fig_metrics_bar(best: pd.DataFrame, out: Path, fmt: str):
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 3.8))
    fig.subplots_adjust(wspace=0.45)

    models = best.index.tolist()
    x = np.arange(len(models))

    for ax, (col, label, hib) in zip(axes, METRICS):
        vals = best[col].values
        colors = [PALETTE.get(m, "#aaaaaa") for m in models]
        bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

        # value labels on bars
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ypos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, ypos + ypos * 0.01,
                    fmt_val(col, v), ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        arrow = "↑ better" if hib else "↓ better"
        ax.set_title(f"{label}\n{arrow}", fontsize=9)

    fig.suptitle("Best-run comparison across models (1 epoch each)", fontsize=11, y=1.01)
    _save(fig, out / f"01_metrics_bar.{fmt}")


# ── figure 2: radar chart ─────────────────────────────────────────────────────

def fig_radar(best: pd.DataFrame, out: Path, fmt: str):
    cols = [c for c, *_ in METRICS]
    hib  = [h for _, _, h in METRICS]
    labels = [l for _, l, _ in METRICS]

    # normalise each metric to [0, 1] where 1 = best
    normed = best[cols].copy()
    for col, higher in zip(cols, hib):
        mn, mx = normed[col].min(), normed[col].max()
        if mx == mn:
            normed[col] = 1.0
        elif higher:
            normed[col] = (normed[col] - mn) / (mx - mn)
        else:
            normed[col] = (mx - normed[col]) / (mx - mn)

    N = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="grey")
    ax.set_ylim(0, 1)

    for model in normed.index:
        vals = normed.loc[model, cols].tolist()
        vals += vals[:1]
        color = PALETTE.get(model, "#aaaaaa")
        ax.plot(angles, vals, color=color, linewidth=1.8, label=MODEL_LABELS.get(model, model))
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8,
              frameon=False)
    ax.set_title("Multi-metric comparison\n(normalised, higher = better)", fontsize=10, pad=20)
    _save(fig, out / f"02_radar.{fmt}")


# ── figure 3: PSNR vs SSIM scatter ───────────────────────────────────────────

def fig_scatter(df: pd.DataFrame, out: Path, fmt: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    for model, grp in df.groupby("model"):
        color = PALETTE.get(model, "#aaaaaa")
        label = MODEL_LABELS.get(model, model).replace("\n", " ")
        ax.scatter(grp["val/psnr"], grp["val/ssim"],
                   color=color, label=label, s=70, zorder=3,
                   edgecolors="white", linewidths=0.5)
        for _, row in grp.iterrows():
            ax.annotate(
                f"bs={int(row['batch_size'])}\nlr={row['lr']}",
                (row["val/psnr"], row["val/ssim"]),
                fontsize=6, color="grey",
                xytext=(4, 2), textcoords="offset points",
            )

    ax.set_xlabel("PSNR (dB)  ↑", fontsize=10)
    ax.set_ylabel("SSIM  ↑", fontsize=10)
    ax.set_title("PSNR vs SSIM — all sweep runs", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, out / f"03_scatter_psnr_ssim.{fmt}")


# ── figure 4: heatmap ─────────────────────────────────────────────────────────

def fig_heatmap(best: pd.DataFrame, out: Path, fmt: str):
    cols = [c for c, *_ in METRICS]
    labels = [l for _, l, _ in METRICS]
    hib = [h for _, _, h in METRICS]

    data = best[cols].copy()
    # flip lower-is-better so the heatmap reads intuitively (darker = worse)
    display = data.copy()
    for col, higher in zip(cols, hib):
        if not higher:
            mn, mx = data[col].min(), data[col].max()
            display[col] = (mx - data[col]) / (mx - mn) if mx != mn else 0.5

    # z-score for colour, raw for annotation
    z = (display - display.mean()) / (display.std() + 1e-9)

    row_labels = [MODEL_LABELS.get(m, m).replace("\n", " ") for m in best.index]
    fig, ax = plt.subplots(figsize=(len(cols) * 1.5 + 1, len(best) * 0.7 + 1.5))
    im = ax.imshow(z.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i, model in enumerate(best.index):
        for j, (col, _, hib_flag) in enumerate(METRICS):
            raw = data.loc[model, col]
            suffix = "" if np.isnan(raw) else ""
            txt = fmt_val(col, raw) if not np.isnan(raw) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")

    plt.colorbar(im, ax=ax, label="z-score (green = best in column)", shrink=0.6)
    ax.set_title("Best-run model × metric overview", fontsize=11, pad=10)
    plt.tight_layout()
    _save(fig, out / f"04_heatmap.{fmt}")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex_table(best: pd.DataFrame):
    cols   = [c for c, *_ in METRICS]
    labels = [l for _, l, _ in METRICS]
    hib    = [h for _, _, h in METRICS]

    col_fmt = "l" + "r" * len(cols)
    print("\n% ── LaTeX table ─────────────────────────────────────────")
    print(r"\begin{table}[h]")
    print(r"  \centering")
    print(r"  \caption{CT denoising model comparison (1 epoch, preliminary).}")
    print(r"  \label{tab:model_comparison}")
    print(f"  \\begin{{tabular}}{{{col_fmt}}}")
    print(r"    \toprule")
    header = "    Model & " + " & ".join(labels) + r" \\"
    print(header)
    print(r"    \midrule")

    for model in best.index:
        row = MODEL_LABELS.get(model, model).replace("\n", " ")
        for col, higher in zip(cols, hib):
            val = best.loc[model, col]
            cell = fmt_val(col, val) if not np.isnan(val) else "—"
            # bold the best value in each column
            col_vals = best[col].dropna()
            best_val = col_vals.max() if higher else col_vals.min()
            if not np.isnan(val) and np.isclose(val, best_val):
                cell = r"\textbf{" + cell + "}"
            row += f" & {cell}"
        row += r" \\"
        print(f"    {row}")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()

    print("── Plain summary ──────────────────────────────────────────")
    header_plain = f"{'Model':<16}" + "".join(f"{l:>12}" for l in labels)
    print(header_plain)
    print("-" * len(header_plain))
    for model in best.index:
        row = f"{MODEL_LABELS.get(model, model).replace(chr(10), ' '):<16}"
        for col, *_ in METRICS:
            val = best.loc[model, col]
            row += f"{fmt_val(col, val):>12}" if not np.isnan(val) else f"{'—':>12}"
        print(row)


# ── util ──────────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="W&B exported CSV file")
    parser.add_argument("--out-dir", default="figures", help="output directory (default: figures/)")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"],
                        help="figure format (default: png)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    print(f"Loaded {len(df)} finished runs from {args.csv!r}")
    if df.empty:
        print("ERROR: no finished runs found in the CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Models present: {sorted(df['model'].unique())}\n")
    best = best_per_model(df)

    print("Generating figures…")
    fig_metrics_bar(best, out, args.fmt)
    fig_radar(best, out, args.fmt)
    fig_scatter(df, out, args.fmt)
    fig_heatmap(best, out, args.fmt)

    print_latex_table(best)


if __name__ == "__main__":
    main()
