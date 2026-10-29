#!/usr/bin/env python3
"""Finite-step departure figure: theory predicts it, the sweep confirms it.

Two panels for the paper's central claim (Section~\\ref{sec:ss-pairs}):

  (a) THEORY.  In the jointly-Gaussian model the denoising risk of the flow's
      readout after integrating to time t is R(t) = (kappa-1)^2 sigma_s^2 +
      kappa^2 sigma_n^2 with the closed-form kappa(t) of Eq.~(kappa). We plot the
      relative risk R(t)/R(0) vs integration time for several SNRs: every curve
      starts at 1.0 at t=0 (the one-step MMSE optimum) and rises monotonically ---
      integrating the flow provably increases error.

  (b) EXPERIMENT.  PSNR gain vs inference-step count (1/4/8/20) from the
      hallucination sweep export, one line per decorrelation radius r. The
      monotone decline matches (a); the dashed line marks the matched-pairing
      one-step regression baseline that one-step flow ties.

  (c) Optional (``--natural-csv``): the same experiment panel on natural images
      (sweep_natural_hallucination export), rendered as a third panel of the same
      figure rather than a second copy of the theory panel.

Panel (a) is pure closed form; panel (b) reads the sweep_hallucination export.
The radius series are drawn with distinct markers/line styles and a small
horizontal offset so that near-identical radii (r=1 vs r=2 on CT, all three on
natural images) stay distinguishable.

    python scripts/figure_steps.py hallucination_export.csv \
        --regression-ref 1.99 --out-dir figures --fmt pdf
    python scripts/figure_steps.py hallucination_export.csv --regression-ref 1.99 \
        --natural-csv natural_hallucination_export.csv --natural-regression-ref 4.09
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
R_COLOR = {1: "#8172B3", 2: "#C44E52", 3: "#937860"}  # r=2 (peak) highlighted
R_MARKER = {1: "s", 2: "o", 3: "^"}
R_LINESTYLE = {1: "--", 2: "-", 3: ":"}
# Multiplicative x-offset per radius (log axis) so overlapping series separate.
R_DODGE = {1: 0.92, 2: 1.0, 3: 1.08}
STEP_COUNTS = [1, 4, 8, 20]


def gamma(t):
    return (1 - t) ** 2 + t ** 2


def kappa(t, snr):
    """Shrinkage coefficient; snr = sigma_s^2 / sigma_n^2 (sigma_n^2 = 1)."""
    return (snr + t) / np.sqrt((snr + gamma(t) * 1.0) * (snr + 1.0))


def risk(k, snr):
    return (k - 1) ** 2 * snr + k ** 2 * 1.0


def panel_theory(ax):
    t = np.linspace(0, 1, 200)
    for snr, ls in [(4.0, "-"), (1.0, "--"), (0.25, ":")]:
        k = kappa(t, snr)
        rr = risk(k, snr) / risk(kappa(0.0, snr), snr)
        ax.plot(t, rr, ls, color="#333", lw=1.8,
                label=fr"SNR $\sigma_s^2/\sigma_n^2={snr:g}$")
    ax.axvline(0, color="#55A868", lw=1)
    # Vertical label along the t=0 line, clear of the legend and the curves.
    ax.text(0.015, 3.0, "one-step optimum ($t=0$)", color="#55A868",
            fontsize=8, rotation=90, ha="left", va="center")
    ax.set_xlabel("integration time $t$")
    ax.set_ylabel(r"relative risk  $R(t)/R(0)$")
    ax.set_title("(a) Theory: iteration provably increases error")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.06, 1.0))


def panel_experiment(ax, csv, reg_ref, title):
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    df = df[df["State"] == "finished"].copy()
    rex = pd.to_numeric(df.get("ssflow_exclude_radius"), errors="coerce")
    base = pd.to_numeric(df["baseline/psnr"], errors="coerce")
    for r in (1, 2, 3):
        sub = df[rex == r]
        if sub.empty:
            continue
        means, errs = [], []
        for k in STEP_COUNTS:
            g = pd.to_numeric(sub[f"val/steps{k}/psnr"], errors="coerce") - base[sub.index]
            means.append(g.mean()); errs.append(g.std())
        xs = [k * R_DODGE[r] for k in STEP_COUNTS]
        ax.errorbar(xs, means, yerr=errs, marker=R_MARKER[r], ls=R_LINESTYLE[r],
                    capsize=3, lw=1.6, ms=5, color=R_COLOR[r], label=fr"flow, $r={r}$")
    if reg_ref is not None:
        ax.axhline(reg_ref, ls="-.", color="#4C72B0", lw=1.2)
        ax.text(STEP_COUNTS[-1] * 1.1, reg_ref, "one-step regression", color="#4C72B0",
                fontsize=8, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_xticks(STEP_COUNTS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("inference steps (Euler)")
    ax.set_ylabel(r"PSNR gain over input (dB)")
    ax.set_title(title)
    # Curves fall from upper-left to lower-right, so lower-left is empty.
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="W&B export of sweep_hallucination.yml")
    ap.add_argument("--regression-ref", type=float, default=None,
                    help="one-step regression baseline (dB) to draw as reference, e.g. 1.99")
    ap.add_argument("--natural-csv", default=None,
                    help="W&B export of sweep_natural_hallucination.yml; adds panel (c)")
    ap.add_argument("--natural-regression-ref", type=float, default=None,
                    help="natural-image one-step regression baseline (dB), e.g. 4.09")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--fmt", default="pdf", choices=["png", "pdf", "svg"])
    a = ap.parse_args()

    if a.natural_csv:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.0))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    panel_theory(ax1)
    panel_experiment(ax2, a.csv, a.regression_ref,
                     "(b) CT: PSNR falls monotonically with steps")
    if a.natural_csv:
        panel_experiment(ax3, a.natural_csv, a.natural_regression_ref,
                         "(c) Natural images: same decline")
    fig.tight_layout()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"steps_departure.{a.fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
