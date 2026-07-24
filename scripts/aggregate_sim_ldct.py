#!/usr/bin/env python3
"""Aggregate the simulated-LDCT robustness sweep (sweep_sim_ldct.yml).

Reads a wandb CSV export of the ``ctdenoiser-sweep`` runs and emits (a) the
tidy per-cell results table ``results/sim_ldct_abdomen.csv`` and (b) the LaTeX
body rows for ``tab:sim-ldct`` in ``paper/main.tex``.  Re-run it once the sweep
finishes (fresh export) to fill the provisional CTformer seeds and the CFM row.

Usage:
    python scripts/aggregate_sim_ldct.py <wandb_export.csv> [--tex]
"""
import argparse
import csv
import statistics as st
import sys

# supervision block order and the model order within each block
MODELS = ["redcnn", "dncnn", "unet", "ctformer", "flowmatching"]
MODES = ["supervised", "n2sim", "n2v"]
PARAMS = {  # for the LaTeX Params column
    "redcnn": r"$1.85\mathrm{M}$",
    "dncnn": r"$0.56\mathrm{M}$",
    "unet": r"$1.95\mathrm{M}$",
    "ctformer": r"$0.40\mathrm{M}$",
    "flowmatching": r"$2.11\mathrm{M}$",
}
CITE = {
    "redcnn": r"RED-CNN~\cite{chen2017lowdose}",
    "dncnn": r"DnCNN~\cite{zhang2017beyond}",
    "unet": "U-Net",
    "ctformer": r"CTformer~\cite{wang2022ctformer}",
    "flowmatching": r"CFM~\cite{lipman2022flow}",
}
SUP = {
    "supervised": "Supervised",
    "n2sim": r"Noise2Sim~\cite{niu2020noise2sim}",
    "n2v": r"Noise2Void~\cite{krull2019noise2void}",
}
# wandb summary keys: best-epoch absolutes + the logged gain
K = dict(psnr="val/psnr.max", ssim="val/ssim.max", rmse="val/rmse.min",
         gmsd="val/gmsd.min", rsr="val/nps_ratio.min", gain="val/psnr_gain")


def mode_of(row):
    return row.get("training-mode") or row.get("training_mode") or ""


def fnum(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", "NaN", None) else None


def collect(rows, model, mode, key):
    return [v for r in rows if r.get("model") == model and mode_of(r) == mode
            and (v := fnum(r, key)) is not None]


def cell(rows, model, mode):
    g = collect(rows, model, mode, K["gain"])
    if not g:
        return None
    m = lambda k: st.mean(collect(rows, model, mode, K[k]))  # noqa: E731
    return dict(n=len(g), psnr=m("psnr"), dpsnr=st.mean(g),
                sd=st.pstdev(g) if len(g) > 1 else 0.0,
                ssim=m("ssim"), rmse=m("rmse"), gmsd=m("gmsd"), rsr=m("rsr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="wandb CSV export of the sim-ldct sweep")
    ap.add_argument("--tex", action="store_true", help="also print LaTeX rows")
    ap.add_argument("--out", default="results/sim_ldct_abdomen.csv")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.export)) if r["State"] == "finished"]
    if not rows:
        sys.exit("no finished runs in export")

    floor = dict(
        psnr=st.mean([fnum(r, "baseline/psnr") for r in rows if fnum(r, "baseline/psnr")]),
        ssim=st.mean([fnum(r, "baseline/ssim") for r in rows if fnum(r, "baseline/ssim")]),
        rmse=st.mean([fnum(r, "baseline/rmse") for r in rows if fnum(r, "baseline/rmse")]),
        gmsd=st.mean([fnum(r, "baseline/gmsd") for r in rows if fnum(r, "baseline/gmsd")]),
        rsr=st.mean([fnum(r, "baseline/nps_ratio") for r in rows if fnum(r, "baseline/nps_ratio")]),
    )

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "supervision", "n_seeds", "psnr", "dpsnr",
                    "dpsnr_sd", "ssim", "rmse", "gmsd", "rsr"])
        w.writerow(["input_floor", "none", "", round(floor["psnr"], 2), 0, 0,
                    round(floor["ssim"], 3), round(floor["rmse"], 4),
                    round(floor["gmsd"], 3), round(floor["rsr"], 4)])
        for model in MODELS:
            for mode in MODES:
                c = cell(rows, model, mode)
                if not c:
                    continue
                w.writerow([model, mode, c["n"], round(c["psnr"], 2),
                            round(c["dpsnr"], 2), round(c["sd"], 2),
                            round(c["ssim"], 3), round(c["rmse"], 4),
                            round(c["gmsd"], 3), round(c["rsr"], 4)])
    print(f"wrote {args.out}")

    if args.tex:
        print("\n% --- tab:sim-ldct body rows (paste into paper/main.tex) ---")
        for mode in MODES:
            cells = [(model, cell(rows, model, mode)) for model in MODELS]
            cells = [(model, c) for model, c in cells if c]
            best = max((c["dpsnr"] for _, c in cells), default=None)
            for model, c in cells:
                prov = c["n"] < 3  # provisional if not all seeds in
                sign = "+" if c["dpsnr"] >= 0 else ""
                gain = (f"{sign}{c['dpsnr']:.2f}" if prov
                        else f"{sign}{c['dpsnr']:.2f}{{\\pm}}{c['sd']:.2f}")
                psnr = f"{c['psnr']:.2f} \\,(${gain}$)"
                if c["dpsnr"] == best:
                    psnr = r"\textbf{" + psnr + "}"
                name = CITE[model] + ("$^{\\dagger}$" if prov else "")
                print(f"    {name:44} & {SUP[mode]:38} & {PARAMS[model]} & "
                      f"{psnr} & ${c['ssim']:.3f}$ & ${c['rmse']:.4f}$ & "
                      f"${c['gmsd']:.3f}$ & ${c['rsr']:.4f}$ \\\\")
            print(r"    \midrule")


if __name__ == "__main__":
    main()
