#!/usr/bin/env python3
"""Build a self-contained HTML benchmark dashboard from CTDenoiser runs.

One ``report.html`` you open in a browser -- no W&B login, no live UI:

  * a **sortable leaderboard** (quality metrics + params + latency + Δ vs the
    identity baseline), click any header to sort;
  * **Pareto efficiency frontiers** -- PSNR vs parameter count and PSNR vs
    inference latency, with the non-dominated models highlighted (the figure a
    denoising paper actually wants: which models are worth their cost);
  * a **per-anatomy** breakdown when runs span more than one HU window;
  * an optional **sample gallery** (``--images DIR`` of PNG panels) embedded
    inline.

Data comes from either W&B (``--wandb ENTITY/PROJECT`` or a sweep) or one or
more CSV exports -- whichever is reachable.

Usage
-----
    python scripts/benchmark_report.py export.csv
    python scripts/benchmark_report.py a.csv b.csv --out report.html
    python scripts/benchmark_report.py --wandb timgsereda/ctdenoiser-sweep
    python scripts/benchmark_report.py export.csv --images figures/samples
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Display config: (column, label, higher_is_better, fmt)
QUALITY = [
    ("val/psnr", "PSNR (dB)", True, "{:.2f}"),
    ("val/ssim", "SSIM", True, "{:.4f}"),
    ("val/rmse", "RMSE", False, "{:.4f}"),
    ("val/gmsd", "GMSD", False, "{:.4f}"),
    ("val/nps_ratio", "NPS", False, "{:.4f}"),
]
COST = [
    ("param_count", "Params", "{:,.0f}"),
    ("model_size_mb", "Size (MB)", "{:.2f}"),
    ("val/latency_ms", "Latency (ms)", "{:.1f}"),
]
MODE_ORDER = ["supervised", "n2v", "zsn2n"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_csv(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_wandb(spec: str) -> pd.DataFrame:
    """Pull runs from ``entity/project`` or ``entity/project/sweeps/ID``."""
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed; `pip install wandb` or pass a CSV instead.")

    api = wandb.Api()
    parts = spec.strip("/").split("/")
    if "sweeps" in parts:
        runs = api.sweep(spec).runs
    elif len(parts) == 2:
        runs = api.runs(spec)
    else:
        sys.exit(f"--wandb expects ENTITY/PROJECT or .../sweeps/ID, got {spec!r}")

    rows = []
    for r in runs:
        row = {"State": r.state}
        row.update({k: v for k, v in r.config.items() if not k.startswith("_")})
        row.update(dict(r.summary))  # val/*, baseline/*, gains
        rows.append(row)
    return pd.DataFrame(rows)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns the way both report sections expect."""
    if "State" in df.columns:
        df = df[df["State"].astype(str).str.strip() == "finished"].copy()

    # training mode lives under inconsistent headers across exports
    mode = None
    for col in ("training_mode", "training-mode"):
        if col in df.columns:
            s = df[col].astype("string").str.strip().replace("", pd.NA)
            mode = s if mode is None else mode.fillna(s)
    df["mode"] = (mode if mode is not None else pd.Series(pd.NA, index=df.index)).fillna(
        "supervised"
    ).astype(str)

    df["model"] = df.get("model", "?").astype("string").str.strip().str.lower().astype(str)
    if "anatomy" in df.columns:
        df["anatomy"] = df["anatomy"].astype("string").fillna("?").astype(str)
    else:
        df["anatomy"] = "?"

    numeric = [c for c, *_ in QUALITY] + [c for c, *_ in COST] + ["baseline/psnr"]
    for c in numeric:
        # Some exports omit cost columns (param_count, latency, …); create them
        # as all-NaN so downstream code can treat "missing" and "empty" alike.
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df = df[df.get("val/psnr").notna()].copy() if "val/psnr" in df.columns else df
    return df


def best_runs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, mode, anatomy): the highest-PSNR run."""
    keys = ["model", "mode", "anatomy"]
    return (
        df.sort_values("val/psnr", ascending=False)
        .drop_duplicates(subset=keys)
        .reset_index(drop=True)
    )


# ── pareto frontier ───────────────────────────────────────────────────────────

def pareto_mask(cost, quality):
    """Boolean mask of non-dominated points (minimise cost, maximise quality)."""
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    n = len(cost)
    keep = np.ones(n, bool)
    for i in range(n):
        if np.isnan(cost[i]) or np.isnan(quality[i]):
            keep[i] = False
            continue
        for j in range(n):
            if i == j or np.isnan(cost[j]) or np.isnan(quality[j]):
                continue
            if cost[j] <= cost[i] and quality[j] >= quality[i] and (
                cost[j] < cost[i] or quality[j] > quality[i]
            ):
                keep[i] = False
                break
    return keep


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def pareto_png(best: pd.DataFrame, cost_col: str, cost_label: str) -> str | None:
    if cost_col not in best.columns:
        return None
    sub = best.dropna(subset=[cost_col, "val/psnr"])
    if sub.empty:
        return None
    cost = sub[cost_col].to_numpy()
    qual = sub["val/psnr"].to_numpy()
    mask = pareto_mask(cost, qual)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(cost[~mask], qual[~mask], c="#bbbbbb", s=55, zorder=2, label="dominated")
    ax.scatter(cost[mask], qual[mask], c="#C44E52", s=90, zorder=3,
               edgecolors="black", linewidths=0.6, label="Pareto-optimal")
    # frontier line through the optimal points, sorted by cost
    front = sub[mask].sort_values(cost_col)
    ax.plot(front[cost_col], front["val/psnr"], color="#C44E52", lw=1.3, zorder=2)
    for _, r in sub.iterrows():
        ax.annotate(f"{r['model']}/{r['mode']}", (r[cost_col], r["val/psnr"]),
                    fontsize=6.5, color="#333", xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel(f"{cost_label}  (lower = cheaper)", fontsize=9)
    ax.set_ylabel("PSNR (dB)  (higher = better)", fontsize=9)
    ax.set_title(f"Efficiency frontier: PSNR vs {cost_label}", fontsize=10)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(fontsize=8, frameon=False)
    return _fig_to_b64(fig)


def anatomy_png(best: pd.DataFrame) -> str | None:
    anats = sorted(best["anatomy"].unique())
    if len(anats) < 2:
        return None
    models = sorted(best["model"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(anats)
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.1), 4))
    for i, a in enumerate(anats):
        vals = [best.loc[(best["model"] == m) & (best["anatomy"] == a), "val/psnr"].max()
                for m in models]
        ax.bar(x + i * width, vals, width, label=a)
    ax.set_xticks(x + width * (len(anats) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Best PSNR by model & anatomy")
    ax.legend(title="anatomy", fontsize=8, frameon=False)
    return _fig_to_b64(fig)


# ── html assembly ─────────────────────────────────────────────────────────────

def _cell(val, fmt):
    return "—" if pd.isna(val) else fmt.format(val)


def leaderboard_html(best: pd.DataFrame) -> str:
    best = best.assign(
        _o=best["mode"].map(lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 99)
    ).sort_values(["val/psnr"], ascending=False)

    multi_anat = best["anatomy"].nunique() > 1
    head = ["Model", "Mode"]
    if multi_anat:
        head.append("Anatomy")
    head += [lbl for _, lbl, *_ in QUALITY] + [lbl for _, lbl, _ in COST] + ["ΔPSNR"]

    ths = "".join(
        f'<th onclick="sortTable({i})">{h}<span class="arrow"></span></th>'
        for i, h in enumerate(head)
    )
    rows = []
    for _, r in best.iterrows():
        cells = [f"<td>{r['model']}</td>", f"<td>{r['mode']}</td>"]
        if multi_anat:
            cells.append(f"<td>{r['anatomy']}</td>")
        for col, _lbl, _hib, fmt in QUALITY:
            txt = _cell(r.get(col), fmt)
            std = r.get(f"{col}_std")
            if not pd.isna(std):
                txt += f' <span class="pm">±{std:.3g}</span>'
            cells.append(f'<td data-v="{r.get(col, "nan")}">{txt}</td>')
        for col, _lbl, fmt in COST:
            cells.append(f'<td data-v="{r.get(col, "nan")}">{_cell(r.get(col), fmt)}</td>')
        gain = (r.get("val/psnr") - r["baseline/psnr"]) if "baseline/psnr" in r and not pd.isna(
            r.get("baseline/psnr")) else np.nan
        cls = "pos" if (not pd.isna(gain) and gain > 0) else "neg"
        cells.append(f'<td data-v="{gain}" class="{cls}">{_cell(gain, "{:+.2f}")}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f'<table id="lb"><thead><tr>{ths}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def gallery_html(images_dir) -> str:
    if not images_dir:
        return ""
    paths = sorted(Path(images_dir).glob("*.png"))
    if not paths:
        return ""
    imgs = []
    for p in paths[:24]:
        b64 = base64.b64encode(p.read_bytes()).decode()
        imgs.append(f'<figure><img src="data:image/png;base64,{b64}">'
                     f'<figcaption>{p.name}</figcaption></figure>')
    return f'<section><h2>Sample gallery</h2><div class="gallery">{"".join(imgs)}</div></section>'


def img_section(title, b64):
    if not b64:
        return ""
    return f'<section><h2>{title}</h2><img class="plot" src="data:image/png;base64,{b64}"></section>'


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>CTDenoiser benchmark</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1000px;color:#222;padding:0 1rem}}
 h1{{margin-bottom:.2rem}} .sub{{color:#777;margin-top:0;font-size:.9rem}}
 section{{margin:2.2rem 0}}
 table{{border-collapse:collapse;width:100%;font-size:.85rem}}
 th,td{{padding:.45rem .6rem;text-align:right;border-bottom:1px solid #eee}}
 th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
 th{{cursor:pointer;user-select:none;background:#fafafa;position:sticky;top:0;border-bottom:2px solid #ddd}}
 th:hover{{background:#f0f0f0}} tbody tr:hover{{background:#fafcff}}
 .pm{{color:#999;font-size:.78em}} .pos{{color:#1a8a3b}} .neg{{color:#c0392b}}
 .arrow{{font-size:.7em;color:#999;margin-left:.2em}}
 img.plot{{max-width:560px;width:100%;border:1px solid #eee;border-radius:6px}}
 .gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:.8rem}}
 figure{{margin:0}} figure img{{width:100%;border:1px solid #eee;border-radius:4px}}
 figcaption{{font-size:.75rem;color:#888;text-align:center}}
</style></head><body>
<h1>CTDenoiser benchmark</h1>
<p class="sub">{n} runs · {nmodels} models · modes: {modes} · anatomies: {anats}{baseline}</p>
<section><h2>Leaderboard <span class="sub">(click a header to sort)</span></h2>{leaderboard}</section>
{pareto_params}{pareto_latency}{anatomy}{gallery}
<script>
function sortTable(col){{
 const t=document.getElementById('lb'),rows=[...t.tBodies[0].rows];
 const cur=t.dataset.sortCol==col?-(t.dataset.sortDir||1):1;
 t.dataset.sortCol=col;t.dataset.sortDir=cur;
 const num=c=>{{const d=c.dataset.v;return d!==undefined?parseFloat(d):c.innerText;}};
 rows.sort((a,b)=>{{let x=num(a.cells[col]),y=num(b.cells[col]);
   if(typeof x==='number'&&typeof y==='number'){{if(isNaN(x))return 1;if(isNaN(y))return -1;return (x-y)*cur;}}
   return (''+x).localeCompare(''+y)*cur;}});
 rows.forEach(r=>t.tBodies[0].appendChild(r));
 [...t.tHead.rows[0].cells].forEach((th,i)=>th.querySelector('.arrow').textContent=i==col?(cur>0?'▲':'▼'):'');
}}
</script></body></html>"""


def build_html(df: pd.DataFrame, images_dir) -> str:
    best = best_runs(df)
    base = df["baseline/psnr"].mean() if "baseline/psnr" in df.columns else np.nan
    baseline_txt = f" · identity baseline {base:.2f} dB PSNR" if not pd.isna(base) else ""
    return HTML.format(
        n=len(df),
        nmodels=df["model"].nunique(),
        modes=", ".join(sorted(df["mode"].unique())),
        anats=", ".join(sorted(df["anatomy"].unique())),
        baseline=baseline_txt,
        leaderboard=leaderboard_html(best),
        pareto_params=img_section("Efficiency: PSNR vs parameters",
                                  pareto_png(best, "param_count", "parameters")),
        pareto_latency=img_section("Efficiency: PSNR vs latency",
                                   pareto_png(best, "val/latency_ms", "latency (ms/slice)")),
        anatomy=img_section("Per-anatomy comparison", anatomy_png(best)),
        gallery=gallery_html(images_dir),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="*", help="W&B CSV export(s)")
    ap.add_argument("--wandb", metavar="ENTITY/PROJECT[/sweeps/ID]",
                    help="pull runs from the W&B API instead of a CSV")
    ap.add_argument("--out", default="report.html", help="output HTML (default: report.html)")
    ap.add_argument("--images", help="directory of sample PNGs to embed as a gallery")
    args = ap.parse_args()

    if args.wandb:
        df = tidy(load_wandb(args.wandb))
    elif args.csv:
        df = tidy(load_csv(args.csv))
    else:
        ap.error("pass a CSV export or --wandb ENTITY/PROJECT")

    if df.empty:
        sys.exit("ERROR: no finished runs with val/psnr found.")

    Path(args.out).write_text(build_html(df, args.images))
    print(f"Loaded {len(df)} runs → wrote {args.out}")
    print(f"  open it: file://{Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
