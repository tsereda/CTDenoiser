# CTDenoiser

Low-dose CT image denoising benchmark. Includes a **CTformer** transformer
architecture (Token2Token Dilation blocks with cyclic shifts) and a
**RED-CNN** convolutional baseline, plus training, evaluation, and
overlapped full-slice inference utilities.

## Layout

```
ctdenoiser/
  models/
    ctformer.py        # CTformer (T2TD / IT2TD + Transformer blocks)
    redcnn.py          # RED-CNN baseline
  data/
    dataset.py         # DICOM / HDF5 CT + NaturalImage + SimulatedLowDoseCT + Synthetic
    dicom.py           # DICOM series reader (HU conversion)
    noise.py           # synthetic-noise regimes (i.i.d. / correlated) + LDCT simulation
  selfsupervised.py    # Noise2Void blind-spot + Noise2Sim similarity self-supervision
  zeroshot.py          # Zero-Shot Noise2Noise + Filter2Noise (per-image, data-free)
  metrics.py           # PSNR / SSIM / RMSE / GMSD / NPS-ratio
  inference.py         # overlapped full-slice inference
  train.py             # training loop / CLI
tests/
  test_models.py          # forward-pass shape sanity checks
  test_selfsupervised.py  # N2V + ZS-N2N checks
docs/
  future.md          # long-term research roadmap
```

## Install

```bash
pip install -e .
```

## Quick start

```bash
# Smoke test (synthetic data)
python -m ctdenoiser.train --model ctformer --epochs 1

# Train on DICOM data (patient-split, full-slice overlapped-inference eval)
python -m ctdenoiser.train --model ctformer \
    --dicom-root /data/ldct_dicom --epochs 50 --batch-size 16

# Run tests
pytest -q
```

## Self-supervised / zero-shot training

By default training is **supervised** (MSE / flow loss against the full-dose
reference). Four methods train *without* a clean target, using only the noisy
low-dose image — useful when paired LDCT/NDCT data are scarce. All are still
evaluated against the clean `full` reference, so their PSNR / SSIM / RMSE / GMSD /
NPS-ratio are directly comparable to the supervised models.

Two are **dataset** self-supervised (`n2v`, `n2sim`) — they reuse any
conv/transformer model and produce a normal checkpoint. Two are **per-image
zero-shot** (`zsn2n`, `f2n`) — they ignore `--model` and train a fresh tiny
network on each slice alone, saving no checkpoint.

```bash
# Noise2Void: blind-spot self-supervision (reuses any conv/transformer model)
python -m ctdenoiser.train --model redcnn --training-mode n2v \
    --dicom-root /data/ldct_dicom --epochs 50 --n2v-mask-fraction 0.02

# Noise2Sim: similarity-based self-supervision (reuses any conv/transformer model)
python -m ctdenoiser.train --model redcnn --training-mode n2sim \
    --dicom-root /data/ldct_dicom --epochs 50 --n2sim-search-radius 4

# Zero-Shot Noise2Noise: a fresh tiny network is trained per image, no training data
python -m ctdenoiser.train --training-mode zsn2n \
    --dicom-root /data/ldct_dicom --zsn2n-iters 2000 --zsn2n-channels 48

# Filter2Noise: per-image attention-guided bilateral filtering, no training data
python -m ctdenoiser.train --training-mode f2n \
    --dicom-root /data/ldct_dicom --f2n-iters 1500 --f2n-layers 2

# Smoke tests on synthetic data
python -m ctdenoiser.train --model unet --training-mode n2v --epochs 1
python -m ctdenoiser.train --model unet --training-mode n2sim --epochs 1
python -m ctdenoiser.train --training-mode zsn2n --synthetic-len 4 --zsn2n-iters 100
python -m ctdenoiser.train --training-mode f2n --synthetic-len 4 --f2n-iters 100
```

- `n2v` masks a fraction of pixels (`--n2v-mask-fraction`), replaces each with a
  random neighbour (`--n2v-neighbor-radius`), and trains the model to predict the
  original value at those blind spots. Produces a normal checkpoint.
- `n2sim` builds a per-pixel target by searching a `--n2sim-search-radius` window
  for the `--n2sim-num-similar` most-similar pixels (matched over a
  `--n2sim-patch-radius` patch) and regresses the noisy image onto their average.
  Exploits non-local self-similarity instead of a blind spot, so the model sees
  the un-masked input. Produces a normal checkpoint.
- `flowmatching` is **not** compatible with `n2v` / `n2sim` (it needs paired targets).
- `zsn2n` ignores `--model`: for each slice it trains a small 2-layer network from
  scratch on that image alone (`--zsn2n-iters`, `--zsn2n-channels`, `--zsn2n-lr`)
  and denoises it. No checkpoint is saved (per-image networks are discarded).
- `f2n` (Filter2Noise) also ignores `--model`: per slice it trains a stack of
  `--f2n-layers` *interpretable* attention-guided bilateral filters
  (`--f2n-radius`, `--f2n-channels`, `--f2n-iters`, `--f2n-lr`) using the same
  self-supervised pair-downsampler loss as `zsn2n`. No checkpoint is saved.

These are the self-supervised rows of the benchmark; see [`docs/future.md`](docs/future.md)
for how they fit the long-term research plan.

## Data

### DICOM series directories

Point `--dicom-root` at a directory of TCIA LDCT SeriesInstanceUID
subdirectories, each containing `.dcm` files. Low/full dose is detected
automatically from the DICOM `SeriesDescription` header and paired by
`PatientID`.

```
ldct_dicom/
  1.2.840.113713.4.100.1.2.xxx/   # Low Dose Images
    1-001.dcm  1-002.dcm  ...
  1.2.840.113713.4.100.1.2.yyy/   # Full Dose Images
    1-001.dcm  1-002.dcm  ...
  ...
```

- Split is **by patient** (`--val-fraction`, `--seed`) to prevent leakage.
- Training uses random `--patch-size` crops.
- Validation runs **full-slice overlapped inference** (margin = patch/4)
  and reports PSNR / SSIM / RMSE / GMSD / NPS-ratio (mean ± per-slice std),
  plus inference latency, peak GPU memory, parameter count, and Δ over the
  identity baseline.
- HU → `[0,1]` via a per-anatomy window (below).

### Per-anatomy HU windows

The `LDCT-and-Projection-data` collection spans abdomen, chest, and head scans,
and each needs its own window — chest lung tissue (~-800 HU) clips to black
under the abdomen soft-tissue window. Pick the window with `--anatomy`:

```bash
python -m ctdenoiser.train --model ctformer --dicom-root /data/ldct_dicom \
    --anatomy chest     # abdomen (default) | chest | head
```

| `--anatomy` | clinical window | `[low, high]` HU |
|-------------|-----------------|------------------|
| `abdomen`   | soft tissue (L40/W400)  | `[-160, +240]` |
| `chest`     | lung (L-600/W1500)      | `[-1350, +150]` |
| `head`      | brain (L40/W80)         | `[0, +80]` |

`--hu-offset` / `--hu-scale` override the preset for a custom window. Keep one
anatomy per `--dicom-root` / HDF5 cache so a patient never maps two windows;
the data layer now **errors** if a patient has two same-dose series rather than
silently mis-pairing.

### Per-anatomy benchmark workflow

The data pod prepares each anatomy cache in a single run: for each anatomy it
downloads the cohort and writes a self-describing cache `/data/ldct_<anatomy>.h5`
(window baked in, anatomy stored in the file attrs), validating each before
moving on. Run the pod **once** and the caches are ready; sweep any of them
anytime — caches coexist on the PVC:

```bash
# 1. one data-pod run -> ldct_abdomen.h5, ldct_chest.h5 (+ /data/natural_images)
#    Idempotent: a valid cache is skipped on re-run, and a missing/corrupt one
#    is re-converted (so an interrupted earlier run self-heals). Override the
#    set with ANATOMIES, e.g. ANATOMIES="abdomen" for a subset.
kubectl apply -f k8s/data_pod.yml -n usd-djha

# 2. one sweep per anatomy (picks /data/ldct_<anatomy>.h5)
python sweep.py sweep.yml --anatomy abdomen --agents 8
python sweep.py sweep.yml --anatomy chest   --agents 8

# 3. one merged dashboard (anatomy column carries through)
python scripts/benchmark_report.py abdomen.csv chest.csv --out report.html
```

> **`head` is not in the default `ANATOMIES`.** The `LDCT-and-Projection-data`
> query for `BodyPartExamined=HEAD` returns 0 paired Low/Full-dose patients, so
> it never produced a cache. Add it back (`ANATOMIES="abdomen chest head"`) to
> re-check — the download step now prints the collection's real body-part labels
> when a cohort comes back empty, so a label mismatch is distinguishable from a
> genuine absence.

`--h5-name` targets an explicit cache filename (e.g. a legacy
`ldct_preprocessed.h5`). All runs can share one W&B project — each carries its
anatomy/window in the logged config, and `benchmark_report.py` facets by it.

### Cross-domain sweeps (natural images + simulated LDCT)

The same data-pod run also prepares the two cross-domain datasets: it fetches
**BSDS500** to `/data/natural_images/` (for `--natural`), and the simulated LDCT
arm (`--sim-ldct`) reuses the abdomen full-dose cache — so no extra download,
just keep `abdomen` in `ANATOMIES`. Both sweeps read their inputs directly from
the read-only `/data` mount, so they use the **lean** job template
`k8s/tr_job_template_synth.yml` (no h5 copy / npy unpack) via `--template`:

```bash
# generalise the theorem beyond CT (i.i.d. + correlated noise, both regimes)
python sweep.py sweep_natural.yml \
    --template k8s/tr_job_template_synth.yml --agents 3      # 90 runs (full grid)

# the decisive one-step-flow-ties-regression pair, off CT data (gaussian)
python sweep.py sweep_natural_flow_vs_reg.yml   --template k8s/tr_job_template_synth.yml --agents 3
python sweep.py sweep_natural_hallucination.yml --template k8s/tr_job_template_synth.yml --agents 3

# second, independent LDCT source (simulated low-dose from full-dose abdomen)
python sweep.py sweep_sim_ldct.yml --template k8s/tr_job_template_synth.yml --agents 3
```

`--anatomy` / `--h5-name` are ignored for the lean template (it reads `/data`
directly). Export each sweep's CSV into `results/` and merge with
`benchmark_report.py` alongside the CT runs.

### Synthetic (smoke test)

Without `--dicom-root`, a synthetic noisy/clean dataset is generated so the
pipeline runs end to end.

### Natural-image denoising (beyond CT)

The one-step-optimal / multi-step-hurts result is **not CT-specific**, so the
`--natural` dataset stresses it on general images: clean images plus synthetic
noise, under two regimes and two pairings. It keeps the same `(low, full)`
interface, so every model and training mode works unchanged.

```bash
# i.i.d. Gaussian noise, clean target (supervised)
python -m ctdenoiser.train --model dncnn --natural --natural-root /data/bsd \
    --noise-std 0.1 --noise-mode gaussian --epochs 50

# spatially-correlated noise, Noise2Noise pairing (two independent noisy views)
python -m ctdenoiser.train --model dncnn --natural --natural-root /data/bsd \
    --noise-mode correlated --correlation-sigma 1.5 --pair-mode noisy --epochs 50
```

`--natural-root` is a directory of clean grayscale images (any PIL-readable
files). `--natural` **requires** either `--natural-root` or, to deliberately run
on a deterministic **procedural** image set (tests / smoke runs, no download),
the explicit `--natural-procedural` — there is no silent fallback, so a mistyped
or missing root fails loudly instead of quietly training on synthetic data:

```bash
python -m ctdenoiser.train --model dncnn --natural --natural-procedural --epochs 1
```

| flag | meaning |
| --- | --- |
| `--noise-mode {gaussian,correlated}` | i.i.d. white vs. blurred (FBP-like) noise |
| `--noise-std` | noise level in `[0,1]` units |
| `--correlation-sigma` | Gaussian blur σ for correlated noise |
| `--pair-mode {clean,noisy}` | clean target (supervised) vs. second noisy view (N2N) |

### Simulated low-dose CT (a second, independent LDCT source)

`--sim-ldct` manufactures the low-dose arm from full-dose CT with a
physically-motivated noise model (correlated by an FBP-like blur, and
signal-dependent so denser tissue is noisier). This gives a low/full source
independent of any single real acquisition — the "does the clinical result hold
on other low-dose data?" robustness axis — without external DICOM.

```bash
# simulate low-dose from an existing full-dose HDF5 cache
python -m ctdenoiser.train --model redcnn --sim-ldct \
    --sim-source /data/ldct_abdomen.h5 --sim-base-std 0.03 --sim-signal-std 0.06 \
    --epochs 50

# procedural CT phantoms (self-contained smoke run)
python -m ctdenoiser.train --model redcnn --sim-ldct --sim-procedural --sim-patients 8 --epochs 1
```

`--sim-ldct` **requires** either `--sim-source` (a
`scripts/convert_dicom_to_h5.py` cache, read from its `/patients/{id}/full`
volumes) or the explicit `--sim-procedural` for phantoms — no silent fallback.
`--pair-mode noisy` yields two independent simulated low-dose views
(Noise2Noise) instead of the clean target.

## Benchmark report

Pass `--wandb-project` to log per-epoch metrics, sample image panels, parameter
count / model size, inference latency, peak GPU memory, dataset provenance (the
exact val patient IDs, anatomy, HU window), and git/torch versions to W&B.

Instead of clicking through the W&B UI, build a **self-contained HTML
dashboard** — a sortable leaderboard, PSNR-vs-params and PSNR-vs-latency Pareto
frontiers, a per-anatomy breakdown, and an optional sample gallery — all in one
file you open in a browser:

```bash
# from a W&B CSV export
python scripts/benchmark_report.py export.csv --out report.html

# straight from the W&B API (no manual export)
python scripts/benchmark_report.py --wandb timgsereda/ctdenoiser-sweep
python scripts/benchmark_report.py --wandb timgsereda/ctdenoiser-sweep/sweeps/ID

# embed denoised sample panels alongside the tables
python scripts/benchmark_report.py export.csv --images figures/samples
```

`scripts/analyze_sweep.py` and `scripts/sweep_report.py` still produce the
static publication figures / LaTeX table and the by-training-mode summary.
