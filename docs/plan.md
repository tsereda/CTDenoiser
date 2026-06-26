# Implementation Plan — Phase 1: Hallucination-Aware Evaluation

This operationalizes Phase 1 of [`future.md`](future.md): the task-based,
hallucination-aware evaluation methodology that is the paper's actual
contribution. Today the benchmark reports only pixel metrics (PSNR / SSIM /
RMSE / GMSD) plus a mislabeled `nps_ratio`; nothing measures whether a known
low-contrast signal is **preserved, erased, or fabricated**. This plan adds
that.

## Target venue

Not AAAI (wrong fit; the work is an applications/evaluation contribution, and
the SSFlow ablation's null multi-step result is a clarifying finding, not a
methods-novelty result). The natural homes are **IEEE TMI / MELBA / Medical
Physics** (journal, best fit for an evaluation methodology) or **WACV
Applications**. A workshop paper with multi-seed + finished runs is the
near-term stake in the ground. This evaluation methodology is the work that
moves the paper out of the "another CT-denoising benchmark" reject pile.

## Key design insight — no CT projector needed

Re-simulating a low-dose pair normally means a forward projector (sinogram →
quantum noise → FBP): a heavy dependency (ASTRA / torch-radon) and a
calibration headache. We avoid it.

The dataset has **real paired low/full acquisitions**, so the real,
spatially-correlated FBP noise is just `n = low − full`. To make a
signal-present low-dose image, add a known lesion `s` to the clean image and
re-attach that same real noise:

```
low_present = full + s + n = (low − full) + full + s = low + s
low_absent  = low                      # the real acquisition, untouched
```

So **the signal-present image is literally `low + s`** — real correlated CT
noise, with a lesion of known location / size / contrast. No projector, no
noise model to calibrate, and it exercises the exact noise the thesis is about.
(A physics simulator only becomes necessary for *cross-dose* generalization in
Phase 2.)

Caveat to document: this assumes the small inserted signal does not change the
noise field, the standard low-contrast signal-insertion assumption in CT
detectability studies. Keep contrast low enough that noise dominates.

## Components

### 1. Signal insertion — `ctdenoiser/detectability.py` (new)

```python
def insert_signal(clean, center, radius_px, contrast_hu, hu_scale, profile="disk"):
    """Add a low-contrast (optionally Gaussian-profiled) disk to a [0,1] slice.
    Normalized amplitude = contrast_hu / hu_scale (e.g. 10 HU / 400 = 0.025)."""
```

- Specify contrast in **HU**, then normalize by the anatomy's `hu_scale`.
- Sample lesion locations inside the body mask, on flat background, at known
  ground-truth coordinates. Sweep a small set of (radius, contrast) operating
  points for a detectability-vs-contrast curve.

### 2. Corrected NPS — extend `ctdenoiser/metrics.py`

```python
def uniform_nps(image, roi_size=64, n_rois=...):
    """Detrended 2-D NPS ensemble-averaged over flat ROIs.
    Returns the radial NPS profile, peak frequency, and total noise power."""
```

- Auto-select flat ROIs by local-variance / gradient threshold inside the body
  mask.
- Report the **NPS shape** of the denoised output vs the input. A denoiser that
  shifts the NPS peak toward low frequency is producing the blotchy / "waxy"
  texture radiologists distrust — invisible to PSNR/SSIM, and exactly the
  "metrics hide it" point.
- Rename the existing `nps_ratio` → `residual_spectrum` (honest name); keep it
  for backward-compatible logging but stop calling it NPS.

### 3. CHO detectability — `ctdenoiser/detectability.py`

Channelized Hotelling Observer for the SKE/BKS task (signal known exactly,
background known statistically):

```python
def cho_detectability(present_rois, absent_rois, signal, n_channels=10):
    """Laguerre-Gauss channels U; Hotelling template w = K^{-1}(v̄_p − v̄_a);
    returns d' and AUC."""
```

- Build present/absent ROI ensembles (`low+s` vs `low`), run both through the
  existing `overlapped_inference` (model-agnostic — reuse as-is), extract ROIs
  at the known locations, channelize, estimate covariance, compute `d'`.
- Report `d'` for **input / denoised / clean** (clean = ceiling).

Headline numbers this yields:
- **Detectability preserved?** `d'_denoised / d'_input`.
- **Contrast recovery** — local lesion contrast pre/post denoise (erasure).
- **Fabrication / hallucination** — `d'` and false-positive rate on
  *signal-absent* ROIs (structure invented where none exists).

**Non-negotiable:** validate the CHO on pure Gaussian noise where `d'` is
analytic, as a unit test — otherwise reviewers won't trust it.

### 4. Harness — `scripts/evaluate_detectability.py` (new)

- Offline; loads a checkpoint (or identity baseline), builds the test set from
  the val full-dose slices, runs the three metrics, logs to W&B / CSV in the
  schema `figures.py` / `benchmark_report.py` already consume.
- Keep it **out of** `train.py:evaluate` — CHO needs many realizations × ROIs,
  far too expensive for the per-epoch loop. The cheap PSNR/SSIM eval stays where
  it is.
- `zsn2n` / `f2n` retrain per image, so detectability eval is very expensive for
  them; restrict to checkpointed models + identity, or accept the cost.

## File-by-file summary

| File | Change |
|---|---|
| `ctdenoiser/detectability.py` | **new** — `insert_signal`, ROI sampling, LG channels, `cho_detectability` |
| `ctdenoiser/metrics.py` | add `uniform_nps`; rename `nps_ratio` → `residual_spectrum` |
| `scripts/evaluate_detectability.py` | **new** — offline CHO/NPS eval over checkpoints → W&B/CSV |
| `scripts/figures.py` | add a detectability panel (`d'` and NPS-shape by method) |
| `tests/test_detectability.py` | **new** — analytic-Gaussian CHO sanity test; `low+s` identity check |
| `paper/main.tex` | corrected-NPS + detectability results = the contribution |

## Sequencing

- [x] **PR 1 (foundation, low-risk):** `insert_signal` + `uniform_nps` +
      `tests/test_detectability.py` (analytic-Gaussian CHO sanity + the
      `low_present = low + s` insertion identity). *Done:*
      `ctdenoiser/detectability.py` (`insert_signal`/`signal_template`,
      `sample_flat_locations`, `extract_rois`); `uniform_nps` added to
      `ctdenoiser/metrics.py`; `nps_ratio` renamed to `residual_spectrum`
      (alias kept).
- [x] **PR 2:** `cho_detectability` (LG channels, template, d'/AUC), validated
      against the analytic-Gaussian unit test
      (`tests/test_detectability.py::test_cho_matches_analytic_dprime_in_white_noise`).
- [x] **PR 3:** `scripts/evaluate_detectability.py` — offline CHO/NPS eval over a
      checkpoint or the identity baseline, CSV logging (input/denoised/clean d',
      detectability-preserved ratio, NPS centroid). *Still to do:* a
      `figures.py` detectability panel (needs real-run CSVs).
- [ ] **PR 4:** run across all benchmark methods + SSFlow; write the
      corrected-NPS + detectability results into `paper/main.tex`.

**≈1.5–2.5 weeks** of focused work to a validated, paper-grade implementation;
the CHO validation is the main risk, the rest is mechanical.

## Parallel rigor work (independent of the above, needed for any venue)

- [ ] Multi-seed (≥3) + per-patient std in Table 1 + a significance test —
      single-seed is an automatic reject, and the SSFlow-vs-Noise2Sim lead
      (+0.2 dB) needs this to survive review.
- [ ] Finish the crashed CFM / CTformer runs to 50 epochs (they currently
      undermine the "objective dominates architecture" claim).
- [ ] One more anatomy (chest/head caches already exist) to move from
      single-window to cross-anatomy.

## What this unlocks for SSFlow

Beyond the benchmark, detectability sharpens the SSFlow story: it shows whether
the model's texture / GMSD behavior corresponds to genuinely better *signal
detectability* or merely prettier-looking noise — directly testing the paper's
central question on our own method.
