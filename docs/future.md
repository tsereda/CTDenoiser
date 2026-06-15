# Research Roadmap

This document records the long-term research direction for CTDenoiser so that
individual changes (new models, training modes, metrics) are understood as steps
toward a concrete publication, not one-offs.

## Thesis / target venue

**WACV applications paper (Option B):** *"Do generative low-dose CT denoisers
hallucinate, and do current metrics hide it? A task-based, hallucination-aware
evaluation."*

The contribution is an **evaluation methodology + an empirical finding**, not a
new architecture. This is a legitimate, well-respected applications-track paper
and is the contribution this codebase is closest to delivering: we already have a
working generative model (conditional flow matching) and a residual-spectrum
metric, and the literature explicitly names cohort-scale hallucination
quantification in CT as an open gap.

## Benchmark substrate

The empirical substrate is a common benchmark of all models trained and evaluated
under one harness (patient-split, full-slice overlapped inference, shared
metrics):

- **Supervised:** RED-CNN, DnCNN, U-Net, CTFormer, Conditional Flow Matching.
- **Self-supervised / zero-shot:** Noise2Void (blind-spot masking) and Zero-Shot
  Noise2Noise (ZS-N2N) — added so the benchmark covers the
  *supervision-free* regime, which the review identifies as the field's clearest
  growth area (paired LDCT/NDCT data are scarce).

These two self-supervised rows are the deliverable of the current change. They are
fixed-cost (no external data, reuse the existing harness) and are evaluated against
the clean full-dose reference so their PSNR/SSIM/RMSE/GMSD/NPS sit in the same
table as the supervised models.

Expected (and reportable) finding: N2V's blind-spot scheme assumes spatially
*independent* noise, which CT violates (noise is spatially correlated and
signal-dependent), so N2V should trail ZS-N2N, which keeps both pixels of each 2x2
block. That contrast motivates the next baselines.

**Planned next baselines:** Noise2Sim (similarity-based, explicitly
correlated-noise-aware) and Filter2Noise (interpretable zero-shot single-image),
closing the gap to current self-supervised SOTA.

## Phased plan

- **Phase 0 — de-risk / fill tables (long pole, start first):** launch full-TCIA
  training of all benchmark methods on the K8s GPU cluster; multi-seed for
  significance. Data and compute are not the blockers.
- **Phase 1 — evaluation methodology:** signal-insertion + low-dose re-simulation
  (insert known low-contrast signals into full-dose images, re-simulate the
  low-dose pair, measure whether each denoiser *preserves / erases / fabricates*
  structure — the FDA/CHO detectability approach, ground-truth known). Add a real
  detectability / hallucination metric and a **corrected NPS**: the current
  `nps_ratio` is mislabeled and must be split into a uniform-ROI NPS plus an
  inserted-signal detectability measure.
- **Phase 2 — experiments:** one-step vs. K-step Euler sampling sweep on the flow
  model to test the "No-New-Denoiser" finding on CT (does iterative stochastic
  sampling trade fidelity/detectability for texture?); cross-dose / cross-kernel
  generalization; ROI qualitative figures.
- **Phase 3 — write-up:** rewrite around the finding, multi-seed tables with
  significance.

## Open gaps tracked

- **Supervision bottleneck** — paired data scarce; self-supervised/zero-shot is the
  growth area (addressed in part by this change).
- **Generalization** — models degrade across scanner / dose / kernel / anatomy.
- **Efficiency / deployability** — diffusion sampling latency vs. clinical
  throughput; favour few-/single-step generative models (flow matching, consistency).
- **Hallucination vs. over-smoothing** — generative fidelity vs. fabricated
  structure; under-quantified at cohort scale (the paper's target).
- **Metric / diagnostic gap** — PSNR/SSIM misalign with detectability; report
  task-based metrics alongside them.
- **Realistic noise modeling** — CT noise is compound-Poisson + electronic
  Gaussian, spatially correlated; not additive white Gaussian.

## How the current change fits

This change delivers the self-supervised/zero-shot benchmark rows (review
Recommendation #1) via `--training-mode {n2v,zsn2n}`. Subsequent changes add the
Phase 1 evaluation methodology (signal insertion, corrected NPS, detectability).
