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

**Positioning against the closest prior art (the differentiator).** Three recent
works each cover one leg of our tripod, and none combines all three at cohort
scale — this is exactly the gap we fill:
- **Tivnan et al. (MICCAI 2024), "Hallucination Index"** — measures hallucination
  *distributionally* (Hellinger distance to a zero-hallucination reference for
  diffusion reconstruction). We instead use a *task-based* (detectability)
  readout and re-attach *real correlated FBP noise* to controlled inserted
  signals.
- **Kc & Zeng (IEEE NSS/MIC 2024), LG-CHO denoiser assessment** — already shows
  the exact "metrics lie" effect (deep denoisers beat LDCT by 2.4–3.8 dB PSNR /
  0.05–0.11 SSIM yet have *inferior* low-contrast detectability), but on
  phantom/insertion data, not cohort-scale, and without a fabrication axis.
- **Li, Zhou, Li & Anastasio (IEEE TMI 2021)** — SKE/BKS binary detection with DNN
  denoisers showing denoising can destroy task-relevant information; the
  detection-efficiency leg, but no fabrication/NPS-texture analysis.

Our novelty claim is the **unification**: controlled signal-insertion + CHO
(d′/AUC) + corrected NPS, run over a cohort, with an explicit
**preserve / erase / fabricate** decomposition.

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

**Why we re-attach *real* FBP noise rather than rely on independence-assuming
self-supervision (theory anchor).** Zhao et al. (J. Imaging Inform. Med. 2025)
make the cleanest argument: Noise2X methods "cannot manage spatially correlated or
non-zero mean noise" in the *image* domain and are theoretically better suited to
the *sinogram* domain, because per-detector photoelectric conversion makes raw
projection noise pixel-independent — whereas FBP smears it into spatially
correlated image-domain noise. Our `n = low − full` re-attachment exercises exactly
this correlated noise, sidestepping the independence assumption that N2V / ZS-N2N /
Neighbor2Neighbor rely on (and which Noise2Sim and AP-BSN-style decorrelation are
designed to break).

## Phased plan

- **Phase 0 — de-risk / fill tables (long pole, start first):** launch full-TCIA
  training of all benchmark methods on the K8s GPU cluster; multi-seed for
  significance. Data and compute are not the blockers. Adopt Eulig et al.
  (Medical Physics 2024) unified-split discipline so PSNR/SSIM stay comparable
  across methods (their core finding is that inconsistent splits make published
  numbers non-comparable).
- **Phase 1 — evaluation methodology:** signal-insertion + low-dose re-simulation
  (insert known low-contrast signals into full-dose images, re-simulate the
  low-dose pair, measure whether each denoiser *preserves / erases / fabricates*
  structure — the FDA/CHO detectability approach, ground-truth known). Add a real
  detectability / hallucination metric and a **corrected NPS**: the current
  `nps_ratio` is mislabeled and must be split into a uniform-ROI NPS plus an
  inserted-signal detectability measure. CHO design (LG channels, d′/AUC) follows
  the Barrett & Myers / AAPM task-based-assessment foundations and must be
  validated against the analytic-Gaussian white-noise case (where d′ is closed
  form) before reviewers will trust it. NPS-in-anatomy caveat: bench NPS/cdMTF on
  uniform phantoms does *not* predict patient-background performance for DL
  denoisers, which is itself the argument for object-insertion into patient slices.
- **Phase 2 — experiments:** one-step vs. K-step Euler sampling sweep on the flow
  model to test the "No-New-Denoiser" finding on CT (does iterative stochastic
  sampling trade fidelity/detectability for texture?); cross-dose / cross-kernel
  generalization; ROI qualitative figures. Frame the trade-off through Blau &
  Michaeli (perception–distortion): realism gains that cost detectability are
  exactly the regime the CHO catches. Hein et al. (PFCM, IEEE TMI 2025) give a
  concrete CT admission that an unconstrained generative process introduces
  variability/mismatch and needs a fidelity-preserving sampler — direct support
  for the one-step result.
- **Phase 3 — write-up:** rewrite around the finding, multi-seed tables with
  significance.

## Open gaps tracked

- **Supervision bottleneck** — paired data scarce; self-supervised/zero-shot is the
  growth area (addressed in part by this change). See Zhao et al. 2025 review for
  why image-domain self-supervision struggles on correlated CT noise.
- **Generalization** — models degrade across scanner / dose / kernel / anatomy.
- **Efficiency / deployability** — diffusion sampling latency vs. clinical
  throughput; favour few-/single-step generative models (flow matching, consistency;
  e.g. Hein et al. PFCM 2025).
- **Hallucination vs. over-smoothing** — generative fidelity vs. fabricated
  structure; under-quantified at cohort scale (the paper's target). Distributional
  (Tivnan 2024), frequency-domain (sFRC; CHEM 2025), and perturbation-based (Antun
  2020) hallucination metrics exist but none couples a *known inserted signal* to a
  *detectability* readout — our niche.
- **Metric / diagnostic gap** — PSNR/SSIM misalign with detectability; report
  task-based metrics alongside them. Strong recent peer-reviewed support: Breger et
  al. (J. Imaging Inform. Med. 2025, "why we need to reassess FR-IQA with medical
  images") and Lee et al. (Medical Image Analysis 2025, LDCT perceptual-IQA dataset
  / MICCAI 2023 challenge, PSNR/SSIM ≠ radiologist perception).
- **Realistic noise modeling** — CT noise is compound-Poisson + electronic
  Gaussian, spatially correlated; not additive white Gaussian.

## Must-cite literature (2024–2026) to bring the paper current

Added from the latest lit review; these are the references the current bib is
missing and should not go to a medical-imaging venue without:

- **Hallucination / task-based evaluation:** Tivnan et al. 2024 (Hallucination
  Index, MICCAI); Kc & Zeng 2024 (LG-CHO denoiser assessment, IEEE NSS/MIC);
  Li, Zhou, Li & Anastasio 2021 (binary detection w/ DNN denoisers, IEEE TMI);
  Bhadra et al. 2021 (hallucinations in tomographic reconstruction, IEEE TMI);
  Antun et al. 2020 (instabilities, PNAS); Blau & Michaeli 2018
  (perception–distortion, CVPR). Optional/frontier: CHEM 2025; sFRC (Kc et al.).
- **Self-supervised / correlated-noise CT:** Niu et al. 2023 (Noise2Sim, IEEE TMI);
  WIA-LD2ND 2024 (MICCAI); Filter2Noise 2025 (Sun et al.; already cited as
  baseline); AP-BSN 2022 (Lee et al., CVPR); Zhao et al. 2025
  (self-supervised LDCT review, J. Imaging Inform. Med.).
- **Generative fidelity / one-step:** Hein et al. 2025 (PFCM, IEEE TMI).
- **Benchmarks / IQA reassessment:** Eulig et al. 2024 (benchmark, Medical Physics);
  Breger et al. 2025 (FR-IQA reassessment, J. Imaging Inform. Med.); Lee et al. 2025
  (LDCT-IQA dataset, Medical Image Analysis).

Metadata to verify before final submission: Noise2Sim cite ambiguity (arXiv
preprint title differs from IEEE TMI 42(6):1590–1602, 2023 — cite the TMI version
as primary); AP-BSN page range varies by index (CVF lists 17725–17734).

## Threshold conditions that would change the framing

- If a generative denoiser shows CHO d′/AUC **non-inferiority to NDCT *and***
  matched NPS at fixed dose, the hallucination concern is largely mitigated and the
  paper becomes a *validation tool* rather than a *warning*.
- If fabricated low-contrast lesions **survive human reading** (a multi-reader study
  confirming the CHO signal), hallucination escalates from methodological caveat to
  patient-safety priority and justifies a follow-on reader study.

## How the current change fits

This change delivers the self-supervised/zero-shot benchmark rows (review
Recommendation #1) via `--training-mode {n2v,zsn2n}`. Subsequent changes add the
Phase 1 evaluation methodology (signal insertion, corrected NPS, detectability).