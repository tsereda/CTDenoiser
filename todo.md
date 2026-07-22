# CTDenoiser — cross-domain datasets TODO

_Snapshot: 2026-07-22_

## Status

**Data ready on the PVC (`/data`):**
- `ldct_abdomen.h5` — 50 patients (real LDCT, abdomen window)
- `ldct_chest.h5` — 10 patients (real LDCT, chest window)
- `natural_images/` — 500 BSDS500 images (for `--natural`)
- `head` — **unavailable**: the `LDCT-and-Projection-data` query returns 0 paired
  low/full-dose HEAD patients, so it's excluded from the default `ANATOMIES`.

**Code + paper edits merged to `main`** (PRs #85–#89): the two datasets, sweep
configs, lean job template, data-pod prep, and the theorem-first paper reframing.

## DONE ✓ — natural-image multi-step-hurts (sweep `j637p0kj`)

The flow/steps arm on natural images (gaussian, std 0.1) is **complete and clean**
— the central claim generalises off CT. Mean ΔPSNR over 3 seeds, monotone at
every exclusion radius:

| r | 1 step | 4 | 8 | 20 |
|---|--------|---|---|-----|
| 1 | +3.91 | +1.94 | +1.59 | +1.37 |
| 2 | +3.90 | +1.89 | +1.52 | +1.30 |
| 3 | +4.05 | +1.92 | +1.54 | +1.30 |

Erosion is *sharper* than CT (+3.9→+1.3 vs +1.88→+0.52); `r` barely matters
(i.i.d. noise has no correlation for the exclusion radius to break — the right
behaviour). Export saved → `results/natural_hallucination_gaussian.csv`.
Figure rendered → `figures/steps_departure_natural.pdf`.

**Also rendered** (plan loose ends): `figures/steps_departure.pdf` (CT, paper
fig:steps) and `figures/detectability.pdf`. Only `figure_qualitative.py` remains
(needs checkpoints on the cluster).

## DONE ✓ — pairing arm: one-step flow TIES regression (sweep `k4e2m06c`)

Direct RED-CNN regressor on the same similarity pairs. One-step flow ties the
regressor at every r (regression marginally ahead, the finite-capacity residual —
same as CT):

| r | flow 1-step | regression | gap |
|---|-------------|------------|-----|
| 1 | +3.91 | +4.12 | +0.21 |
| 2 | +3.90 | +4.09 | +0.19 |
| 3 | +4.05 | +4.20 | +0.15 |

Export → `results/natural_flow_vs_reg_gaussian.csv`. Natural steps figure
re-rendered with the regression reference line (`--regression-ref 4.09`).

## DONE ✓ — cross-domain claim written into the paper

Both halves now hold off CT. Added to `paper/main.tex`: a "The result is not
CT-specific" paragraph + `tab:natural` in the theorem section, and Limitations
updated (theorem now has cross-domain empirical support; abdomen-only caveat
scoped to the clinical numbers). **Needs a local `pdflatex` build to eyeball.**

## Then — breadth (optional, if GPU/time before 7/28)

- [ ] Run the other two sweeps:
  - [ ] `sweep_sim_ldct.yml` (45 runs) — clinical robustness on simulated LDCT
        (reuses `ldct_abdomen.h5`; keeps `--eval-detectability`).
  - [ ] `sweep_natural.yml` (90 runs) — full cross-domain benchmark, both noise
        regimes (gaussian + correlated). Trim a seed/regime if GPU is tight.
- [ ] Merge all CSVs into one dashboard (`scripts/benchmark_report.py`).

## Open / later

- [ ] **head**: run the data pod once with `ANATOMIES="abdomen chest head"` to
      capture the new empty-cohort diagnosis (real BodyPartExamined labels), then
      decide: keep it out (no low-dose arm) or remap `BODY_PART` if the paired
      data sits under a different label.
- [ ] Extend the figure scripts (`scripts/figure_*.py`) to render the natural /
      sim-ldct facets alongside the CT results.
