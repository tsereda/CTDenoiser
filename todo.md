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

## NEXT — the pairing arm (completes the cross-domain claim)

```bash
# regression arm — confirms one-step flow TIES regression on natural images
python sweep.py sweep_natural_flow_vs_reg.yml --template k8s/tr_job_template_synth.yml --agents 3
```
Wait for the previous Job to finish first (same `ctdenoiser-sweep` name), or use a
separate namespace. Then re-run `figure_steps.py` with `--regression-ref <best>` so
the natural figure gets its dashed one-step-regression line like the CT one.

## Then — after each finishes

- [ ] Export CSV → `results/` (`natural_flow_vs_reg_gaussian.csv`, etc.).
      `python scripts/benchmark_report.py --wandb timgsereda/ctdenoiser-sweep/sweeps/<ID>`
- [ ] Draft the cross-domain results paragraph + table in `paper/main.tex`
      (edit #3 from the review — the empirical answer to "Gaussian-only" / "CT-specific").
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
- [ ] FYI for the separate-namespace plan: the k8s manifests hardcode
      `namespace: usd-djha` and the PVC `ctdenoiser` (namespace-scoped) — a job in
      another namespace needs that same PVC available there.
