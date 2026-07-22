# CTDenoiser — cross-domain datasets TODO

_Snapshot: 2026-07-22_

## Status

**Data ready on the PVC (`/data`):**
- `ldct_abdomen.h5` — 50 patients (real LDCT, abdomen window)
- `ldct_chest.h5` — 10 patients (real LDCT, chest window)
- `natural_images/` — 500 BSDS500 images (for `--natural`)
- `head` — **unavailable**: the `LDCT-and-Projection-data` query returns 0 paired
  low/full-dose HEAD patients, so it's excluded from the default `ANATOMIES`.

**Code merged to `main`** (PRs #85–#87): the two new datasets
(`NaturalImageDenoisingDataset`, `SimulatedLowDoseCTDataset`), the sweep configs,
the lean job template, and the data-pod prep.

## Tonight — run these two (highest leverage, cheapest: 9 runs each)

The **"one-step flow ties regression, multi-step hurts — outside CT"** pair. They
overlay cell-for-cell (same exclude-radius × seed grid, gaussian noise), so
together they generalise the paper's central claim off CT data.

```bash
# flow / steps arm  — multi-step-hurts (val/steps{k}/psnr erodes with k)
python sweep.py sweep_natural_hallucination.yml --template k8s/tr_job_template_synth.yml --agents 3

# regression arm    — one-step flow ties regression
python sweep.py sweep_natural_flow_vs_reg.yml   --template k8s/tr_job_template_synth.yml --agents 3
```

- Running in **separate namespaces** so the two k8s Jobs don't share the
  `ctdenoiser-sweep` name (the deploy step deletes a same-named Job).
- `--agents 3` = 3 pods × 3 packed agents = all 9 runs at once. Drop to
  `--agents 1–2` if A100s are scarce (fewer pods, more waves).

## Tomorrow — after they finish

- [ ] Export each sweep's W&B CSV → `results/`
      (`natural_hallucination_gaussian.csv`, `natural_flow_vs_reg_gaussian.csv`).
      `python scripts/benchmark_report.py --wandb timgsereda/ctdenoiser-sweep/sweeps/<ID>`
- [ ] Sanity-check the result: regression(r) roughly flat across exclude-radius
      vs flow(r); `val/steps{k}/psnr` monotonically erodes 1→4→8→20.
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
