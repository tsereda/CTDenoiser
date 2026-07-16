# Result exports

W&B CSV exports behind the paper's tables and figures (abdomen, 3 seeds, 50 ep).
Each row is one run; `det/*` columns hold the CHO/NPS detectability metrics.

| File | W&B sweep | Config | Feeds |
|------|-----------|--------|-------|
| `benchmark_abdomen.csv`   | `55if6jcw` | `sweep.yml` (5 arch × 3 modes × 3 seeds) | `tab:results`, `tab:detect`, `fig:detect` |
| `flow_vs_reg_abdomen.csv` | `iidab9pq` | `sweep_flow_vs_reg.yml` (RED-CNN N2Sim × r∈{1,2,3} × 3 seeds) | `tab:flow-vs-reg` (regression arm) |
| `hallucination_abdomen.csv` | `spy7s4zh` | `sweep_hallucination.yml` (SSFlow × r × eval-steps {1,4,8,20}) | `tab:flow-vs-reg` (flow arm), `tab:ssflow-steps`, `fig:steps` |

## Regenerate figures
```
python scripts/figure_steps.py         results/hallucination_abdomen.csv --regression-ref 1.99 --out-dir figures --fmt pdf
python scripts/figure_detectability.py results/benchmark_abdomen.csv     --out-dir figures --fmt pdf
```
Multi-anatomy exports (chest/head) drop in here with the same names + `_chest`/`_head`.
