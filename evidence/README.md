# Evidence

The aggregate artifacts the revised manuscript and the response to reviewers cite. Each is the file a stated number was read from, published so that a reader can check a claim without rebuilding anything.

| File | What it is | Answers | |
|---|---|---|---|
| `t_final.json` | the single source every number in the manuscript and both letters is read from; nothing in the paper is typed by hand | R1-3, R1-5, R2-1, R2-4 | 31 KB |
| `lambda_check.json` | the gate scan: the learned quantity's quantiles over every trained model, its global maximum, and the verdict on the ln 2 bound | R1-1 | 1 KB |
| `t42_summary.json` | per-object, per-condition counts and observed null rates for the two discriminating leakage classes | R1-4 | 15 KB |
| `fair_same_host_recovery.json` | the equal-information recovery comparison: effect, interval, exact p, and the scope of what was and was not reproduced | R2-2 | 3 KB |
| `fair_same_host_recovery_cells.csv` | the cell-level table the line above is computed from; experiments/recompute_fair_pair.py reproduces the result from it alone | R2-2 | 120 rows |

These are aggregates and synthetic-regime cells only. Row-level derived tables for MIMIC-IV and eICU are restricted under the PhysioNet data use agreements and are not distributed here; the code to rebuild them from an authorized download is in `experiments/`, and the mask checksums are in `data_manifests/`.
