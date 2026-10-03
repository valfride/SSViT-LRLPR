# EMA-versus-student statistical summary

Positive deltas mean the matched-epoch EMA model is more accurate than the student.

## Across-seed exact-match accuracy

| Source | F | Seeds | Exact match, mean +/- std (%) |
|---|---:|---:|---:|
| ghost | 1 | 3 | 59.200 +/- 0.300 |
| ghost | 3 | 3 | 72.411 +/- 0.476 |
| ghost | 5 | 3 | 76.433 +/- 0.517 |
| student | 1 | 3 | 59.011 +/- 0.454 |
| student | 3 | 3 | 72.200 +/- 0.850 |
| student | 5 | 3 | 76.033 +/- 0.601 |

## Paired EMA-minus-student effects

| F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |
|---:|---:|---:|---:|
| 1 | 3 | 0.189 +/- 0.190 | [-0.267, 0.689] |
| 3 | 3 | 0.211 +/- 0.389 | [-0.300, 0.744] |
| 5 | 3 | 0.400 +/- 0.133 | [-0.011, 0.811] |

Per-seed exact McNemar tests are stored in ema_paired_seed_tests.csv.
