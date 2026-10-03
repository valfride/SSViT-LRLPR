# EMA-versus-student statistical summary (best vs best)

Positive deltas mean the independently validation-selected best EMA checkpoint is more accurate than the independently selected best student.

## Across-seed exact-match accuracy

| Source | F | Seeds | Exact match, mean +/- std (%) |
|---|---:|---:|---:|
| ghost | 1 | 3 | 60.267 +/- 0.788 |
| ghost | 3 | 3 | 73.500 +/- 1.272 |
| ghost | 5 | 3 | 77.700 +/- 0.929 |
| student | 1 | 3 | 58.833 +/- 0.491 |
| student | 3 | 3 | 72.467 +/- 0.872 |
| student | 5 | 3 | 75.956 +/- 0.367 |

## Paired EMA-minus-student effects

| F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |
|---:|---:|---:|---:|
| 1 | 3 | 1.433 +/- 0.426 | [0.500, 2.378] |
| 3 | 3 | 1.033 +/- 0.437 | [0.189, 1.922] |
| 5 | 3 | 1.744 +/- 0.589 | [0.867, 2.667] |

Per-seed exact McNemar tests are stored in ema_paired_seed_tests.csv.
