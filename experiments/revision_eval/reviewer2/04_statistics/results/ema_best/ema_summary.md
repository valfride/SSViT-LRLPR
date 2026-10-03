# EMA-versus-student statistical summary (best vs best)

Positive deltas mean the independently validation-selected best EMA checkpoint is more accurate than the independently selected best student.

## Across-seed exact-match accuracy

| Source | F | Seeds | Exact match, mean +/- std (%) |
|---|---:|---:|---:|
| ghost | 1 | 3 | 60.278 +/- 0.769 |
| ghost | 3 | 3 | 73.511 +/- 1.253 |
| ghost | 5 | 3 | 77.700 +/- 0.929 |
| student | 1 | 3 | 58.844 +/- 0.506 |
| student | 3 | 3 | 72.467 +/- 0.872 |
| student | 5 | 3 | 75.956 +/- 0.367 |

## Paired EMA-minus-student effects

| F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |
|---:|---:|---:|---:|
| 1 | 3 | 1.433 +/- 0.416 | [0.500, 2.367] |
| 3 | 3 | 1.044 +/- 0.419 | [0.167, 1.900] |
| 5 | 3 | 1.744 +/- 0.589 | [0.833, 2.633] |

Per-seed exact McNemar tests are stored in ema_paired_seed_tests.csv.
