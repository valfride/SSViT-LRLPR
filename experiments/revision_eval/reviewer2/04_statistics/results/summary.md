# Controlled-ablation statistical summary

Positive paired deltas mean the full model is more accurate than the ablation.

## Across-seed exact-match accuracy

| Variant | F | Seeds | Exact match, mean +/- std (%) |
|---|---:|---:|---:|
| full | 1 | 3 | 60.278 +/- 0.800 |
| full | 3 | 3 | 73.511 +/- 1.253 |
| full | 5 | 3 | 77.700 +/- 0.929 |
| no_restormer | 1 | 3 | 59.567 +/- 0.788 |
| no_restormer | 3 | 3 | 73.211 +/- 0.542 |
| no_restormer | 5 | 3 | 76.556 +/- 0.336 |
| no_pixelshuffle | 1 | 3 | 59.322 +/- 0.429 |
| no_pixelshuffle | 3 | 3 | 72.667 +/- 0.481 |
| no_pixelshuffle | 5 | 3 | 76.311 +/- 0.670 |
| no_sfb | 1 | 3 | 59.356 +/- 0.367 |
| no_sfb | 3 | 3 | 72.567 +/- 0.426 |
| no_sfb | 5 | 3 | 76.667 +/- 0.731 |
| linear_head | 1 | 3 | 59.367 +/- 0.814 |
| linear_head | 3 | 3 | 73.267 +/- 0.285 |
| linear_head | 5 | 3 | 77.556 +/- 0.310 |

## Paired full-minus-ablation effects

| Ablation | F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |
|---|---:|---:|---:|---:|
| no_restormer | 1 | 3 | 0.711 +/- 1.268 | [-0.745, 2.278] |
| no_restormer | 3 | 3 | 0.300 +/- 1.301 | [-1.222, 1.789] |
| no_restormer | 5 | 3 | 1.144 +/- 0.701 | [0.078, 2.189] |
| no_pixelshuffle | 1 | 3 | 0.956 +/- 0.372 | [-0.100, 2.000] |
| no_pixelshuffle | 3 | 3 | 0.844 +/- 1.204 | [-0.611, 2.233] |
| no_pixelshuffle | 5 | 3 | 1.389 +/- 1.551 | [-0.289, 3.056] |
| no_sfb | 1 | 3 | 0.922 +/- 0.918 | [-0.367, 2.289] |
| no_sfb | 3 | 3 | 0.944 +/- 1.422 | [-0.700, 2.500] |
| no_sfb | 5 | 3 | 1.033 +/- 1.233 | [-0.433, 2.378] |
| linear_head | 1 | 3 | 0.911 +/- 1.500 | [-0.767, 2.567] |
| linear_head | 3 | 3 | 0.244 +/- 1.271 | [-1.300, 1.656] |
| linear_head | 5 | 3 | 0.144 +/- 1.078 | [-1.211, 1.378] |

Per-seed exact McNemar tests, with Holm correction within each seed/F family, are stored in paired_seed_tests.csv.
