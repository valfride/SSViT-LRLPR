# Controlled-ablation statistical summary

Positive paired deltas mean the full model is more accurate than the ablation.

## Across-seed exact-match accuracy

| Variant | F | Seeds | Exact match, mean +/- std (%) |
|---|---:|---:|---:|
| full | 1 | 3 | 60.267 +/- 0.788 |
| full | 3 | 3 | 73.500 +/- 1.272 |
| full | 5 | 3 | 77.700 +/- 0.929 |
| no_restormer | 1 | 3 | 59.544 +/- 0.757 |
| no_restormer | 3 | 3 | 73.211 +/- 0.564 |
| no_restormer | 5 | 3 | 76.544 +/- 0.353 |
| no_pixelshuffle | 1 | 3 | 59.367 +/- 0.384 |
| no_pixelshuffle | 3 | 3 | 72.678 +/- 0.467 |
| no_pixelshuffle | 5 | 3 | 76.311 +/- 0.638 |
| no_sfb | 1 | 3 | 59.344 +/- 0.350 |
| no_sfb | 3 | 3 | 72.567 +/- 0.426 |
| no_sfb | 5 | 3 | 76.656 +/- 0.719 |
| linear_head | 1 | 3 | 59.344 +/- 0.792 |
| linear_head | 3 | 3 | 73.267 +/- 0.285 |
| linear_head | 5 | 3 | 77.533 +/- 0.291 |

## Paired full-minus-ablation effects

| Ablation | F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |
|---|---:|---:|---:|---:|
| no_restormer | 1 | 3 | 0.722 +/- 1.239 | [-0.756, 2.267] |
| no_restormer | 3 | 3 | 0.289 +/- 1.301 | [-1.200, 1.767] |
| no_restormer | 5 | 3 | 1.156 +/- 0.685 | [0.100, 2.167] |
| no_pixelshuffle | 1 | 3 | 0.900 +/- 0.406 | [-0.200, 1.956] |
| no_pixelshuffle | 3 | 3 | 0.822 +/- 1.208 | [-0.656, 2.233] |
| no_pixelshuffle | 5 | 3 | 1.389 +/- 1.518 | [-0.244, 3.011] |
| no_sfb | 1 | 3 | 0.922 +/- 0.900 | [-0.367, 2.244] |
| no_sfb | 3 | 3 | 0.933 +/- 1.440 | [-0.733, 2.500] |
| no_sfb | 5 | 3 | 1.044 +/- 1.237 | [-0.400, 2.400] |
| linear_head | 1 | 3 | 0.922 +/- 1.451 | [-0.700, 2.600] |
| linear_head | 3 | 3 | 0.233 +/- 1.290 | [-1.300, 1.633] |
| linear_head | 5 | 3 | 0.167 +/- 1.068 | [-1.200, 1.378] |

Per-seed exact McNemar tests, with Holm correction within each seed/F family, are stored in paired_seed_tests.csv.
