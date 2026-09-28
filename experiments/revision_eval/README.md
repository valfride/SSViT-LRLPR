# OJ-ITS Major Revision Workspace

## Canonical submitted model
`submitted_model/`

Reproduced LRLPR-26 test results:
- F=1 BJP: 60.60%
- F=3 BJP: 74.17%
- F=5 BJP: 78.47%

## Editor-in-Chief
- `eic/01_recent_ojits_literature/`
  - Add relevant recent OJ-ITS literature.

## Reviewer 1
- `reviewer1/01_efficiency/`
  - Parameters, FLOPs, memory, per-frame latency, per-tracklet latency.
- `reviewer1/02_cross_dataset/`
  - External real-world evaluation.
- `reviewer1/03_decoder_specification/`
  - Decoder layers, dimensions, heads, positional queries, token consumption,
    fixed-length handling, decoding details.
- `reviewer1/04_svtrv2_ar/`
  - Investigate/add SVTRv2-AR baseline.

## Reviewer 2
- `reviewer2/01_controlled_ablation/`
  - Controlled Restormer / PixelShuffle / SFB / cosine / EMA ablations.
- `reviewer2/02_table_consistency/`
  - Reconcile Table 1 and Table 2.
- `reviewer2/03_temporal_fusion/`
  - Compare BJP/product-rule, probability averaging, logit averaging,
    and sequence-level majority voting.
- `reviewer2/04_statistics/`
  - Multiple seeds, std/CI, significance.
- `reviewer2/05_reproducibility/`
  - Exact architecture, augmentation, training schedule, stopping,
    decoder and EMA details.

## Shared
`shared/` contains reusable scripts, manifests, tables, and notes.
