# Reviewer 2.4 — Statistical validation

The statistical pipeline consumes the structured outputs produced by the controlled
ablation evaluator.

## 1. Evaluate the trained models

After training is complete and a GPU is free:

```bash
python3 experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_all.py \
  --gpu 0 \
  --skip-existing \
  --keep-going
```

By default this evaluates the EMA/ghost checkpoint for:

- variants: `full`, `no_restormer`, `no_pixelshuffle`, `no_sfb`, `linear_head`;
- seeds: 42, 123, 2026;
- temporal settings: F=1, F=3, F=5;
- fusion: product-rule / sum-log-probability (`bayes` in the evaluator CLI).

The frozen submitted checkpoint is used as the full-model seed-42 reference.
Other runs are read from the controlled-ablation checkpoint tree.

Outputs are written under:

```text
reviewer2/01_controlled_ablation/results/evaluations/
  <variant>/seed<seed>/ghost/
    F1_bayes.json
    F1_bayes_predictions.csv
    F3_bayes.json
    F3_bayes_predictions.csv
    F5_bayes.json
    F5_bayes_predictions.csv
```

## 2. Run the statistics

```bash
python3 experiments/revision_eval/reviewer2/04_statistics/scripts/analyze_ablation_statistics.py \
  --strict
```

The analysis reports:

- mean and sample standard deviation across training seeds;
- paired track-level bootstrap 95% confidence intervals;
- exact paired McNemar tests for each seed and temporal setting;
- Holm-Bonferroni-adjusted McNemar p-values within each seed/F comparison family;
- hierarchical paired-bootstrap 95% confidence intervals that resample tracks
  within seeds and then resample seeds.

For paired ablation effects, the reported quantity is:

```text
delta = full-model exact-match accuracy - ablation exact-match accuracy
```

Therefore, a positive delta means the full model is more accurate.

Primary outputs:

```text
reviewer2/04_statistics/results/
  ablation_seed_summary.csv
  paired_seed_tests.csv
  paired_hierarchical_bootstrap.csv
  summary.md
  analysis_manifest.json
```

The bootstrap RNG seed and number of bootstrap replicates are recorded in the
analysis manifest so the reported intervals are reproducible.
