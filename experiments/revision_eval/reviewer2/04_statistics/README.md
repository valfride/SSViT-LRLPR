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

## 2. Run the structural/classifier statistics

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

## 3. Evaluate the matched-epoch EMA ablation

EMA must be isolated without changing checkpoint-selection epoch. The dedicated
evaluator finds the EMA/ghost checkpoint already used for each full-model seed,
then evaluates the student checkpoint from that exact same epoch.

First pull the latest revision tooling, then run a dry-run:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --dry-run \
  --strict
```

For seeds 123 and 2026, the selected EMA epoch is read from the existing F1 ghost
evaluation JSON and the matching student checkpoint is taken from the corresponding
controlled full-model run.

For seed 42, the script reads the epoch stored in the frozen submitted checkpoint
and searches the historical submitted training trajectory:

```text
experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/student_weights/
```

If the submitted checkpoint does not expose an epoch, provide it explicitly:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --seed42-epoch EPOCH \
  --dry-run \
  --strict
```

If the historical student checkpoint is stored elsewhere, use either
`--seed42-run-dir` or `--seed42-student-checkpoint`.

Once the dry-run resolves all three matched pairs, evaluate them:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --gpu 0 \
  --skip-existing \
  --keep-going \
  --strict
```

The student outputs are written beside the existing EMA outputs:

```text
reviewer2/01_controlled_ablation/results/evaluations/
  full/
    seed42/
      ghost/
      student/
    seed123/
      ghost/
      student/
    seed2026/
      ghost/
      student/
```

The evaluator also records the exact EMA epoch and matching student checkpoint in:

```text
reviewer2/01_controlled_ablation/results/ema_ablation/
  ema_evaluation_manifest.json
```

## 4. Run the EMA statistics

After all nine matched student evaluations (3 seeds x F1/F3/F5) are complete:

```bash
python3 \
  experiments/revision_eval/reviewer2/04_statistics/scripts/analyze_ema_statistics.py \
  --strict
```

This reports mean +/- sample standard deviation for EMA and student weights,
per-seed paired bootstrap 95% confidence intervals, exact McNemar tests, and a
hierarchical paired-bootstrap 95% confidence interval across seeds.

For this analysis:

```text
delta = EMA exact-match accuracy - matched-epoch student exact-match accuracy
```

A positive delta therefore means EMA improves exact recognition.

EMA outputs are written under:

```text
reviewer2/04_statistics/results/ema/
  ema_seed_summary.csv
  ema_paired_seed_tests.csv
  ema_hierarchical_bootstrap.csv
  ema_summary.md
  ema_analysis_manifest.json
```

The bootstrap RNG seed and number of bootstrap replicates are recorded in the
analysis manifests so the reported intervals are reproducible.
