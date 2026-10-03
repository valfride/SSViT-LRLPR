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

## 3. Evaluate the EMA ablation

The EMA evaluator supports two checkpoint-selection protocols.

### 3.1 Matched-final-epoch comparison

This is the stricter causal comparison. It evaluates `student_weights/last.pth`
and `ghost_weights/last.pth` from the same completed training run and requires
the two checkpoints to report the same epoch.

Dry-run:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --selection last \
  --dry-run \
  --strict
```

Evaluate:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --selection last \
  --gpu 0 \
  --skip-existing \
  --keep-going \
  --strict
```

Outputs remain under the original path for backward compatibility:

```text
reviewer2/01_controlled_ablation/results/ema_ablation/
  evaluations/seed<seed>/{student,ghost}/
  ema_evaluation_manifest.json
```

### 3.2 Best-vs-best comparison

This is the practical validation-selection comparison and is the one intended for
the primary EMA ablation table. Student and EMA checkpoints are selected
independently using the highest validation accuracy encoded in the retained
top-checkpoint filenames:

```text
student_weights/student_acc_<acc>_ep_<epoch>.pth
ghost_weights/ghost_acc_<acc>_ep_<epoch>.pth
```

The best student and best EMA are therefore allowed to come from different epochs.
Ties in validation accuracy are broken by later epoch.

Dry-run:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --selection best \
  --dry-run \
  --strict
```

Evaluate:

```bash
python3 \
  experiments/revision_eval/reviewer2/01_controlled_ablation/scripts/evaluate_ema_ablation.py \
  --selection best \
  --gpu 0 \
  --skip-existing \
  --keep-going \
  --strict
```

Best-vs-best outputs are kept separate from the matched-final-epoch results:

```text
reviewer2/01_controlled_ablation/results/ema_ablation/best/
  evaluations/seed<seed>/{student,ghost}/
  ema_evaluation_manifest.json
```

For seed 42, both modes use the historical submitted training trajectory by
default:

```text
experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/
```

Use `--seed42-run-dir` only if that historical run is stored elsewhere.

## 4. Run the EMA statistics

Matched-final-epoch statistics:

```bash
python3 \
  experiments/revision_eval/reviewer2/04_statistics/scripts/analyze_ema_statistics.py \
  --selection last \
  --strict
```

These remain under:

```text
reviewer2/04_statistics/results/ema/
```

Best-vs-best statistics:

```bash
python3 \
  experiments/revision_eval/reviewer2/04_statistics/scripts/analyze_ema_statistics.py \
  --selection best \
  --strict
```

These are written separately under:

```text
reviewer2/04_statistics/results/ema_best/
  ema_seed_summary.csv
  ema_paired_seed_tests.csv
  ema_hierarchical_bootstrap.csv
  ema_summary.md
  ema_analysis_manifest.json
```

Both analyses report mean +/- sample standard deviation, paired track-level
bootstrap 95% confidence intervals, exact McNemar tests, and hierarchical
paired-bootstrap 95% confidence intervals across seeds.

For both protocols:

```text
delta = EMA exact-match accuracy - student exact-match accuracy
```

A positive delta means EMA has higher exact recognition under that checkpoint
selection protocol.

The two protocols answer different questions and should not be conflated:
`best` measures the practical effect after independent validation-based model
selection, whereas `last` isolates EMA at a common training epoch.


## 5. Deterministic evaluation check

The evaluation entry point overrides the training-time cuDNN benchmark setting and
uses deterministic cuDNN kernels:

```text
seed = 42
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

This affects evaluation only; the training configuration is unchanged. The metrics
JSON records these settings under `reproducibility`.

Before freezing manuscript numbers, verify one representative checkpoint by running
the exact same evaluation twice and comparing the prediction CSVs byte-for-byte:

```bash
python3 \
  experiments/revision_eval/shared/scripts/verify_eval_determinism.py \
  --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
  --checkpoints experiments/revision_eval/submitted_model \
  --split /home/vwnascimento/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k \
  --frames 1 \
  --fusion bayes \
  --gpu 0
```

A successful run ends with:

```text
Predictions byte-identical: True
Metrics identical:          True
✅ Determinism check PASSED.
```

The check writes `determinism_manifest.json` under
`experiments/revision_eval/shared/results/determinism_check/`. By default the two
temporary run directories are removed after a successful comparison; use
`--keep-output` to retain them.
