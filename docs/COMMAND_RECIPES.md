# Paper commands

From the source root, set `WEIGHTS` to your private JSON asset mapping, `TEST` to frozen3,000track test LMDB, `VAL` to frozen 1,000 track validation, `LPLCV2` to exact padded 7,023 LMDB, and `OUT` to a fresh private parent outside this checkout.

## Tables1–3

```bash
python -m reproducibility.run plan --table 1
python -m reproducibility.run preflight --table 1 --weights "$WEIGHTS" --dataset "$TEST"
python -m reproducibility.run preflight --table 3 --weights "$WEIGHTS" --dataset "$LPLCV2"

# Separately authorized fresh executions only:
python -m reproducibility.run evaluate --table 1 --weights "$WEIGHTS" \
  --dataset "$TEST" --output "$OUT/table1" --execute
python -m reproducibility.run evaluate --table 2 --weights "$WEIGHTS" \
  --dataset "$TEST" --output "$OUT/table2" --execute
python -m reproducibility.run evaluate --table 3 --weights "$WEIGHTS" \
  --dataset "$LPLCV2" --output "$OUT/table3" --execute

python -m reproducibility.verify --aggregate "$OUT/table1/aggregate.json"
python -m reproducibility.verify --aggregate "$OUT/table2/aggregate.json"
python -m reproducibility.verify --aggregate "$OUT/table3/aggregate.json"
```

Table1: 24 evaluations,exact/partial counts. Table2: 64 post-selection controls,8 models xF3/F5 x4methods. Table3:8F1evaluations and private per-annotation subgroup joins. PRF retains `--fusion bayes`; other final rules are `average`,`majority`,`char_majority`. Underlying mode is `val` for metrics with explicit test phase; no checkpoint selection. `logit_average` remains compatible but is not final Table2.

`--models ours` supports partial diagnostics; complete-table verification requires all 8 models. Frozen validation choice remains PRF in 16 settings; a separately authorized **replay**, never reselection, can use Table2 `--validation --dataset "$VAL"`. Existing selection aggregates are in `VALIDATION_SELECTION.csv`. The wrapper never changes choices from new validation or test scores.

The original LPLCv2 paired-cohort builder and geometry helper are included under `experiments/revision_eval/reviewer1/02_cross_dataset/scripts/`. It uses licensed original annotations/images and frozen rules; only the padded 20% 7,023variant is reported. `--preflight`, `--build` and `--verify-only` are separate historical operations. No builder was executed here. Exact output file digests are required by the runner.

## Table4

```bash
python -m reproducibility.run plan --table 4
python -m reproducibility.run preflight --table 4 --weights "$WEIGHTS" --dataset "$TEST"
# Separately authorized GPU benchmark only:
python -m reproducibility.run evaluate --table 4 --weights "$WEIGHTS" \
  --dataset "$TEST" --output "$OUT/table4" --execute
python -m reproducibility.verify --efficiency "$OUT/table4/benchmark/efficiency_summary.csv"
```

All 8 models and checkpoint-file overrides are explicit; default legacy omission of AR is avoided. FP16, 100 same tracks, 30 warmups/100 synchronized iterations, 10 FLOP samples. Report batched-F5 including fusion, not sequential-F5. Parameters/FLOPs must match displayed values; timing/memory differences remain explicit without inventing a tolerance. `--strict-timing` requires exact displayed agreement. Retain preflight/run/worker evidence.

## Table5

```bash
python -m reproducibility.run plan --table 5
python -m reproducibility.run preflight --table 5 --weights "$WEIGHTS" --dataset "$TEST"
# Separately authorized frozen-state inference; no retraining/reselection:
python -m reproducibility.run evaluate --table 5 --weights "$WEIGHTS" \
  --dataset "$TEST" --output "$OUT/table5" --execute
python -m reproducibility.verify --aggregate "$OUT/table5/aggregate.json"
```

18 original selected EMA/primary states x3 frames=54 evaluations. Full and four one-factor variants use seeds42/123/2026. The selected primary states belong to the same full-model trajectories, not separately trained architectures. Verification uses aggregate counts/means/sample-std, without raw predictions or significance claims. Original15configs remain under `experiments/revision_eval/reviewer2/01_controlled_ablation/configs/`.

## Training is separate

```bash
# Prospective training only; separately authorized, never on historical outputs:
CUDA_VISIBLE_DEVICES=0 DEBUG=True python train_gan.py \
  --config ablation_configs/paper_cnn_vit_from_scratch.yaml \
  --save /path/to/new/private/training-output --seed 42
```

This CE recipe is not an exact replay of the continued original. Baseline retraining must use its own frozen config/native objective/scheduler and fresh outputs. Neither new training nor generic directory discovery may replace the original selected files in the paper manifests.

## Reported values versus count arithmetic

```bash
# Passes reported-reference checks with the three exact approved Table3 displays disclosed.
python -m reproducibility.verify --references --strict-paper-display

# Separate arithmetic diagnostic: intentionally fails for those three historical differences.
python -m reproducibility.verify --references --strict-direct-rounding
```

The same switches apply to `--aggregate` for a complete Table3 result. Approval is limited to the reported reference displays; a raw evaluator percentage of80.5 in place of100*5650/7023 still fails arithmetic validation. Manuscript updates are not fetched or edited automatically. The displayed Tables 1–5 were checked against pinned manuscript `775a045`; use `python -m reproducibility.paper_audit --paper-root /path/to/authorized/paper` for that reviewed layout. No manuscript is bundled. Diagnostic `--frames` subsets remain subject to complete-table aggregate verification.
