#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
REV="$ROOT/experiments/revision_eval"

CFG="$REV/submitted_model/config_snapshot.yaml"
CKPT="$REV/submitted_model"
OUT="$REV/reviewer2/03_temporal_fusion/results"
RUNNER="$REV/shared/scripts/run_eval_matrix.py"
COLLECTOR="$REV/shared/scripts/collect_eval_results.py"

DATA="${DATA:-/home/vwnascimento/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k}"

python3 "$RUNNER"     --config "$CFG"     --checkpoints "$CKPT"     --split "$DATA"     --output-dir "$OUT"     --frames 3 5     --fusions bayes average logit_average majority     --skip-existing

python3 "$COLLECTOR"     --input-dir "$OUT"     --output-csv "$OUT/fusion_summary.csv"     --output-md "$OUT/fusion_summary.md"
