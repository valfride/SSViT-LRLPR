#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
REV="$ROOT/experiments/revision_eval"

CFG="$ROOT/baselines_configs/SVTRV2_BASELINE.yaml"
RUNNER="$REV/shared/scripts/run_eval_matrix.py"
COLLECTOR="$REV/shared/scripts/collect_eval_results.py"
OUT="$REV/reviewer2/03_temporal_fusion/results/svtrv2_ctc"

DATA="${DATA:-/home/vwnascimento/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k}"
SVTRV2_CKPT_DIR="${SVTRV2_CKPT_DIR:-}"

if [[ -z "$SVTRV2_CKPT_DIR" ]]; then
    echo "ERROR: set SVTRV2_CKPT_DIR to the directory containing the trained SVTRv2 checkpoint(s)." >&2
    echo "Example:" >&2
    echo "  SVTRV2_CKPT_DIR=/path/to/svtrv2/checkpoints bash $0" >&2
    exit 2
fi

if [[ ! -d "$SVTRV2_CKPT_DIR" ]]; then
    echo "ERROR: checkpoint directory does not exist: $SVTRV2_CKPT_DIR" >&2
    exit 2
fi

mkdir -p "$OUT"

# F=1 is a checkpoint/protocol sanity check; temporal fusion is not meaningful
# with a single frame, so evaluate it only once.
python3 "$RUNNER" \
    --config "$CFG" \
    --checkpoints "$SVTRV2_CKPT_DIR" \
    --split "$DATA" \
    --output-dir "$OUT" \
    --frames 1 \
    --fusions bayes \
    --keep-going

# Reviewer 2.3 comparison: pre-collapse CTC pooling versus post-decoding
# sequence-level and character-position-level fusion.
python3 "$RUNNER" \
    --config "$CFG" \
    --checkpoints "$SVTRV2_CKPT_DIR" \
    --split "$DATA" \
    --output-dir "$OUT" \
    --frames 3 5 \
    --fusions bayes average logit_average majority char_majority \
    --keep-going

python3 "$COLLECTOR" \
    --input-dir "$OUT" \
    --output-csv "$OUT/fusion_summary.csv" \
    --output-md "$OUT/fusion_summary.md"
