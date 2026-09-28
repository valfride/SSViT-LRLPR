#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
REV="$ROOT/experiments/revision_eval"

CFG="$REV/submitted_model/config_snapshot.yaml"
CKPT="$REV/submitted_model"
DATA="${DATA:-/home/vwnascimento/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k}"
OUT="$REV/submitted_model/results"

mkdir -p "$OUT"

for F in 1 3 5; do
    echo "============================================================"
    echo "Submitted model | F=$F | BJP"
    echo "============================================================"

    python3 "$ROOT/test.py" \
        --config "$CFG" \
        --checkpoints "$CKPT" \
        --split "$DATA" \
        --mode val \
        --in_images "$F" \
        --fusion bayes \
        | tee "$OUT/F${F}_bayes.txt"
done
