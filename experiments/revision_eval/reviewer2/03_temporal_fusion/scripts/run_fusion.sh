#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
REV="$ROOT/experiments/revision_eval"

CFG="$REV/submitted_model/config_snapshot.yaml"
CKPT="$REV/submitted_model"
OUT="$REV/reviewer2/03_temporal_fusion/results"

DATA="${DATA:-/home/vwnascimento/doc2025/project1/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k}"

mkdir -p "$OUT"

for F in 3 5; do
    for FUSION in bayes average logit_average majority; do
        echo
        echo "============================================================"
        echo "Submitted model | F=$F | Fusion=$FUSION"
        echo "============================================================"

        python3 "$ROOT/test.py" \
            --config "$CFG" \
            --checkpoints "$CKPT" \
            --split "$DATA" \
            --mode val \
            --in_images "$F" \
            --fusion "$FUSION" \
            | tee "$OUT/F${F}_${FUSION}.txt"
    done
done
