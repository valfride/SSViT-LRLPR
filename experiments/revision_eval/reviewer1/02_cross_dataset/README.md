# Reviewer 1.2 — frozen LPLCv2 Poor-only cross-dataset experiment

**Status:** dataset preparation script ready; external sample selection, crop QA,
and inference remain to be executed on daugman. This protocol must be frozen
before model predictions are examined.

## Scientific scope

- Source: independently acquired LPLCv2 images and local annotations_v2.json.
- Target: **3,000 single-frame plate crops**, LPLCv2 **Poor** (leg = 1).
- Valid labels: **exactly seven** uppercase ASCII alphanumeric characters.
- Exclude occluded annotations (occ = true), missing source images, malformed
  annotations, invalid/out-of-bounds bounding boxes, undecodable images.
- Enforce one distinct **OCR plate label** and one distinct **source image**
  per selected test example. Select with a fixed seed **2026**.
- Keep the plate crop from the supplied bounding rectangle, losslessly as PNG.
  Do not synthetically degrade, fine-tune, or manually cherry-pick hard/easy
  examples. Evaluator provides the original bicubic 96x32 resize.
- Use **F=1 only**, since these are independent stills, not 5-frame tracklets.
- Report overall exact-match RR, character accuracy, CER, sample eligibility,
  and image/camera domain-shift characteristics. These 3,000 plate crops are
  NOT the same unit of analysis as the original 3,000 five-frame LRLPR tracks.

LPLCv2 was provided for academic use under a request-based license. Do not
commit dataset images, crops, annotations, selection manifests, or output
predictions to public repositories without checking the permission conditions.

## Dataset preparation on daugman

From the evaluation code checkout:

    cd ~/doc2025/project1
    git fetch origin
    git switch revision/reviewer-eval-tooling
    git pull --ff-only

Run the new script against the existing *local* dataset:

    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_poor.py \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --output ~/doc2025/LMDB-Datasets/LPLCv2_PoorUnique_TEST_3k \
      --seed 2026 \
      --count 3000

Optional: add the following flag to the SAME command before preparing if
we also want to prevent any OCR-label overlap with our original LRLPR-26
frozen **test** tracklets (a stricter external-identity restriction):

    --exclude-labels-metadata ~/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k

This optional exclusion changes the sampled population and **must be decided
before inference**. Do not use it after observing model predictions. Unique
labels WITHIN LPLCv2 are enforced by default regardless of this option.

On success, the output directory contains:
- data.mdb, lock.mdb, metadata.pkl: F=1-compatible external test LMDB.
- selection_manifest.json, selection_manifest.csv: sample identities, label,
  source filename, camera, original bounding coordinates, crop dimensions.
- dataset_info.json, validation.json: eligibility and integrity summary.
- crops/track_XXXXXX/lr-001.png: lossless original-pixel plate crops.
- contact_sheet.jpg: 60 deterministic selections for **crop geometry QA**.

If the requested output directory already exists, preparation fails rather
than changing the frozen sample selection. No original LPLCv2 source files
are written or modified. The script also verifies all stored LMDB entries.

## Required QA BEFORE model inference

1. Inspect contact_sheet.jpg locally: confirm the axis-aligned box contains
   the complete plate, with no systematic mis-crop.
2. Read dataset_info.json and validation.json, confirm 3,000 selected,
   3,000 unique labels, 3,000 unique source images and that all 3,000 PNG
   records decode. Check camera and time-of-day distribution.
3. If the crop convention is wrong, **do not run inference**; fix the
   extraction method, version the new preparation protocol, and reselect
   before evaluating any model.
4. Record the frozen SHA-256 of selection_manifest.json. Do not exclude
   individual samples based on OCR predictions.

Verify the already-prepared set without modifying it:

    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_poor.py \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --output ~/doc2025/LMDB-Datasets/LPLCv2_PoorUnique_TEST_3k \
      --count 3000 \
      --verify-only

## Frozen-model inference AFTER QA

Run the proposed model with its **existing submitted validation-selected**
checkpoint (do not train, retune, or change data augmentation):

    python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
      --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
      --checkpoints experiments/revision_eval/submitted_model \
      --split ~/doc2025/LMDB-Datasets/LPLCv2_PoorUnique_TEST_3k \
      --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_poor_ours \
      --frames 1 --fusions bayes

Run the SVTRv2-AR baseline from its actual frozen checkpoint directory:

    python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
      --config baselines_configs/SVTRV2_AR_BASELINE.yaml \
      --checkpoints /PATH/TO/FROZEN/SVTRV2_AR/CHECKPOINT_DIRECTORY \
      --split ~/doc2025/LMDB-Datasets/LPLCv2_PoorUnique_TEST_3k \
      --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_poor_svtrv2_ar \
      --frames 1 --fusions bayes

The evaluator creates F1_bayes.json and a prediction CSV. Report RR,
character accuracy, and CER from its structured metrics. Cross-dataset
evaluation is single-image; there is no meaningful temporal-fusion
comparison at F=1.

## Reviewer-response integration after results

Update the manuscript with a clear description of the 7-character,
Poor-only, unique-identity subset, its selection/exclusion counts,
single-image crop construction, F=1 frozen checkpoints, recognition metrics
and limitations (especially legibility-conditioned sample selection).
Then replace the explicit pending R1.2 placeholder in the response letter.

Do not assert overall LPLCv2 generalization beyond the eligible Poor subset.
