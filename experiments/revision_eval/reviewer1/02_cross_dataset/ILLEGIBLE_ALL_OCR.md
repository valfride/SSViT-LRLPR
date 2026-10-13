# LPLCv2 Illegible — complete OCR-labelled subset (supplementary study)

**Status:** preparation was attempted on daugman, but stopped after
600 crops when one OCR-annotated Illegible plate had an **out-of-bounds**
bounding box. No completed LMDB was published. The builder now runs a
**complete decoded-image geometry preflight** before attempting to construct
a dataset. The previously frozen Poor subset is unchanged.

## Why this is a separate experiment

The annotation audit of local LPLC_dataset/annotations_v2.json found:

| Item | Count |
| --- | ---: |
| Illegible LP annotations (leg=0) | 5,362 |
| Nonempty Illegible OCR strings | 1,190 |
| Strict seven-character uppercase alphanumeric OCR strings | **1,189** |
| Empty/missing OCR | 4,172 |

The 1,189 seven-character Illegible examples are **not guaranteed to have
visually verifiable ground truth in the provided image**. Some labels may
have been obtained with other imagery, temporal context or other means;
without source provenance, do not treat unreadable-from-image GT as a
verified OCR target. This is a diagnostic *as-annotated* experiment,
not interchangeable with a benchmark of reliably recognizable plates.

## Eligibility: ALL available compatible Illegible OCR annotations

The dedicated script `scripts/prepare_lplcv2_illegible.py` selects
**every** `leg == 0` annotation whose raw `ocr` matches
`^[A-Z0-9]{7}$`, in deterministic source-filename/annotation-index
order. It does **not** random-sample, remove repeated labels, or
remove repeated source photographs. Distinct annotation indices
within the same photograph become separate F=1 samples.

**Completeness is the priority here.** Consequently the primary
all-available version also retains:
- known motorcycle types (recorded in manifest),
- missing vehicle types (recorded as empty; no invented identity),
- source annotations marked `occ=true`.

This differs deliberately from the non-motorcycle, non-occluded
Poor subset. For an optional distinct, motorcycle-excluded
sensitivity version, the script accepts `--exclude-known-motorcycles`,
but **that output must not be described as all 1,189 annotations**.
Do not discard unknown vehicle types under the assumption that they
represent motorcycles.

Crop processing is the same: original bounding-box pixel data,
lossless PNG in LMDB, 96×32 bicubic resizing only in the existing
frozen F=1 evaluator. Every entry has its own virtual track_XXXXXX
directory to keep F=1 sequence grouping correct.

**Reproducibility:** annotation JSON SHA-256, selection manifest SHA-256,
original source filename, annotation index, coordinates, label,
vehicle type, occlusion flag and image dimensions are saved.
No existing LMDB/manifest is overwritten. Output stays local in
the request-restricted dataset workspace.

## Prepare and audit on daugman

From the evaluation checkout:

```bash
cd ~/doc2025/project1
git pull --ff-only

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_illegible.py \
  --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
  --audit-only
```

Expected seven-character label count based on the previous audit: 1,189.
Build the ALL-available variant:

```bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_illegible.py \
  --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleAllOCR_TEST \
  --expect-all 1189

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_illegible.py \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleAllOCR_TEST \
  --verify-only
```

The script refuses to silently drop candidates with missing images,
corrupt image bytes, or invalid/out-of-bounds boxes. The complete
version fails if the total seven-character annotation count differs
from 1,189. Investigate before changing `--expect-all`.

### IMPORTANT: failed first build / bbox geometry preflight

During the first local run, the script stopped after preparing 600
staging crops when annotation index `0` in image
`89626413-aa6e-4f7a-91af-ac8c17f2b153.jpg`
exceeded its source-image dimensions. The atomic staging directory
should have been cleaned up automatically. The error has **not**
established whether the overflow is one pixel or almost the whole
plate; do not guess.

Before rebuilding, run the updated read-only geometry preflight:

```bash
cd ~/doc2025/project1
git pull --ff-only

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_illegible.py \
  --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
  --audit-only \
  --audit-report ~/doc2025/LMDB-Datasets/LPLCv2_Illegible_geometry_audit.json
```

The console now prints geometry summary, and the new JSON contains
all invalid/out-of-bounds annotations with raw box, image size,
clipped visible intersection and percentage of annotation area
outside the source image.

The **default** builder remains strict: any out-of-frame annotation
stops creation, with details from preflight. Only after reviewing
these cases, an **explicit** `--clip-out-of-bounds` option may be
added to the normal build command. It accepts boxes intersecting
the frame only when at least **95% of original bbox area is inside**
(default `--min-inside-fraction 0.95`). The manifest records both
original and clipped boxes, whether clipping occurred, and area
fraction visible. Completely outside, invalid, missing, or too-severely
clipped cases remain blocking. This is a geometric salvage of
visible pixels, **not** correction of ground truth or recovery of
missing characters. Do not enable clipping or loosen the fraction
threshold solely to reach the planned sample count. Assess the
actual geometry audit first.

An Illegible source-image box may itself be incorrect, even if
it falls within bounds. Review source-image QA before treating
OCR labels as visually verified ground truth.

Generated files: `data.mdb`, `metadata.pkl`, lossless `crops/`,
`selection_manifest.json`, `selection_manifest.csv`,
`dataset_info.json`, `validation.json` and `contact_sheet.jpg`.

## Independent visual annotation-target audit BEFORE inference

The generalized QA script now supports this smaller sample with
duplicate labels and repeated source images:

```bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/audit_lplcv2_annotation_quality.py \
  --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
  --manifest ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleAllOCR_TEST/selection_manifest.json \
  --output-dir ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleAllOCR_annotation_QA
```

This creates `qa_samples.csv`, `qa_summary.json`,
`all_pages/page_*.jpg` and `priority_pages/page_*.jpg`.
Automated flags (edge proximity, low contrast, plate aspect)
are **not** evidence of incorrect annotations.

Special attention: if a GT label cannot be independently read from the
supplied frame, classify it as *unverifiable*, not automatically
wrong, and don't use prediction success/failure to decide exclusions.
Illegible instances will naturally have more unreadable crop images.

## Inference (only after QA and provenance check)

If we choose to run the optional as-annotated experiment, use frozen
F=1 CNN–ViT and SVTRv2-AR checkpoints and report the selection limits.

```bash
python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
  --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
  --checkpoints experiments/revision_eval/submitted_model \
  --split ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleAllOCR_TEST \
  --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_illegible_all_ours \
  --frames 1 --fusions bayes
```

Do not combine the Illegible score with Poor without clearly reporting
which ground truths are verifiable and the different inclusion rules.
This is supplementary; it does not replace or change the 3,000-plate
Poor benchmark or its 24.57% as-annotated CNN–ViT result.
