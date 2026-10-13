# Reviewer 1.2 — frozen LPLCv2 Poor-only cross-dataset experiment

**Status:** the revised vehicle-type-filtered 3,000-sample LMDB was
successfully prepared on daugman (13 October 2026). The motorcycle-excluded
selection and SHA-256 are now recorded below. **Visual crop QA, overlap audit,
and frozen-model inference are still pending**. The previous LMDB is retained.

## Scientific scope

- Source: independently acquired LPLCv2 images and local annotations_v2.json.
- Target: **3,000 single-frame plate crops**, LPLCv2 **Poor** (leg = 1).
- Valid labels: **exactly seven** uppercase ASCII alphanumeric characters.
- Exclude occluded annotations (occ = true), missing source images, malformed
  annotations, invalid/out-of-bounds bounding boxes, undecodable images.
- Require a recognized non-motorcycle `anns[].car.type`. Motorcycle variants,
  missing types, and unrecognized types are excluded; the script reports each
  normalized type and exclusion count before selection. This is a strict
  metadata-based class filter, not a learned motorcycle detector.
- Enforce one distinct **OCR plate label** and one distinct **source image**
  per selected test example. Select with a fixed seed **2026**, drawing random
  distinct OCR labels first and then picking a random viable image per label
  (without reusing source images).
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

First, **audit available vehicle categories**, without writing data:

    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_poor.py \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --audit-only

Inspect `type_*` entries, `motorcycle_excluded`, `missing_vehicle_type`,
`unrecognized_vehicle_type`, and `unique_labels_before_image_decode`.
There must be at least 3,000 valid unique labels before we proceed.

Then create the **new**, separate filtered dataset:

    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_poor.py \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --output ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
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
      --output ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
      --count 3000 \
      --verify-only

## Frozen-model inference AFTER QA

Run the proposed model with its **existing submitted validation-selected**
checkpoint (do not train, retune, or change data augmentation):

    python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
      --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
      --checkpoints experiments/revision_eval/submitted_model \
      --split ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
      --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_poor_ours \
      --frames 1 --fusions bayes

Run the SVTRv2-AR baseline from its actual frozen checkpoint directory:

    python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
      --config baselines_configs/SVTRV2_AR_BASELINE.yaml \
      --checkpoints /PATH/TO/FROZEN/SVTRV2_AR/CHECKPOINT_DIRECTORY \
      --split ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
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

## Vehicle-type audit and classification correction — 13 October 2026

The first type-filter audit (before the two corrections below) reported
3,330 accepted Poor annotations with 3,214 unique OCR labels and 3,273
unique source images. It also reported 3,450 missing vehicle types and
138 motorcycle-flagged annotations.

This revealed two classification errors in the script's strict
non-motorcycle allowlist:

- `MICRO ONIBUS` (23 annotations): should be accepted as a microbus,
  alongside `MICROONIBUS`.
- `MOTOR CASA` (1 annotation): should be accepted as a motorhome, not
  rejected by a naive substring search for `MOTO`.

Both cases are now corrected: the motorcycle check matches explicit normalized
vehicle-type values, not arbitrary substrings. Genuine `MOTOCICLETA` (124
annotations) and `MOTONETA` (13) remain excluded. Unknown/missing types still
remain excluded rather than guessed.

**Rerun `--audit-only` with the updated script before generating the new
LMDB.** Expected preliminary accepted annotation count is 3,354, assuming the
listed type counts describe all otherwise eligible cases; verify the actual
unique-label count from the new audit output. The 3,450 missing types represent
a substantial selection restriction, so the results must be described as
recognition on the *vehicle-type-annotated, non-motorcycle Poor subset* rather
than on all Poor LPLCv2 plates.

## Historical unfiltered preparation outcome — 12 October 2026

Before the vehicle-type exclusion was added, the preparation script was
successfully executed on `daugman` against
`~/doc2025/LMDB-Datasets/LPLC_dataset/annotations_v2.json`.
**These numbers and the manifest SHA-256 refer to the old, unfiltered selection**,
which is preserved at
`~/doc2025/LMDB-Datasets/LPLCv2_PoorUnique_TEST_3k`.
The old selection should **not** be used for the motorcycle-excluded evaluation:

| Audit field | Value |
| --- | ---: |
| Poor annotated plate instances | 7,520 |
| Invalid/incompatible OCR labels | 577 |
| Valid seven-character OCR labels | 6,943 |
| Excluded as occluded | 2 |
| Eligible candidates before decode | 6,941 |
| Unique eligible OCR labels | 6,780 |
| Unique eligible source images | 6,666 |
| Selected plate crops | **3,000** |
| Unique selected OCR labels | **3,000** |
| Unique selected source images | **3,000** |
| Source-image conflicts skipped while selecting | 57 |
| Distinct cameras in selection | 422 |
| Rain-flagged samples | 73 |
| Faulty-camera-flagged samples | 211 |
| Crop width range | 17–226 pixels |
| Crop height range | 10–104 pixels |

Time of day in selected subset: afternoon 940, evening 874, morning 769,
night 417. Acquisition day: 0: 879, 1: 1,157, 2: 964.

**Frozen selection manifest SHA-256:**

`5a2ddbfff8a92309d50094b9097b42338ee825b18edc9a7b2d38e4d18610f5d0`

The sample was generated **without** `--exclude-labels-metadata`; overlap
with LRLPR-26 labels must still be audited independently before the results
are characterized as cross-dataset identity-disjoint. A cross-label overlap
audit does not automatically invalidate a cross-dataset evaluation, but it
changes the claims that may be made. Do not silently change the frozen
selection after inspecting predictions.

**Pending:** run the updated vehicle-type candidate audit; create the new
`LPLCv2_PoorNonMotoUnique_TEST_3k` set; record its new audit counts and
manifest SHA-256; visually inspect its contact sheet, audit label overlap,
then run F=1 frozen-model inference. No predictions were used in choosing
the motorcycle exclusion criterion. Dataset crops and annotation manifests
stay outside the public repository.

## Confirmed motorcycle-excluded eligibility audit — 13 October 2026

The corrected vehicle-type filter was run on daugman with `--audit-only`.
Observed counts (before final image decoding/crop construction):

| Count | Value |
| --- | ---: |
| Poor plate annotations | 7,520 |
| Valid seven-character labels | 6,943 |
| Accepted annotated non-motorcycle plates | 3,354 |
| Unique accepted OCR labels | 3,236 |
| Unique accepted source images | 3,296 |
| Motorcycle types excluded | 137 |
| `MOTOCICLETA` / `MOTONETA` | 124 / 13 |
| Missing vehicle types excluded | 3,450 |
| Microbuses (`MICRO ONIBUS`) now accepted | 23 |
| Motorhomes (`MOTOR CASA`) now accepted | 1 |

**Next step:** generate the new `LPLCv2_PoorNonMotoUnique_TEST_3k` with
seed 2026, inspect its contact sheet, and record its new SHA-256 before
inference. The count of eligible items is not the count finally selected.
Do not use the older unfiltered manifest hash for this dataset.


## Frozen non-motorcycle test set — 13 October 2026

**Status:** successful 3,000-sample preparation from the corrected type
filter. Inference has **not** yet been run or used to revise selection.

- LMDB: `~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k`
- Seed: `2026`
- Selected: **3,000** plate crops, **3,000** distinct OCR strings,
  **3,000** distinct source images
- Source-image conflicts skipped: **35**
- Cameras: **383**
- Time: morning **799**, afternoon **890**, evening **927**, night **384**
- Acquisition day: 0: **610**, 1: **1,310**, 2: **1,080**
- Rain-flagged: **14**; faulty-camera flagged: **151**
- Crop dimensions (native pixels): width **24–283**, height **9–104**

Selected annotation vehicle-type distribution:

| Normalized vehicle type | Selected plates |
| --- | ---: |
| AUTOMOVEL | 2,420 |
| CAMINHONETE | 201 |
| CAMIONETA | 182 |
| UTILITARIO | 101 |
| ONIBUS | 38 |
| CAMINHAO | 28 |
| MICRO ONIBUS | 21 |
| SEMI REBOQUE | 5 |
| CAMINHAO TRATOR | 3 |
| MOTOR CASA | 1 |
| **Total** | **3,000** |

**Frozen selection manifest SHA-256:**

`8a08180604d88b652c6d915853830b4b1fd0a9ff6bcceace597d8a57f199d408`

This hash replaces the old **unfiltered** sample's hash as the intended
external evaluation set. The old selection and its hash are preserved above
only to document why the vehicle-type filter was introduced.

**LMDB integrity verification (13 October 2026): PASS.** The
`--verify-only` command reported:

- `verified_samples: 3000`
- `unique_labels: 3000`
- `unique_source_images: 3000`
- `all_legibility: "Poor (leg=1)"`
- `lmdb_decoded: 3000`
- `split: "test"`, `frames: 1`

This verifies LMDB decoding and manifest uniqueness, **not** visual crop
completeness. Crop contact-sheet inspection and LRLPR-26 label-overlap
audit remain pending before frozen-model inference.

**Pending checks:** inspect `contact_sheet.jpg`, especially the original
9-pixel-high plates and atypical aspect ratios, to make sure the annotations
contain complete plate regions. Verify the generated LMDB if needed with
`--verify-only`, and audit OCR-label overlap with the LRLPR-26 canonical
test set. Overlap checking should **not** silently alter this already frozen
selection. Proceed to F=1 frozen-checkpoint inference only after geometric
QA passes. Since missing car.type records were excluded, only claim
performance for the type-annotated non-motorcycle Poor subset, not all Poor
LPLCv2 plates.
