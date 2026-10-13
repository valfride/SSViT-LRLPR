# Reviewer 1.2 — frozen LPLCv2 Poor-only cross-dataset experiment

**Status:** the revised vehicle-type-filtered 3,000-sample LMDB was
prepared and integrity-verified on daugman (13 October 2026). Its manifest
hash and 19-label overlap audit have been recorded. The frozen CNN–ViT F=1
inference has now been reported (24.57% RR); the SVTRv2-AR external baseline
and detailed visual crop QA remain outstanding. The previous LMDB is retained.

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

## Label-overlap audit — 13 October 2026

A read-only comparison of OCR labels in the final frozen LPLCv2
`LPLCv2_PoorNonMotoUnique_TEST_3k/selection_manifest.json` with
`CompetitionDataset_LMDB_TEST_3k/metadata.pkl` found:

- External unique OCR labels: **3,000**
- Original LRLPR-26 test unique OCR labels: **3,000**
- Labels occurring in both test sets: **19 (0.63% of external)**

The overlap refers to **matching seven-character strings**, not confirmed
duplicated images, capture sessions, or physical vehicles. The frozen
external selection is **not being changed** in response to this audit.
The manuscript must not claim label-disjoint generalization or verified
absence of all source-image overlap; instead describe the independently
acquired external dataset, the within-set uniqueness constraints, and the
observed 0.63% cross-test-label overlap.

**Still pending before inference:** visual crop QA using the new
`contact_sheet.jpg`; no model predictions have been used to select
or remove external samples.

## Frozen CNN–ViT external F=1 evaluation — 13 October 2026

The **user-reported local evaluator output** for the frozen submitted-model
checkpoint, using the motorcycle-excluded, label-unique 3,000-sample LPLCv2
Poor test set at F=1, is:

| Metric | Result |
| --- | ---: |
| Full seven-character exact-match RR | **24.57% (737/3000)** |
| Character accuracy (corresponding slots) | **72.22% (15166/21000)** |
| Character error rate (Levenshtein edits per GT character) | **27.73% (5824/21000)** |
| At least 6/7 correct | 47.63% (1429/3000) |
| At least 5/7 correct | 66.40% (1992/3000) |
| Incorrect complete plates | 2,263 |
| Old Brazilian layout | 33.95% (202/595) |
| Mercosur layout | 22.25% (535/2405) |

The difference between positional mismatches (21000 - 15166 = 5834)
and reported edit errors (5824) is not necessarily an arithmetic mistake:
the evaluator's slot metric and CER have different error definitions.
Verify using the machine-readable evaluator JSON before manuscript insertion.

The canonical LRLPR-26 F=1 result is 60.60%, corresponding to a
**36.03 percentage point lower RR** on this *different, deliberately
Poor-conditioned* external sample. The difference cannot be attributed
solely to domain shift (sampling and legibility selection differ).
These 3,000 individual plates are not 3,000 five-frame tracklets.

Confidence-threshold summary from the same evaluator:

| Minimum confidence | Retained | Coverage | Exact plate accuracy |
| --- | ---: | ---: | ---: |
| 0.00 | 3000 | 100.00% | 24.57% |
| 0.50 | 2956 | 98.53% | 24.93% |
| 0.70 | 2579 | 85.97% | 28.46% |
| 0.80 | 1987 | 66.23% | 36.13% |
| 0.90 | 1098 | 36.60% | 55.37% |
| 0.95 | 621 | 20.70% | 71.66% |
| 0.98 | 331 | 11.03% | 87.92% |
| 0.99 | 210 | 7.00% | 92.86% |

Mean confidence correct: 0.9466; incorrect: 0.8011; difference:
0.1455. This indicates *discrimination* by score, not necessarily
well-calibrated probabilities: no ECE/Brier/reliability analysis was
reported. Accuracy at confidence thresholds is **selective coverage**,
not all-sample accuracy.

Local machine-generated outputs (not committed; preserve for provenance):

- results/lplcv2_poor_ours/F1_bayes.json
- results/lplcv2_poor_ours/F1_bayes_predictions.csv
- results/lplcv2_poor_ours/matrix_manifest.json

**Remaining:** verify attached contact sheet (including native crops as
small as 9 pixels high); independently check metrics JSON/model checkpoint
provenance; run frozen SVTRv2-AR baseline on the *same manifest* at F=1;
report both models consistently in the manuscript and response letter.
The 19/3000 overlap with LRLPR-26 test labels (0.63%) was audited but
did not change the frozen selection. Report the restriction to
vehicle-type-annotated non-motorcycle Poor samples.

## Brazilian/Mercosur layout-composition check — 13 October 2026

The canonical V1/LRLPR-26 3,000-tracklet F=1 evaluation JSON at
`experiments/revision_eval/submitted_model/results/F1_bayes.json`
records **600 old Brazilian plates (20.00%)** and **2,400 Mercosur
plates (80.00%)**. In comparison, the current frozen external 3,000
LPLCv2 Poor plate crops comprise **595 Brazilian (19.83%)** and
**2,405 Mercosur (80.17%)**. The class composition differs by
only **five plates** (0.17 percentage points for either class).

The observed external CNN–ViT RR is **737/3000 = 24.5667% (24.57%)**.
A two-stratum prevalence-standardized RR, weighting the *observed*
LPLCv2 per-layout accuracy by the **exact V1 20%/80% layout mix**, is:

`0.20 * (202/595) + 0.80 * (535/2405) = 0.24586174`
(or **24.59% RR**).

This is only **0.02 percentage points** above the
observed **24.57%**; neither metric requires changing any
individual sample. Given the near-identical layout mix, no
post-inference re-sampling/replacement of five plates is warranted.
The standardization corrects only layout proportions, *not* other
dataset or difficulty differences. Maintain the original SHA-256 and
the original per-format and overall metrics as primary outcomes.

Note the layout-specific performance reversal: the original LRLPR-26
F=1 RR was **41.33%** old Brazilian (248/600), **65.42%** Mercosur
(1570/2400); the external Poor subset was **33.95%** old Brazilian
(202/595), **22.25%** Mercosur (535/2405). Interpret cautiously as
legibility, acquisition conditions, crop scale, and plate format may be
confounded; no isolated format-causal conclusion follows.


## Native-resolution and cross-legibility sample audit (pending local run)

**Motivation:** an external *Poor-legibility* label is **not** a
native-low-resolution ground-truth label. LPLCv2 covers Perfect, Good,
Poor, and Illegible plates with OCR text available on many plates
outside Poor. The official README reports 36,414 OCR-annotated LPs
overall; the local number of strict seven-character OCR labels in
each class has **not** yet been measured. The LPLCv2 paper defines
Poor by substantial distortions while still allowing recognition;
it does not constrain native bounding-box pixel dimensions.

Our external LPLCv2 result of 24.57% exact RR therefore supports
**zero-shot transfer to selected Poor-legibility images**, but cannot
be used by itself to claim performance on naturally low-resolution
LPs. The model forcibly resizes each crop to 96x32, including large
native crops, which must be distinguished from native acquisition LR.

An independent, **read-only** local QA script has been added at
`scripts/audit_lplcv2_samples.py`; it does not change the frozen
dataset or use model predictions. On daugman:

    cd ~/doc2025/project1
    git pull --ff-only
    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/audit_lplcv2_samples.py \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --manifest ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k/selection_manifest.json \
      --output-dir ~/doc2025/LMDB-Datasets/LPLCv2_quality_audit_20261013 \
      --seed 2026

It writes `audit_summary.json`, including class-wise counts of OCR
strings valid under our exact `[A-Z0-9]{7}` protocol; native
bounding-box width/height distributions by legibility; and size
statistics for the frozen 3,000-plate subset (e.g., native width
above/below 96 px and height above/below 32 px).

It also creates seven contact sheets: `legibility_perfect.jpg`,
`legibility_good.jpg`, `legibility_poor.jpg`,
`legibility_illegible.jpg`, `selected_random.jpg`,
`selected_smallest_heights.jpg`, and
`selected_largest_widths.jpg`. Each sample shows its full source
image with the annotated plate box, original-pixel plate crop
(display-scaled for human inspection), and the model's bicubic 96x32
version (displayed 2x). All outputs must stay local or be shared
only as allowed by the dataset's request-based license.

**Interpretation warning:** plate dimensions alone cannot establish
natural degradation: blur, compression, and lighting require
visual assessment. Also, requiring a recorded non-motorcycle
`car.type` removed 3,450 eligible seven-character Poor plate
annotations with missing vehicle types; this type-known restriction
may introduce further selection bias. If we expand evaluation to
Perfect/Good or geometry-stratified subsets, define it as a clearly
labeled **supplementary** analysis, preserving the original manifest
and its 24.57% result without post-hoc editing.


## Completed native plate-size and OCR-availability audit — 13 October 2026

Local read-only audit on daugman, using `annotations_v2.json` and the
unchanged selected manifest SHA-256
`8a08180604d88b652c6d915853830b4b1fd0a9ff6bcceace597d8a57f199d408`:

| Legibility | LP annotations | Valid seven-character OCR labels | Median bbox width (px) | Median bbox height (px) | Bbox width <=96 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Perfect | 18,425 | 18,416 | 98 | 38 | 8,925 (48.4%) |
| Good | 10,180 | 10,142 | 68 | 26 | 8,263 (81.2%) |
| Poor | 7,520 | 6,943 | 44 | 16 | 7,062 (93.9%) |
| Illegible | 5,362 | 1,189 | 31 | 13 | 5,327 (99.3%) |

**Selected frozen 3,000 non-motorcycle Poor plates:**

- Median original annotated bbox: **45 × 16 px**
- Native crop width <=96 px: **2,812 / 3,000 = 93.73%**
- Native crop width <=48 px: **2,106 / 3,000 = 70.20%**
- Native crop height <=32 px: **2,769 / 3,000 = 92.30%**
- Width >96 px: **188**; height >32 px: **231**
- All audit-preview decoding/crop-geometry failure lists were **empty**,
  but this does **not** replace human visual QA.

These numbers establish that plate bounding boxes are generally
small in the provided *source photographs*, and the 96×32 inference
resize therefore enlarges many of them. They do **not**, without
additional provenance, prove original camera sensor resolution or
rule out upstream image resampling. Nor does bbox size identify the
dominant perceptual defect (blur, glare, compression, etc.).

**Important scope:** OCR text is not restricted to Poor. Good has
10,142 strict seven-character labels and frequently small plate
crops; a separate, preregistered Good sample or cross-class
native-size-matched test may be useful as supplementary analysis.
Do not retroactively substitute one for the already evaluated
frozen Poor sample.

Dataset-count discrepancy requiring provenance check: the audit
counted **36,757 nonempty OCR fields** across all four classes,
versus **36,414** 'LP Text Available' in the public dataset README.
These counts are not identical definitions and may reflect dataset
version or filtering. Do not claim they agree without investigation.
Locally there are **36,690 strict seven-character OCR strings**
across the four classes.

### Exploratory zero-shot accuracy by native plate dimensions

A read-only script, `scripts/analyze_accuracy_by_native_size.py`,
joins the unchanged manifest with the existing CNN–ViT prediction CSV
by track ID. It validates all 3,000 labels and gives exact-match RR,
character-slot accuracy, and CER in disjoint native-width/native-height
bins; results are **descriptive post-hoc subgroup analyses**, not
evidence of causal impact from size alone.

    cd ~/doc2025/project1
    git pull --ff-only
    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/analyze_accuracy_by_native_size.py \
      --manifest ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k/selection_manifest.json \
      --predictions experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_poor_ours/F1_bayes_predictions.csv \
      --output ~/doc2025/LMDB-Datasets/LPLCv2_quality_audit_20261013/accuracy_by_native_size.json

The original frozen test selection and predictions are read-only.
Keep the output within the restricted local dataset workspace.


## Measured native-size accuracy on frozen LPLCv2 Poor test — 13 October 2026

The read-only local `analyze_accuracy_by_native_size.py` was run on
the **unchanged** frozen 3,000-sample selection and submitted CNN–ViT
F=1 prediction CSV; overall accuracy reconciled at **737/3000 =
24.567%**. Selected annotation geometry in both axes <=96×32
comprised **2,768/3,000 = 92.27%**. This is an image-annotation
dimension measure, *not* proof of untouched sensor resolution.

### Accuracy by original bounding-box width

| Native width (px) | Plates | Exact RR | Character-slot accuracy | CER |
| --- | ---: | ---: | ---: | ---: |
| <=32 | 39 | 23.08% | 60.07% | 39.93% |
| 33–48 | 2,067 | 23.17% | 72.12% | 27.84% |
| 49–64 | 498 | **34.94%** | 77.51% | 22.43% |
| 65–96 | 208 | 19.71% | 67.38% | 32.49% |
| >96 | 188 | 18.09% | 67.17% | 32.83% |

### Accuracy by original bounding-box height

| Native height (px) | Plates | Exact RR | Character-slot accuracy | CER |
| --- | ---: | ---: | ---: | ---: |
| <=12 | 147 | 12.93% | 55.01% | 44.90% |
| 13–16 | 1,456 | 19.30% | 70.10% | 29.86% |
| 17–24 | 993 | **35.95%** | 79.63% | 20.34% |
| 25–32 | 173 | 23.70% | 69.20% | 30.55% |
| >32 | 231 | 16.88% | 66.91% | 33.09% |

**Interpretation:** apparent accuracy is not monotonic in bounding-box
dimensions: medium-height / medium-width samples performed better
than the very small and larger-crop groups. This is an exploratory
**post-hoc** finding and may reflect plate layout, camera/source,
illumination, distortion, selection bias, and other confounders.
Do not infer causal dependence or advocate filtering larger plates
based on these test results.

**Next diagnostic (read-only):** the analysis script now also emits
`by_layout` and `by_time_of_day` tables with native-width/height
bin counts and accuracies. Run it again with a **new** output filename
(e.g., `accuracy_by_native_size_stratified.json`) rather than
overwriting the old output; examine whether the nonmonotonic
pattern persists separately within Brazilian/Mercosur classes. The
sample and model prediction CSV must remain unchanged.


## Visual QA of native-versus-model contact sheet — 13 October 2026

A user-shared `selected_random.jpg` contact sheet (24 selected
examples, with source photograph, original-pixel crop and bicubic
96×32 input) was visually inspected following the native-size audit.

**Observed:** many plates really are small and degraded *within the
provided source photographs*, showing night exposure issues, blur,
strong color casts, low contrast and blockiness. Examples include
`ANV5648` (41×17 px), `AHQ9H77` (43×15 px),
`BCJ0129` (41×14 px; magenta illumination) and
`AAF9666` (70×23 px; severely dark/noisy). Larger samples such as
`BBH6J46` (101×40 px) may remain hard due to very low contrast.
No assertion about untouched sensor-native files can be made from
a contact sheet alone.

**Possible annotation/cropping problem:** the random sample
`AYL8217` (34×12 px; source filename prefix `077c8ab4`)
appears as an almost featureless gray crop with a dark line, rather
than a visible seven-character plate. It needs a magnified source-image
inspection with its original annotation rectangle overlaid. This is
a qualitative QA **flag**, not a verified misannotation or permission
to change the frozen test subset. The montage cannot establish how
many other plates have incorrect coordinates.

**Protocol:** preserve the complete frozen 3,000-plate manifest,
its existing predictions and its reported 24.57% RR. Inspect flagged
source/annotation cases and report any annotation-quality limitation
transparently. If a separate quality-controlled evaluation is
justified, define objective image-only exclusion rules and report
it explicitly as a secondary sensitivity analysis; never silently
remove hard examples after reading OCR predictions. The crop QA
has not yet been completed for all 3,000 samples.


## Annotation-target correctness concern — 13 October 2026

After the first native-resolution contact-sheet audit, the source
photograph associated with the selected annotation "AYL8217" was
inspected with its original red annotation bounding rectangle.

**Observed:** the box sits on a partially visible vehicle at the
**top edge of the source image**, and its crop looks like an almost
uniform gray strip with a dark border. A prominent silver SUV
occupies the foreground, but **its plate is NOT inside the
annotated box**. Because the source contains multiple vehicles,
this does **not** demonstrate that the OCR label belongs to the
foreground SUV. It could be a legitimate annotation of a different,
partially visible vehicle, with ground truth unavailable from
this particular frame; it could also be a mismatched/incorrect
annotation. Only a source-aware review can distinguish these cases.

**Scientific consequence:** the 24.57% external CNN–ViT exact-match
result is a reproducible output on the *as-annotated* frozen subset,
but **annotation-target quality is not yet independently verified**.
Do not present it as a fully quality-audited external accuracy or
conclude that the errors arise only from natural degradation.

To assess potential prevalence, new read-only script
\`scripts/audit_lplcv2_annotation_quality.py\` produces review pages
for **all 3,000 samples**, plus pages of priority cases selected
without looking at model predictions. Each item displays: a full
source image with the original rectangle, a detailed crop-neighborhood
view, and the extracted image patch. The outputs never alter or
overwrite the frozen LMDB/manifest/predictions.

    cd ~/doc2025/project1
    git pull --ff-only
    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/audit_lplcv2_annotation_quality.py \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --manifest ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k/selection_manifest.json \
      --output-dir ~/doc2025/LMDB-Datasets/LPLCv2_annotation_QA_20261013

Outputs:
- \`qa_summary.json\`: objective priority-flag counts and manifest SHA;
- \`qa_samples.csv\`: 3,000 rows, with blank \`manual_status\`
  and \`manual_note\` columns for independent manual inspection;
- \`priority_pages/page_*.jpg\`: edge-adjacent, low-contrast or unusual
  aspect-ratio crops prioritized for manual review;
- \`all_pages/page_*.jpg\`: **all** 3,000 instances for denominator-wide
  visual review (24 per image).

**Priority rules are diagnostic flags, NOT exclusion criteria.**
Edge-adjacent plates can be legitimate; dark/low-texture samples can
be genuinely Poor. A reviewer should distinguish (i) correctly
localized but difficult plates, (ii) partial/out-of-frame plates,
(iii) incorrect/mismatched box, and (iv) unverifiable OCR GT. The
annotated plate may refer to a **different vehicle** from the most
prominent one in the photograph. Do not reconstruct or substitute
another vehicle's visible plate as the target.

Record the number of reviewed, suspect, and confirmed invalid
annotations *before* any secondary quality-controlled evaluation.
Retain the **original 3,000-sample results unaltered**, explicitly
labeling any future filtered analysis as an additional, post-hoc
sensitivity check. Data assets and review sheets remain local,
subject to LPLCv2 redistribution restrictions.


## Full 3,000-sample automated annotation-QA audit — 13 October 2026

Read-only audit of frozen manifest SHA-256
8a08180604d88b652c6d915853830b4b1fd0a9ff6bcceace597d8a57f199d408:

| QA field | Count |
| --- | ---: |
| Source images successfully decoded | 3,000 |
| At least one heuristic flag | 501 (16.70%) |
| No heuristic flags | 2,499 (83.30%) |
| Low grayscale standard deviation (below 16) | 339 |
| Bounding box within 3.5% of source-image edge | 162 |
| Annotated bbox aspect outside [1.7, 4.0] | 24 |
| Review pages covering all 3,000 | 125 |
| Priority-flagged pages | 21 |

The flag types overlap (525 flag occurrences across 501 unique
images). A flagged example is NOT automatically a ground-truth
error; unflagged examples are NOT necessarily correct either.
These numbers cannot be described as an annotation-error rate.

### Planned blinded manual annotation-quality review

Use the new script scripts/prepare_blind_annotation_review.py to
take a seeded, random, 100-example review sample from the 501
flagged and 200-example review sample from the 2,499 unflagged.
It generates local review contact sheets and blind_review.csv
with manual_status/manual_note fields; no model predictions
or model correctness are read.

    cd ~/doc2025/project1
    git pull --ff-only
    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_blind_annotation_review.py \
      --qa-dir ~/doc2025/LMDB-Datasets/LPLCv2_annotation_QA_20261013 \
      --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
      --output-dir ~/doc2025/LMDB-Datasets/LPLCv2_manual_QA_pilot_20261013 \
      --flagged 100 --unflagged 200 --seed 2026

Review flagged_sample_pages and unflagged_sample_pages,
then assign exactly one manual_status to each row:
valid_target, unverifiable_text, wrong_box_or_target,
wrong_ocr_label, uncertain. A correctly localized but
independently unreadable plate is distinct from a wrong box
or proven OCR mislabel.

After all 300 human classifications, run the read-only summary:

    python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_blind_annotation_review.py \
      --output-dir ~/doc2025/LMDB-Datasets/LPLCv2_manual_QA_pilot_20261013 \
      --summarize

The output gives descriptive stratum-weighted proportions,
not statistically certified counts or confidence intervals.
Preserve all 3,000 frozen samples and existing predictions,
including the original 24.57% as-annotated result. If
material annotation failures are confirmed, extend manual
QA and report sensitivity analyses separately, without
retroactive undocumented sample exclusion.

Keep all restricted sample imagery and manual reports local.

## Separate complete Illegible OCR subset (pending)

For a supplementary *OCR-labelled Illegible* (`leg=0`)
evaluation, see [ILLEGIBLE_ALL_OCR.md](ILLEGIBLE_ALL_OCR.md). The
local annotation audit found 1,189 exact seven-character strings,
of which **284 were the placeholder `UNKNOWN`**. The correct
structurally valid Brazilian/Mercosur subset contains **905 plate
annotations (890 distinct labels)**. The only out-of-bounds
annotation was `UNKNOWN`, so the 905 retained plates pass
bounding-box geometry preflight. Unlike the 3,000-plate Poor
subset, this version does not deduplicate labels/sources or
silently exclude motorcycles, occlusions or missing vehicle types.
The corrected `--require-plate-format` LMDB
(`LPLCv2_IllegibleBrazilianOCR_TEST`) has now been built and
`--verify-only` succeeded: **905 decodable records, 905 unique
annotation IDs, 890 unique labels, 891 source photographs, 25
known motorcycles, 318 missing vehicle types, zero occluded and
zero clipped crops**. The automated source-image annotation
QA has now also run successfully on all 905: **300 flagged** (33.15%)
and **605 unflagged**; flags comprise 184 low-contrast, 74
frame-boundary, and 63 aspect-ratio occurrences with overlaps.
The audit manifest SHA-256 is
`7e74f46d991db9835739e7b99b5c01f1be6da7b21ba1b50b2c9bef848b7dfb0e`.
**Human review of annotation targets remains pending** before
using these labels as verified OCR ground truth. Do not present visually
unverifiable labels as confirmed OCR ground truth or revise the
original frozen Poor benchmark after observing these samples.


## Planned LPLCv2 mixed-difficulty F=1 set (905 Illegible + 2095 Poor)

See [MIXED_POOR_ILLEGIBLE_3K.md](MIXED_POOR_ILLEGIBLE_3K.md).
The authors explicitly say in Section III of
[Wojcik et al. (IJCNN 2026)](https://arxiv.org/pdf/2604.08741)
that OCR labels were automatically proposed and manually reviewed,
that plate text is omitted where human validation is unreliable,
and that camera zoom made some otherwise Illegible plates
verifiable. They also cross-validated OCR against vehicle data.
This is independent documentary support for the intentional
presence of text annotations in the Illegible category.

The new mixed subset is **exploratory and pending preparation**:
retain all 905 Brazilian-format OCR-labelled Illegible annotations
and draw 2,095 from the **previously frozen** 3,000-image Poor test
LMDB with fixed seed 2026, after excluding cross-class duplicate
source photographs/plate strings. The mixture is exactly 3,000
**plate crops**, potentially fewer distinct source photographs,
and has distinct inclusion criteria by class. Parent LMDBs and
frozen results remain unchanged. The builder validates parent
selection SHA-256s and will refuse the build if insufficient
Poor candidates remain. Run the dry-run from the linked protocol
and inspect counts *before* construction or inference.


### Frozen model-input aspect-ratio consistency

Both submitted CNN–ViT and SVTRv2-AR F=1 validation wrappers
**directly bicubic-resize** all decoded crops to **96×32 (3:1)**,
with no padding. The mixed-set builder now validates both frozen
configs, measures original crop aspect-ratio distributions by
legibility, and simulates the 96×32 bicubic resize for all mixed
LMDB records during verification. It does not change the original
lossless crop bytes. Native source crops need not be exactly 3:1;
the resizing can stretch non-3:1 crops. See the dedicated
[MIXED_POOR_ILLEGIBLE_3K.md](MIXED_POOR_ILLEGIBLE_3K.md)
aspect-ratio protocol. Any aspect-preserving letterbox
evaluation would be a separate preprocessing variant.
