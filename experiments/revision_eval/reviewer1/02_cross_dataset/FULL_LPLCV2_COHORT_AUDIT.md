# Full eligible LPLCv2 Poor + Illegible geometry audit

**Status:** read-only audit committed; **no full-cohort LMDB has been built or evaluated**. All earlier 3,000-crop mixed and Poor-only experiments remain frozen and unchanged.

## Why a new population?

The current 3,000-crop mixed LPLCv2 test set was sampled from a **previously sampled, 3,000-crop non-motorcycle Poor LMDB** and an Illegible pool. Its Poor group excludes several vehicle types, and this eligibility differs from the Illegible pool. To evaluate **all eligible annotations**, use the original local `LPLC_dataset/annotations_v2.json` and `LPLC_dataset/images/` directly. Do not take the union of old LMDBs or refill to 3,000.

The user proposed original crop width/height in **[2.5, 3.5]** (rather than the earlier mixed subset's [1.7, 4.0]), plus a crop-size constraint.

### Cross-check against DatasetAcquisition

The committed [DatasetAcquisition example configuration](https://github.com/valfride/DatasetAcquisition/blob/main/config/lrlpr_extension.example.yaml) sets `selection.minimum_crop_width: 8` and `selection.minimum_crop_height: 4`.
Its [curator selector](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/selection.py) supports an **optional lower-only** `minimum_crop_aspect_ratio` but **no upper bound**. The example configuration does not enable that aspect filter. Thus [2.5, 3.5] is **our explicitly stricter LPLCv2 selection**, not a rule copied from DatasetAcquisition. The acquisition size minimum (8x4) is a **permissive integrity floor**, not a guarantee of useful OCR information.

## Verified DatasetAcquisition curator semantics and cross-evaluation fairness

Read directly from the current public `valfride/DatasetAcquisition` `main` branch:

- [`lrlpr_curator/source.py`](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/source.py): `validate_observation_crop` decodes the **already cropped** source image and checks its actual width and height against the configured minimums (8x4 if unspecified). Its ROI/detector metadata is not used to recrop the exported plate.
- [`lrlpr_curator/selection.py`](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/selection.py): each candidate **track** requires at least 2 x (5 images + 2 skipped edges) = 14 observations. The first five after the skipped start are HR; the last five before the skipped end are LR. This is **temporal ordering**, not deterministic native-pixel-size ranking. Optional `minimum_crop_aspect_ratio` is **lower-only**; optional `minimum_hr_lr_median_area_ratio` applies to a **track's HR/LR median areas**, not to an isolated F=1 crop. Neither option is enabled in the tracked example configuration.
- [`lrlpr_curator/common.py`](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/common.py): accepted labels follow the two Brazilian seven-character formats.
- [`lrlpr_curator/curator_run.py`](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/curator_run.py): the example policy enforces *global* unique labels for exported tracks, also preventing exact repeated selected-content fingerprints.
- [`lrlpr_curator/exporter.py`](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/exporter.py): exported track crops are read back and written as JPEG quality 95; no new detector-box crop or resize is performed at export.
- [`scripts/audit_curated_track_image_geometry.py`](https://github.com/valfride/DatasetAcquisition/blob/main/scripts/audit_curated_track_image_geometry.py): the **separate post-hoc audit** defaults to a 2.0 *minimum* native aspect ratio and checks HR/LR median-area ratios of 1.25 and 1.5. These are **audit thresholds**, not evidence that those filters were enforced by the curator or by the official LRLPR-26 train/test partitions.
- The active scenario-specific `config/lrlpr_extension.yaml` is deliberately [Git-ignored](https://github.com/valfride/DatasetAcquisition/blob/main/.gitignore). **We cannot infer the actual production curation thresholds from the public example.** Also, `CompetitionDatasetV2` includes inherited pre-curated V1 tracks; this curator's defaults do not establish the geometry distribution of the published, frozen LRLPR-26 dataset.

### Fairness decision for the new full-source external experiment

**Fairness between the eight frozen recognizers** requires the same image identities, valid label handling, original crop geometry, the **existing identical 96x32 bicubic preprocessing**, no checkpoint retuning, and no model-dependent exclusion. All models must be evaluated on exactly the same fixed output LMDB.

**Training-domain comparability** is a separate objective. The proposed 2.5-3.5 native ratio band is **stricter than DatasetAcquisition's tracked curation configuration** and therefore defines a **geometry-matched sensitivity cohort**, not the unbiased full LPLCv2 population or an exact reconstruction of the official LRLPR-26 curation. The 8x4 floor matches the repository default for basic crop validity, **not** an OCR-quality minimum. Preserve the read-only audit's broader `all_valid_geometry_before_ratio_filter` totals and report rejection rates by Poor/Illegible. Do not call the selected population simply “all LPLCv2.”

Keep every qualifying original annotation including motorcycles, occluded records, missing vehicle types, and repeated source photos/plate strings; the latter are counted and should be disclosed. Imposing the curator's **global label-uniqueness policy** on this external dataset would create a *different*, potentially less representative selected cohort and is not necessary for equal-input comparison. If desired, a separately labelled unique-label sensitivity check may be conducted after the primary evaluation, without replacing it.

Do **not** turn LPLCv2 crops into JPEG 95, enforce HR/LR median-area ratios, or sort F=1 samples temporally just to mimic the multi-frame acquisition curator: those would alter the published input distribution or impose undefined single-frame conditions. Keep LPLCv2 source PNG crops unchanged and note acquisition/export compression as one uncontrolled domain difference.

Before freezing the new dataset we should inspect the **actual LRLPR-26 original LR crop native dimension/aspect histogram** if available; it is more relevant for cross-domain matching than extrapolating the `CompetitionDatasetV2` example configuration.

## Read-only audit on daugman (no inference or crops written)

```bash
cd `/doc2025/project1
git pull --ff-only
AUDIT=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/audit_lplcv2_full_cohort_geometry.py
python3 -m py_compile "$AUDIT"
python3 "$AUDIT" \
  --dataset-root `/doc2025/LMDB-Datasets/LPLC_dataset \
  --min-aspect 2.5 --max-aspect 3.5 \
  --min-width 8 --min-height 4
```

The script reads the **original full annotation source**; it accepts `leg=1` Poor and `leg=0` Illegible, **old Brazilian or Mercosur 7-character OCR format**, a decoded source image and a fully in-bounds original axis-aligned crop, then applies inclusive native aspect and minimum native width/height. It does **not** exclude motorcycles, occlusion, unknown vehicle types, repeated source images, or repeated OCR labels. It never reads recognizer predictions. All included plate crops will receive the unchanged frozen 96x32 bicubic resize during later evaluation (no padding).

The audit prints **only aggregates**, including totals before and after geometry filtering, exclusion counts, original-annotation SHA-256, source/label repeat counts, native width and height percentiles, native aspect percentiles, and how many selected annotations survive progressively stronger width/height cutoffs (24x8, 30x10, 36x12, 42x14, 48x16).

The optional `--output-json` should point **outside the public Git checkout** or be kept excluded from Git.

First compare `retained_after_ratio_and_size_filters` and `joint_size_thresholds_after_aspect` for Poor and Illegible. This lets us decide whether 8x4 is too permissive **based on coverage and original crop resolution**, before running models. A stricter native size minimum would define **a separately named population**; never quietly change the original frozen 3,000-crop dataset.

## After reviewing the audit

1. Freeze the population definition (native ratio 2.5–3.5; approved minimum width/height; exact original annotation file SHA-256).
2. Build a **new** full-cohort F=1 LMDB from the selected original images/labels, without a 3,000-item cap or predictions in selection. Atomically stage and independently validate every crop and metadata entry. Keep an untracked local manifest and count repeated photos/labels.
3. Run all eight **original-validation-selected checkpoints** on the same new LMDB with frozen 96x32 bicubic resizing.
4. Summarize overall, Poor and Illegible full-plate accuracy, partial accuracy, character accuracy and CER. Repeated photo/label records imply some non-independent observations.
5. Treat this as a **separate exploratory diagnostic**. Do not replace the manuscript's existing cross-dataset tables without reviewing new findings. No McNemar test is planned, per user instruction.

**Dataset restrictions:** never commit images, annotation JSON, crops, per-annotation manifests, checkpoints or prediction CSVs to public GitHub.
