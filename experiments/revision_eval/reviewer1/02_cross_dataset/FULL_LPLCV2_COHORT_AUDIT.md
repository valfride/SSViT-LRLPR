# Full eligible LPLCv2 Poor + Illegible geometry audit

**Status:** read-only audit committed; **no full-cohort LMDB has been built or evaluated**. All earlier 3,000-crop mixed and Poor-only experiments remain frozen and unchanged.

## Why a new population?

The current 3,000-crop mixed LPLCv2 test set was sampled from a **previously sampled, 3,000-crop non-motorcycle Poor LMDB** and an Illegible pool. Its Poor group excludes several vehicle types, and this eligibility differs from the Illegible pool. To evaluate **all eligible annotations**, use the original local `LPLC_dataset/annotations_v2.json` and `LPLC_dataset/images/` directly. Do not take the union of old LMDBs or refill to 3,000.

The user proposed original crop width/height in **[2.5, 3.5]** (rather than the earlier mixed subset's [1.7, 4.0]), plus a crop-size constraint.

### Cross-check against DatasetAcquisition

The committed [DatasetAcquisition example configuration](https://github.com/valfride/DatasetAcquisition/blob/main/config/lrlpr_extension.example.yaml) sets `selection.minimum_crop_width: 8` and `selection.minimum_crop_height: 4`.
Its [curator selector](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/selection.py) supports an **optional lower-only** `minimum_crop_aspect_ratio` but **no upper bound**. The example configuration does not enable that aspect filter. Thus [2.5, 3.5] is **our explicitly stricter LPLCv2 selection**, not a rule copied from DatasetAcquisition. The acquisition size minimum (8x4) is a **permissive integrity floor**, not a guarantee of useful OCR information.

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
