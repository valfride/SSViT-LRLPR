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

## Important acquisition-time 20%-padding verification

The user correctly identified an earlier **acquisition** step that is
not visible from the curator's selection configuration alone.

- [`config/intelbrasPTZvip.yaml`](https://github.com/valfride/DatasetAcquisition/blob/main/config/intelbrasPTZvip.yaml)
  sets `camera.args.padding_ratio: 0.2`.
- [`cameras/rtsp_pipeline.py`](https://github.com/valfride/DatasetAcquisition/blob/main/cameras/rtsp_pipeline.py):
  `process_results` first calls
  `crop_xywh(frame, detector_xywh, 0.0)` for `lp_type()`
  classification, then `append_track` calls
  `crop_xywh(frame, detector_xywh, self.padding_ratio)` for
  the **actually saved** track crop.
- In `crop_xywh`, `px=w*padding` and `py=h*padding`,
  and `floor/ceil` box coordinates add **20% of original
  bounding-box width/height on each of the two corresponding
  sides**, before source-frame clipping. Unclipped geometry is
  approximately `1.4*w x 1.4*h`, hence **1.96 times** the
  original detector-box area.
- The acquisition's `lp_type()` checks the **tight (zero-padding)
  detector crop** against car/motorcycle aspect-range constraints;
  `crop_xywh` also requires `min_crop_width: 24`,
  `min_crop_height: 8` and `min_crop_area: 192` **by default**
  (these are separate from the curator's later 8x4 floor).
  Check the actual acquisition runtime's overrides before
  calling these default minimums enforced production values.
- Acquisition saves the initial `lp_*.jpg` crops at **JPEG
  quality 90**; the later
  [`lrlpr_curator/exporter.py`](https://github.com/valfride/DatasetAcquisition/blob/main/lrlpr_curator/exporter.py)
  re-encodes the selected crops at **JPEG quality 95**.

**Implication:** the original LPLCv2 builders crop the tight
floor/ceil axis-aligned envelope of each quadrilateral label and
write lossless PNGs. They **do not add the 20%-per-side acquisition
padding**, so the plate occupies a different fraction of the
96x32 input tensor. This may affect architecture comparisons
under domain shift. Do not equate the LPLCv2 annotation rectangle
with the acquisition detector's original bbox without explaining
the proxy. Moreover, the public acquisition code relates to a
particular acquisition pipeline and does **not prove that every
crop in the frozen competition LRLPR-26 train/test set** went
through it, especially the inherited dataset records.

The read-only geometry audit now reports BOTH tight native-box
dimensions and a **diagnostic 20%-per-side expansion**, including
padded dimensions after frame clipping, how often padding would
clip at the frame boundary, how often the complete expanded box
retains a 2-pixel source-frame margin, and how many tight boxes
pass the camera's default 24x8/192 pixel crop check. The
`--min-aspect` and `--min-width` filters still refer to the
**tight original annotation rectangle**, not the padded one.
**No image pixels are altered or new LMDB built in the audit.**

After inspecting the totals, a **new, explicitly named padded
LPLCv2 LMDB** can be built for an acquisition-context
sensitivity comparison with all eight frozen models, alongside
the existing unpadded LPLCv2 results. Both the padding rule
and any clipping/2-pixel-margin exclusion decisions must be
fixed for **all architectures** before inference. Avoid
replacing the original benchmark with a padded-only result
chosen after observing which model performs better.

## Actual production configuration supplied by the user (13 October 2026)

The production config from peixoto has `images_per_role=5`,
`temporal_edge_skip=2`, `require_fully_in_bounds=true`,
`minimum_visible_fraction=1.0`, `minimum_border_margin_px=2`,
`minimum_crop_width=8`, `minimum_crop_height=4`,
`require_crop_matches_requested_roi=true`,
`maximum_sequence_gap=3`, `maximum_frame_gap=30`,
`minimum_window_separation=0`, and
`minimum_hr_lr_area_ratio=1.1`. The label policy is
`mode=unique_only`. No native aspect-ratio bounds were supplied.

**Critical code/config verification gap:** in the current public
`DatasetAcquisition/main` implementation,
`select_lr_hr()` reads
`minimum_hr_lr_median_area_ratio` instead of the production
`minimum_hr_lr_area_ratio` key. The latter is not found as an
enforced key in the public implementation. Likewise, the actual
selection path does not appear to read `require_fully_in_bounds`,
`minimum_visible_fraction`, `require_crop_matches_requested_roi`,
`maximum_sequence_gap`, `maximum_frame_gap`, or
`minimum_window_separation`. The
`minimum_border_margin_px=2` field **is** passed to
`compute_roi_geometry()` and affects its `roi_in_bounds`
flag, but the selector does not itself reject based on that
flag. ROI warning flags in the review pipeline should not be
confused with mandatory exclusion.

This analysis concerns the public branch currently accessible in
GitHub; the user's installed production revision on peixoto has
**not** been examined. Thus do **not** claim that all configured
ROI, temporal-gap, or HR/LR-area checks were enforced on the
frozen LRLPR-26 crops without confirming the deployed code and
review workflow. Matching configuration fields alone is
insufficient to establish matched dataset curation.

A safe read-only check on peixoto, without printing credentials:

~~~bash
cd ~/doc2025/DatasetAcquisition
git rev-parse HEAD
rg -n 'minimum_hr_lr_area_ratio|minimum_hr_lr_median_area_ratio|require_fully_in_bounds|minimum_visible_fraction|require_crop_matches_requested_roi|maximum_sequence_gap|maximum_frame_gap|minimum_window_separation|minimum_border_margin_px' lrlpr_curator scripts
~~~

### What this means for the new LPLCv2 eligibility audit

Apply **8x4** as the confirmed production minimum-size values
and **fully in-frame original annotation rectangles** as
reasonable, symmetric crop-integrity requirements. Treat native
**2.5–3.5** aspect ratio as a *new*, explicitly defined external
geometry-control rule, not an acquisition criterion. The audit
reports how many original LPLCv2 bounding rectangles have less
than **2 pixels between their axis-aligned crop rectangle and
the full image edge**, but does **not discard them**:
the LPLCv2 polygon-annotation rectangle and DatasetAcquisition's
detector-requested ROI are not equivalent objects, and
enforcement of the production margin is not confirmed.

No temporal-gap or HR/LR-separation restrictions are imposed
on individual F=1 annotated images. We continue to count
repeated source photos and OCR labels instead of silently
applying global uniqueness to LPLCv2; this is needed to report
all eligible annotations, not to recreate a globally unique
track training set. Keep unchanged source crops and frozen
96x32 bicubic preprocessing for each architecture.

## Read-only audit on daugman (no inference or crops written)

```bash
cd ~/doc2025/project1
git pull --ff-only
AUDIT=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/audit_lplcv2_full_cohort_geometry.py
python3 -m py_compile "$AUDIT"
python3 "$AUDIT" \
  --dataset-root ~/doc2025/LMDB-Datasets/LPLC_dataset \
  --min-aspect 2.5 --max-aspect 3.5 \
  --min-width 8 --min-height 4
```

The script reads the **original full annotation source**; it accepts `leg=1` Poor and `leg=0` Illegible, **old Brazilian or Mercosur 7-character OCR format**, a decoded source image and a fully in-bounds original axis-aligned crop, then applies inclusive native aspect and minimum native width/height. It does **not** exclude motorcycles, occlusion, unknown vehicle types, repeated source images, or repeated OCR labels. It never reads recognizer predictions. All included plate crops will receive the unchanged frozen 96x32 bicubic resize during later evaluation (no padding).

The audit prints **only aggregates**, including totals before and after geometry filtering, exclusion counts, original-annotation SHA-256, source/label repeat counts, native width and height percentiles, native aspect percentiles, zero/under-2px source-image border margin counts, acquisition-like 20%-per-side padded-crop geometry and clipping diagnostics, and how many selected annotations survive progressively stronger width/height cutoffs (24x8, 30x10, 36x12, 42x14, 48x16).

The optional `--output-json` should point **outside the public Git checkout** or be kept excluded from Git.

First compare `retained_after_ratio_and_size_filters` and `joint_size_thresholds_after_aspect` for Poor and Illegible. This lets us decide whether 8x4 is too permissive **based on coverage and original crop resolution**, before running models. A stricter native size minimum would define **a separately named population**; never quietly change the original frozen 3,000-crop dataset.

## Observed full-source LPLCv2 geometry audit on daugman (13 October 2026)

The user ran the local read-only audit on the **original**
`~/doc2025/LMDB-Datasets/LPLC_dataset/annotations_v2.json`
with native aspect range **[2.5,3.5]** and minimum **8x4**.
The source JSON SHA-256 was:

```text
2da194f7393970b7ddf11085a9bb5adb3d6780eca49d9d7389eed0ca06d49a49
```

No dataset has yet been built from this cohort and no model inference
has been run. These are prediction-blind *eligibility counts*.

| Measure | Poor (`leg=1`) | Illegible (`leg=0`) | Both |
|---|---:|---:|---:|
| All source legibility annotations | 7,520 | 5,362 | 12,882 |
| Rejected by Brazilian/Mercosur OCR format (including unknown OCR) | 586 | 4,457 | 5,043 |
| Valid seven-character plate-format + full in-bounds crop | 6,934 | 905 | 7,839 |
| Outside native aspect [2.5,3.5] | 1,534 | 263 | 1,797 |
| Eligible with original tight size >=8x4 and aspect in range | **5,400** | **642** | **6,042** |
| Eligible additionally requiring tight size >=24x8 | **5,400** | **634** | **6,034** |
| Eligible additionally requiring tight size >=30x10 | 5,398 | 586 | 5,984 |
| Eligible additionally requiring tight size >=36x12 | 5,013 | 425 | 5,438 |
| Eligible additionally requiring tight size >=42x14 | 3,519 | 198 | 3,717 |
| Eligible additionally requiring tight size >=48x16 | 1,717 | 80 | 1,797 |

Within the 2.5–3.5 / >=8x4 filtered set:
Poor includes 5,204 distinct source photographs, 5,293
distinct OCR strings, and 2 cases annotated as occluded;
Illegible includes 635 distinct photos and 630 OCR labels,
without annotated occlusion. Repeated identities **must be
retained and disclosed** for the full-eligible case.

The ratio/size filter retains **5,400/6,934 (77.88%)** of
otherwise-valid Poor annotations and **642/905 (70.94%)**
of otherwise-valid Illegible annotations. This does **not**
cover all Poor/Illegible *raw* source annotations; OCR
placeholders/nonmatching plate formats were excluded.
The size threshold is especially consequential for
Illegible: >=36x12 would remove 217 of the 642
aspect-filtered Illegible instances, whereas >=24x8
removes only eight, and none of the 5,400 Poor instances.

**Candidate policy after seeing the prediction-blind inventory:**
native **tight-box** aspect ratio **[2.5,3.5]** and
minimum native **24x8** (area >=192), borrowing the initial
camera `valid_crop` defaults before its `lp_type()`
classification. This yields **6,034** annotation entries
before any further ROI-padding/border policy decision.
This is the recommended *candidate*, not a completed
or officially frozen image dataset.

**Padding geometry follow-up completed on daugman:**
the user ran the updated audit on the same frozen
annotation JSON SHA256 with aspect [2.5,3.5] and tight
minimum 24x8. It reported:

| Diagnostic | Poor | Illegible | Both |
| --- | ---: | ---: | ---: |
| Eligible native tight crops | 5,400 | 634 | 6,034 |
| 20%-per-side padded box fully in-frame | 5,311 | 619 | 5,930 |
| 20%-per-side padded box >=2 px from each source-image edge | **5,305** | **618** | **5,923** |
| Padded box would be clipped at edge | 89 | 15 | 104 |
| Fully in bounds but padded margin <2 px | 6 | 1 | 7 |

This strict padded-image-margin filter would retain
**5,923/6,034 = 98.16%** of the tight-box-eligible
population. It is a defensible image-integrity sensitivity
cohort but not proof that these 2-pixel margins were
actually enforced by the public curator.

**Acquisition aspect-range correction:** the camera
[`cameras/rtsp_pipeline.py`](https://github.com/valfride/DatasetAcquisition/blob/main/cameras/rtsp_pipeline.py)
supplies default car-classification `min_aspect=2.1` and
`max_aspect=3.5` in the absence of an override.
The checked [`config/intelbrasPTZvip.yaml`](https://github.com/valfride/DatasetAcquisition/blob/main/config/intelbrasPTZvip.yaml)
does not override these fields. The classifier first
checks **the tight, zero-padded detector crop**, then the
saved image is **padded 20% per side**.
Thus the user's 2.5-3.5 geometry control is **stricter**
than the public acquisition camera's apparent 2.1-3.5
car-classification defaults. For a closer
acquisition-style match, run the prediction-blind audit
again with `--min-aspect 2.1 --max-aspect 3.5` (and
tight crop 24x8); confirm the deployed acquisition
arguments before describing this as an exact match.

**Cohort decision not yet frozen:** two distinct,
scientifically valid questions need clear labels:
(1) all eligible native images with 2.5-3.5 geometry
to compare eight models, following acquisition-style
20% crop padding but faithfully retaining the
acquisition implementation's boundary clipping, and
(2) a stricter fully padded, >=2px-margin
*clean-geometry sensitivity* cohort, which is 5,923
entries under the current band. A separate, broader
2.1-3.5 cohort would more closely reproduce the camera's
car aspect filter. Do not change policies
based on model inference metrics. Keep the exact
sample identities fixed between padded and tight
variants, and do not run any model yet.

**Causal caution:** a sample-selection band of 2.5–3.5
can change difficulty proportions; the acquisition pipeline
also adds **20% detector-box padding per side**, not present
in the earlier LPLCv2 tight bounding crops. The exact
LPLCv2 label polygon envelope is a proxy for the acquisition
detector's bounding box, not identical by definition.
If two padded and unpadded cohorts are built, they must
share exactly the **same annotation identities**, trained
weights, input tensor sizes and evaluation protocol.
No McNemar test is planned.

## Broader 2.1–3.5 acquisition-style full-cohort audit (confirmed on daugman)

The user ran the read-only original-source audit on the SAME source
annotation SHA256 `2da194f7393970b7ddf11085a9bb5adb3d6780eca49d9d7389eed0ca06d49a49`,
with tight-box native aspect **[2.1,3.5]** and minimum
tight width/height **24x8**, following the acquisition camera's
public default `lp_type()` car-candidate geometry.

| Filter | Poor | Illegible | All |
| --- | ---: | ---: | ---: |
| Valid original OCR and fully in-frame native box | 6,934 | 905 | 7,839 |
| Native ratio [2.1,3.5], size >=24x8 | 6,360 | 803 | **7,163** |
| 20%-each-side padded bbox would not clip | 6,248 | 784 | 7,032 |
| Padded box keeps **>=2 px** source-image margin | **6,240** | **783** | **7,023** |
| Geometry-eligible rejected by strict 2px padded margin | 120 | 20 | 140 |

Of 7,163 native-eligible crops, **7,023 (98.05%)** survive
the strict padded ROI margin rule. The strict cohort comprises
**6,240 Poor (88.85%) and 783 Illegible (11.15%)**.
The broad native-eligible cohort comprises 6,360 Poor and 803
Illegible. Compared to [2.5,3.5], broadening the lower
bound to **2.1** increases the strict-margin count from
5,923 to **7,023** (+1,100).

**Chosen clean-geometry acquisition-style primary cohort: 7,023.**
This is a selection decision based on acquisition-related
geometry and the read-only population inventory, **not on
recognizer accuracy**. It is an explicitly selected subset,
not the full original LPLCv2 dataset nor an exact reconstruction
of every source in LRLPR-26. The 2-pixel requirement matches
the user's intended curator configuration, but the accessible
`selection.py` did not clearly enforce it as an exclusion.

**Paired image-framing sensitivity experiment:** build the
identical 7,023 original annotation identities twice:
(1) unmodified tight floor/ceil box (lossless PNG) and
(2) expand the tight box by **20% of its width/height on
each side**, floor/ceil to integer bounds, and crop from the
same original source frame (lossless PNG). All padded boxes
in this strict cohort fit the source image with >=2px margin.
Each pair uses matching label, order, track ID and exact
source image. The same eight original validation-selected
checkpoints and 96x32 bicubic F=1 evaluation are then applied.

These paired corpora do **not** make the cross-dataset
experiment fully domain-matched: native source distributions,
label uncertainty, detector vs annotation box definitions,
historical acquisition/curation variants and JPEG versus
lossless source encoding still differ.

### Frozen dataset builder committed (run locally; no evaluation yet)

[`scripts/build_lplcv2_acquisition_pair.py`](scripts/build_lplcv2_acquisition_pair.py)
pins the original annotation SHA and the expected
**6,240+783** identities. It will **fail** rather than silently
continue if source annotations or cohort counts change.
It writes two separate F=1-compatible LMDBs only after all
inputs pass preflight. It writes the *same frozen
`selection_manifest.json`* to both, records lossless
crop-byte hashes in local metadata, verifies crop dimensions
and LMDB/crop byte identity, and refuses to overwrite
existing output paths. Restricted crop pixels, labels,
per-sample filenames and manifests remain on daugman.

From `~/doc2025/project1` after `git pull --ff-only`:

~~~bash
BUILDER=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/build_lplcv2_acquisition_pair.py

# Source SHA + exact all-eligible cohort counts. No files written.
python3 -m py_compile "$BUILDER"
python3 "$BUILDER" --preflight

# Run ONLY after successful preflight; new directories, never overwrite.
python3 "$BUILDER" --build

# Repeat independently after the first successful build.
python3 "$BUILDER" --verify-only
~~~

Default user-local outputs:

- `~/doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Tight_7023`
- `~/doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Padded20pct_7023`

**Important:** the previously committed
`run_lplcv2_all_models.py` deliberately asserts the
**old 3,000-crop SHA and sample count**, and its result paths
must not be reused. We need a distinct **7,023-crop**
evaluation runner / checkpoint-lock context after the user
verifies both new LMDBs. Never directly point the old
`--run` command at these new directories.

## Paired LMDB dataset build and verification completed (daugman)

The user independently ran `build_lplcv2_acquisition_pair.py --verify-only`
after the original build found that the target output directories
already existed. Both directories were populated and passed the
builder's complete LMDB/crop-byte/metadata revalidation:

```text
VERIFIED tight 7023 crops; manifest SHA256 107d53e3c2f48ecfe118bb222a828080f6750632b7c9b51a523ef186347ffb48
VERIFIED padded 7023 crops; manifest SHA256 107d53e3c2f48ecfe118bb222a828080f6750632b7c9b51a523ef186347ffb48
PAIRED COHORT VERIFIED: identical manifest and sample order
```

The 7,023-sample selected population and the identical manifests for both crop variants were verified independently on daugman. **The eight padded-only model evaluations have now completed; see the verified aggregate results below.** The tight version was not used for model inference.

### Padded-only eight-model frozen evaluation (user decision)

**Decision, 13 October 2026:** run **ONLY** on the
20%-per-side padded LPLCv2 cohort of **7,023 images**
(**6,240 Poor; 783 Illegible**). The verified tight
version remains preserved on daugman for provenance and
paired identity checks, but **no recognizer inference is
planned on the tight version**. This is **8 model evaluations**,
not 16. No McNemar test or other paired significance analysis.

[`scripts/run_lplcv2_acq7023_all_models.py`](scripts/run_lplcv2_acq7023_all_models.py)
has been updated so `--variant padded` is the **default**;
preflight checks evaluation status only for the selected variant,
while both frozen manifests are still inspected to confirm paired
cohort provenance. The script reuses the old, local
validation-selected checkpoint lock; it does NOT reselect
weights using LPLCv2 data. All results go to new ignored
`results/lplcv2_acq7023_padded_*` directories.
Existing 3,000-image predictions are untouched.

Run on daugman:

~~~bash
cd ~/doc2025/project1
git pull --ff-only

RUNNER=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/run_lplcv2_acq7023_all_models.py

# No inference: verify frozen datasets, F=1 configs, locked checkpoints.
python3 -m py_compile "$RUNNER"
python3 "$RUNNER" --preflight --variant padded

# Eight original-validation-selected recognizers on the padded set ONLY.
python3 "$RUNNER" --run --variant padded

# Recompute/validate all aggregate scores against each prediction CSV.
# Prints All / Poor / Illegible; no tight-vs-padded comparison.
python3 "$RUNNER" --report --variant padded
~~~

The model evaluation format is frozen 96x32 bicubic, F=1,
no augmentation/TTA/SWA, with all eight checkpoints selected
from original LRLPR-26 validation, and no post-hoc adaptation.
Reported metrics: full-plate exact RR 7/7; partial >=6/7
and >=5/7; positional character accuracy; Levenshtein CER;
plus separate legibility-group counts and rates.
The source annotation SHA256 is
`2da194f7393970b7ddf11085a9bb5adb3d6780eca49d9d7389eed0ca06d49a49`
and the shared frozen manifest SHA256 is
`107d53e3c2f48ecfe118bb222a828080f6750632b7c9b51a523ef186347ffb48`.

The user has verified both LMDBs via the independent
`build_lplcv2_acquisition_pair.py --verify-only`
validator. Do **not** invoke the old 3,000-crop runner here.
Completed valid runs are skipped; partially written result
directories must be examined before any retry. Do not commit
checkpoints, local lock, per-plate strings, manifests, CSVs
or inference logs.

## Completed 8/8: acquisition-aligned padded-crop LPLCv2 evaluation (13 October 2026)

The user ran `run_lplcv2_acq7023_all_models.py --report --variant padded` and reported
**`padded fully verified: 8/8 models`**. The auditor verifies the per-sample
prediction-to-manifest joins, recomputes the overall integer metrics and checks
that Poor+Illegible totals add exactly to All; the eight exact-correct
subgroup counts also independently add to the overall numerator. These
are **completed inference results**, not extrapolated estimates.

The same frozen cohort of **7,023 original annotation instances** (6,240 Poor
and 783 Illegible), frozen selection-manifest SHA256
`107d53e3c2f48ecfe118bb222a828080f6750632b7c9b51a523ef186347ffb48`,
was processed at F=1, identical 96x32 bicubic resizing, using only the
**20%-per-side padded** LPLCv2 crops, without retraining, augmentation,
TTA, SWA, or target-data checkpoint tuning. The crop selection
was fixed before the reported inference. The verified **tight** LMDB
was **not evaluated** and is not used to make padding-effect claims.

### Overall (n=7,023)

| Model | 7/7 (%) | >=6 (%) | >=5 (%) | Character accuracy (%) | CER (%) | Exact plates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDiff4STR | 19.41 | 45.35 | 67.65 | 71.44 | 28.51 | 1363/7023 |
| CPPD | 34.56 | 62.47 | 79.05 | 79.17 | 20.82 | 2427/7023 |
| IGTR | 32.83 | 62.82 | 81.40 | 80.06 | 19.93 | 2306/7023 |
| LISTER | 24.75 | 49.57 | 68.12 | 71.95 | 27.87 | 1738/7023 |
| SVTRv2 | 34.22 | 62.51 | 79.13 | 78.56 | 19.79 | 2403/7023 |
| OTE | 19.78 | 45.92 | 68.39 | 71.62 | 28.33 | 1389/7023 |
| SVTRv2-AR | 36.45 | 63.75 | 79.31 | 79.78 | 20.19 | 2560/7023 |
| **Ours (frozen EMA)** | **54.44** | **80.45** | **91.39** | **88.39** | **11.60** | **3823/7023** |

### Poor-labelled (n=6,240)

| Model | 7/7 (%) | >=6 (%) | >=5 (%) | Character accuracy (%) | CER (%) | Exact plates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDiff4STR | 20.10 | 46.62 | 68.72 | 72.09 | 27.88 | 1254/6240 |
| CPPD | 35.90 | 64.46 | 80.58 | 80.15 | 19.84 | 2240/6240 |
| IGTR | 33.77 | 64.13 | 82.50 | 80.78 | 19.21 | 2107/6240 |
| LISTER | 25.19 | 50.02 | 68.97 | 72.33 | 27.49 | 1572/6240 |
| SVTRv2 | 34.82 | 63.41 | 79.95 | 78.98 | 19.29 | 2173/6240 |
| OTE | 19.95 | 46.60 | 69.17 | 72.07 | 27.89 | 1245/6240 |
| SVTRv2-AR | 37.79 | 65.71 | 80.67 | 80.76 | 19.21 | 2358/6240 |
| **Ours (frozen EMA)** | **55.80** | **81.78** | **92.50** | **89.15** | **10.84** | **3482/6240** |

### Illegible-labelled (n=783)

| Model | 7/7 (%) | >=6 (%) | >=5 (%) | Character accuracy (%) | CER (%) | Exact plates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDiff4STR | 13.92 | 35.25 | 59.13 | 66.34 | 33.59 | 109/783 |
| CPPD | 23.88 | 46.62 | 66.92 | 71.36 | 28.64 | 187/783 |
| IGTR | 25.42 | 52.36 | 72.67 | 74.31 | 25.65 | 199/783 |
| LISTER | 21.20 | 45.98 | 61.30 | 68.97 | 30.89 | 166/783 |
| SVTRv2 | 29.37 | 55.30 | 72.54 | 75.21 | 23.79 | 230/783 |
| OTE | 18.39 | 40.49 | 62.20 | 68.09 | 31.86 | 144/783 |
| SVTRv2-AR | 25.80 | 48.15 | 68.45 | 71.98 | 27.97 | 202/783 |
| **Ours (frozen EMA)** | **43.55** | **69.86** | **82.50** | **82.32** | **17.66** | **341/783** |

All eight integer numerator checks pass: MDiff4STR 1254+109=1363;
CPPD 2240+187=2427; IGTR 2107+199=2306; LISTER 1572+166=1738;
SVTRv2 2173+230=2403; OTE 1245+144=1389;
SVTRv2-AR 2358+202=2560; Ours 3482+341=3823.
All three exact-match percentages from integer correct counts
match their two-decimal reported values.

**Descriptive findings:** the proposed CNN–ViT leads all eight models
on every reported metric, both in the combined sample and in
each of the two annotated legibility categories.
On all 7,023 instances it reaches **54.44% exact RR (3,823)**,
versus runner-up **SVTRv2-AR at 36.45% (2,560)**,
**+17.98 percentage points and 1,263 extra correctly recognized
plates**. On Poor, the margin is **+18.01 pp** over SVTRv2-AR
(55.80% vs 37.79%; +1,124 correct plates). On Illegible-labelled
records, **+14.18 pp** over SVTRv2
(43.55% vs 29.37%; +111 correct plates).
The strongest other overall character accuracy is
IGTR's 80.06% (vs Ours 88.39%);
the lowest other overall CER is SVTRv2's 19.79%
(vs Ours 11.60%). These are observed margins only;
**no significance tests are claimed**.

**Important limits for the journal:** this is the acquisition-aligned
20%-padded, fully-in-frame and >=2px-margin, native-aspect
[2.1,3.5], tight-minimum 24x8 **selected** LPLCv2 subgroup;
it is **not the complete LPLCv2 data** or an official
representative public benchmark. Illegible is an annotation label,
not verified human readability. Original LPLCv2 annotations
with OCR placeholders/invalid Brazilian-format labels were
ineligible, especially many Illegible entries. The sample includes
repeated source images and plate labels; observations are not
all independent. No synthetic augmentation or target adaptation
was applied.

**Never attribute the change versus the earlier 3,000-image results
to 20% padding alone.** The new evaluation also changes source
pool inclusion, native aspect ratio limits, minimum dimensions,
padded-margin inclusion and class mix
(6,240/783 versus 2,158/842). Since no models were evaluated
on the verified 7,023-image tight variant, there is no
controlled paired crop-padding effect estimate.
The appropriate paper statement is **superior descriptive
recognition on this predeclared acquisition-aligned cohort**,
without unverified causal robustness explanations.
Do not silently overwrite the older 3,000-image diagnostic
results or their historical claims.

**Dataset restrictions:** never commit images, annotation JSON, crops, per-annotation manifests, checkpoints or prediction CSVs to public GitHub.
