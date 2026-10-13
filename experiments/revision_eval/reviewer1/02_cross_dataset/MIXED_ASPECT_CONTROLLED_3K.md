# LPLCv2 mixed Poor/Illegible: native aspect-controlled 3,000-crop F=1 test

**Status:** the revised **two-sided [1.7,4.0] LMDB was built on daugman**
(13 October 2026), with 842 Illegible and 2,158 Poor crops (3,000 total),
zero cross-class source/label overlap, and original crop geometry within
[1.7, 4.0]. The build returns its final selection-manifest SHA-256 and
calls the internal `verify(stage)` before staging is atomically renamed.
The independent post-build `--verify-only` was executed on daugman
and **passed all 3,000 samples**. The recomputed selection-manifest
SHA-256 is identical to the frozen build manifest.
The previous one-sided [1.7, infinity) dry-run (844/2,156) is retained
below as historical evidence.
This is an explicitly post-hoc, new test **variant**. Never overwrite
the previous mixed 905 Illegible + 2,095 Poor dataset, or replace
the earlier frozen Poor-only 3,000-case external evaluation.

## Frozen built LMDB provenance

The completed build command reported:

- Output directory: `~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k`
- **Selection-manifest SHA-256**:
  `3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05`
- Source Illegible parent SHA-256:
  `7e74f46d991db9835739e7b99b5c01f1be6da7b21ba1b50b2c9bef848b7dfb0e`
- Source Poor parent SHA-256:
  `8a08180604d88b652c6d915853830b4b1fd0a9ff6bcceace597d8a57f199d408`
- Submitted CNN–ViT config SHA-256:
  `9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`
- SVTRv2-AR baseline config SHA-256:
  `f8c656f5486018553c88746678de842d7dd5ff3ebb30f1697ad48a3e8dc0a31f`
- Included files: `data.mdb`, `metadata.pkl`,
  `selection_manifest.json`, `dataset_info.json`,
  `validation.json`, `crops/`

**Independent verification passed.** The builder validated all 3,000
crop/metadata records during atomic staging, and the later, independent
post-build `--verify-only` pass reconfirmed all 3,000 samples and
the exact frozen manifest SHA-256. The dataset's restricted images,
crop data, and per-sample annotation manifest remain on daugman.

## Selection rule (independent of recognizer predictions)

Start with the EXACT original 3,000-crop mixed selection reconstructed
deterministically by `prepare_lplcv2_mixed_3k.py` (seed 2026):

- 905 Illegible samples with valid Brazilian/Mercosur plate-format GT.
- 2,095 Poor samples drawn from the already frozen 3,000-crop
  non-motorcycle Poor LMDB, excluding every Poor source photograph
  or plate OCR string overlapping with the 905 Illegible pool.

Then use the **native** lossless crop width and height already
recorded in the source manifests:

1. Remove **every** initially selected crop with native aspect ratio
   **below 1.7 or above 4.0**, from either Illegible or Poor.
   Bounds are inclusive: exactly **1.7 and 4.0 are retained**.
2. Keep every initially selected crop with ratio **in [1.7, 4.0]**;
   do not change its OCR label, pixels or inclusion for other reasons.
3. Replace removed entries 1-for-1 with **unused** crops from the
   already frozen Poor-3k source pool, with native crop ratios in
   **[2.5, 3.5]** inclusive and no overlapping Illegible source
   photograph or plate OCR string.
4. Sample replacements using `random.Random(seed+17)` and reshuffle
   the resulting 3,000 entries using `random.Random(seed+41)`.
   If there are too few qualifying unused Poor examples,
   **stop** and report the deficit. Do not silently broaden the
   aspect-ratio range or reuse model evaluation correctness.
5. Since the original Illegible-905 is exhaustive, removed Illegible
   samples **cannot** be replenished with extra Illegible
   annotations. The final ratio of classes will be
   `Illegible=905-n_removed_Illegible`, and
   `Poor=2095+n_removed_Illegible`; exact figures are only known
   once the dry-run finishes.

The builder validates the two original source manifest SHA-256s
(see [MIXED_POOR_ILLEGIBLE_3K.md](MIXED_POOR_ILLEGIBLE_3K.md)),
prevents cross-subset source/label overlaps, reports all removed and
replacement annotation identities, retains the unmodified PNG
bytes, and audits the native aspect ratio of the final selection.
Its frozen manifest records `selection_origin` per sample
(`original_seeded_mix` vs `near_ideal_poor_refill`), as well as
original source LMDB track IDs and image filename/index.

## Model geometry remains unchanged

Both CNN–ViT and SVTRv2-AR receive the **same**
`cv2.resize(...,(96,32),interpolation=cv2.INTER_CUBIC)`
operation from their frozen `VSR_Sequence_collate_fn` wrappers.
Native aspect-ratio filtering makes extreme stretching less common,
but the remaining crops may still be geometrically stretched.
**No padding, resizing, relabeling, lossless-crop modification,
fine-tuning or checkpoint changes** are introduced.

This deliberately changes the evaluated population, especially by
dropping unusual-perspective narrow and wide plate crops, some of
which can be legitimate. It therefore cannot be
reported as the same dataset or as an unbiased improvement of OCR
performance. Any comparison to the unfiltered mixed set is a
**post-hoc geometry sensitivity analysis**, and the prior
Poor-only external RR of 24.57% remains unchanged.

## Successful two-sided [1.7, 4.0] dry-run — 13 October 2026

The user ran the fixed-seed [1.7,4.0] selection on daugman and
provided the full geometry summary. This describes a selected
population, **not yet a built or verified LMDB**.

| Property | Observed |
| --- | ---: |
| Total plate crops / distinct annotation identities | **3,000 / 3,000** |
| Illegible | **842 (28.07%)** |
| Poor | **2,158 (71.93%)** |
| Unique source photographs | 2,986 |
| Unique OCR labels | 2,985 |
| Cross-class overlapping source photographs | **0** |
| Cross-class overlapping OCR labels | **0** |
| Candidate Poor after original Illegible overlap exclusion | 2,929 |
| Original Poor overlaps removed by image / label | 45 / 26 |
| Minimum / maximum original crop aspect ratio | **1.7059 / 4.0** |
| Median original crop aspect ratio | 2.7333 |
| Exact native 3:1 crops | 160 |
| Crops with native ratio >10% from 3:1 | 1,365 |
| Crops with native ratio >20% from 3:1 | 359 |

Final plate formats: Poor Mercosur 1,732, Poor old Brazilian
426, Illegible Mercosur 687, Illegible old Brazilian 155:
**2,419 Mercosur, 581 old Brazilian**.

| Native crop measurement | Illegible (842) | Poor (2,158) |
| --- | ---: | ---: |
| Width p10 / median / p90 (px) | 29 / 38 / 48 | 36 / 45 / 74 |
| Height p10 / median / p90 (px) | 11 / 14 / 19 | 13 / 16 / 27 |
| Aspect min / max | 1.7059 / 3.8824 | 1.75 / 4.0 |
| Aspect p10 / median / p90 | 2.3082 / 2.6667 / 3.0 | 2.4 / 2.75 / 3.0323 |
| Aspect exactly 3:1 | 41 | 119 |
| Aspect >10% off 3:1 | 452 | 913 |
| Aspect >20% off 3:1 | 134 | 225 |

Compared with the initial no-geometry-filter 905/2095 mixture,
63 Illegible annotations were removed (905 → 842), offset by a
net increase of 63 Poor crops. The **total number** of narrow and
wide removals (including Poor-to-Poor replacements) is unknown
from this excerpt; the complete run reports `removed_below_min`,
`removed_above_max`, `removed_counts` and `removed_total`.
Do not equate the net Poor increase with the total replacement count.

Both frozen configuration checks passed with **96×32 bicubic input,
no padding**, SHA-256
`9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`
(CNN–ViT) and
`f8c656f5486018553c88746678de842d7dd5ff3ebb30f1697ad48a3e8dc0a31f`
(SVTRv2-AR).

**Next:** run both frozen recognition models at F=1 on the
**same verified** `LPLCv2_MixedAspect1p7to4p0_TEST_3k` directory.
Do not alter the earlier Poor-only or prior mixed selections.

## Historical lower-bound-only dry-run — 13 October 2026

The user ran the old builder with **only** `--min-native-aspect 1.7`,
without a maximum, and supplied the following statistics.
**These results do NOT represent the revised [1.7,4.0] filter.**
The new dry-run must still establish its class counts.

| Property | Original mixed 3,000 | Geometry-controlled preview |
| --- | ---: | ---: |
| Total plate crops | 3,000 | **3,000** |
| Illegible | 905 | **844** |
| Poor | 2,095 | **2,156** |
| Distinct source photographs | 2,986 | **2,986** |
| Distinct plate OCR strings | 2,985 | **2,985** |
| Cross-class source/label overlap | 0 / 0 | **0 / 0** |
| Min native crop width/height | 0.963 | **1.7059** |
| Median native aspect ratio | 2.7222 | **2.7333** |
| Native aspect deviation >10% from 3:1 | 1,419 | **1,369** |
| Native aspect deviation >20% from 3:1 | 438 | **363** |

Sixty-one Illegible samples were removed by the **old minimum-only**
filter (905 → 844). Additional Illegible samples over 4.0 may
be removed by the new two-sided filter.
The total number of removed and replenished Poor samples
cannot be determined from the *final class totals alone*; inspect
`native_aspect_selection.removed_counts` and
`native_aspect_selection.added_near_ideal_poor` in the original
dry-run output. The builder records them automatically.

Final layout: Poor Mercosur 1,731; Illegible Mercosur 689;
Poor old Brazilian 425; Illegible old Brazilian 155.
The final subset includes **2,420 Mercosur and 580 old Brazilian**
plate crops.

Final original crop ratios by class:

| Measurement | Illegible (844) | Poor (2,156) |
| --- | ---: | ---: |
| Width p10 / median / p90 | 29 / 38 / 48 | 36 / 45 / 74 |
| Height p10 / median / p90 | 11 / 14 / 19 | 13 / 16 / 27 |
| Aspect min–max | 1.7059–4.2222 | 1.75–4.6176 |
| Aspect p10 / median / p90 | 2.3091 / 2.6667 / 3.0 | 2.4 / 2.75 / 3.0377 |
| Exactly 3:1 | 41 | 117 |
| >10% off 3:1 | 454 | 915 |
| >20% off 3:1 | 136 | 227 |

Model configuration SHA-256s checked by the dry-run:

- Frozen CNN–ViT:
  `9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`
- SVTRv2-AR:
  `f8c656f5486018553c88746678de842d7dd5ff3ebb30f1697ad48a3e8dc0a31f`

Both configs specify the same direct bicubic resize to **96×32**,
without padding. The **historical** filter applied only the lower
bound (1.7), allowing very wide crops. The **revised** filter
also rejects aspect ratios above 4.0. The historical 75-case
reduction in
`>20%` native-aspect deviations is descriptive and does not
establish improved OCR accuracy or annotation correctness.

The existing baseline datasets and evaluation scores remain
unchanged. The next step is to run the **new two-sided dry-run**
below, inspect removal and replacement supply, and then build
in a **new output directory** if viable; verify all 3,000 crops, freeze the
resulting new selection manifest SHA-256, and run the frozen
models with independent result output paths.

### Selection regression tests

From the repository root, run:

```bash
python3 -m unittest discover \
  -s experiments/revision_eval/reviewer1/02_cross_dataset/scripts \
  -p 'test_prepare_lplcv2_mixed_3k.py' -v
```

The tests check inclusive [1.7,4.0] boundaries, separately counted
narrow/wide removals, deterministic no-overlap Poor replenishment,
failure on insufficient supplies, and the old unfiltered/lower-only modes.

## Phase 1: dry-run on daugman (do this FIRST)

```bash
cd ~/doc2025/project1
git pull --ff-only

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k \
  --seed 2026 \
  --min-native-aspect 1.7 \
  --max-native-aspect 4.0 \
  --refill-aspect-min 2.5 \
  --refill-aspect-max 3.5 \
  --dry-run
```

Inspect `native_aspect_selection`:
`removed_counts`, `removed_total`,
`unused_near_ideal_poor_candidates`,
`added_near_ideal_poor`, and `final_counts`.
The successful dry-run yielded 842 Illegible and 2,158 Poor
crops, with all native ratios within **[1.7, 4.0]** and zero
cross-class source/label overlap. Keep the complete per-sample
removal/replacement report for provenance, and ensure the
non-dry build uses the same flags and seed.

If the shortage safeguard triggers, inspect the actual inventory
before negotiating a documented alternative. Do NOT silently
change the agreed threshold/replacement interval to make it pass.

## Phase 2: build NEW frozen variant and verify

**Already executed successfully on daugman.** Preserved here
for reproducibility. Do not rerun in the same output directory:

```bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k \
  --seed 2026 \
  --min-native-aspect 1.7 \
  --max-native-aspect 4.0 \
  --refill-aspect-min 2.5 \
  --refill-aspect-max 3.5
```

```bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k \
  --verify-only
```

The independent `--verify-only` run returned `verified: 3000`,
`all_test_split: true`, `all_crops_lmdb_match: true`,
`all_model_inputs_96x32_bicubic: true`, and exactly the frozen
manifest SHA-256. The verified 842/2,158 counts and the native
aspect ratio range 1.7059–4.0 match the build. The verifier
also validates category counts, original annotation uniqueness,
and the absence of cross-subset source-image/OCR-label overlaps.
All restricted images/manifests/predictions remain local on daugman.

## Evaluation only after manifest freeze

Test both **frozen** models at F=1 against the identical new LMDB,
with dedicated `lplcv2_mixed_aspect_ours` and
`lplcv2_mixed_aspect_svtrv2_ar` result directories; never reuse
previous output folders. Report overall and class-stratified
recognition, CER, and per-position accuracy, including original
vs newly selected crop counts. Do not select or remove additional
samples based on predictions; complete the independent
source-target annotation review when feasible.


## First frozen F=1 result: SVTRv2-AR baseline (13 October 2026)

**Inference completed; aggregate metrics supplied by the user**
from `results/lplcv2_mixed_aspect_svtrv2_ar/F1_bayes.json`.
Validation mode, F=1, `SVTRV2_AR`, bayes/BJP fusion, **no TTA** and
**no SWA**; random seed 42, deterministic cuDNN.
The checkpoint-selection record names exactly
`student_acc_0.7500_ep_52.pth` through the symlink-only
checkpoint directory. Its actual source path is
`experiments/baselines/SVTRV2_AR_BASELINE_svtrv2_ar/student_weights/student_acc_0.7500_ep_52.pth`
as independently inspected on daugman. **Checkpoint SHA-256 is
not yet supplied**; do not claim a particular bytes hash.

| Metric | Frozen SVTRv2-AR on verified mixed aspect-controlled LMDB |
| --- | ---: |
| Exact-match recognition rate (RR) | **20.07% (602/3,000)** |
| Character-slot accuracy | **68.51% (14,387/21,000)** |
| Character error rate (CER) | **31.40% (6,593 edits/21,000 GT chars)** |
| At least six correct slots | 40.27% (1,208/3,000) |
| At least five correct slots | 59.90% (1,797/3,000) |
| Old Brazilian exact-match RR | **26.51% (154/581)** |
| Mercosur exact-match RR | **18.52% (448/2,419)** |
| Mean confidence (correct vs incorrect) | 0.84143 vs 0.65510 |

These are exact reported `F1_bayes.json` aggregate results; no
sample predictions are stored in this tracked documentation. The
corresponding prediction CSV is maintained locally on daugman.
The separate 842-case Illegible and 2,158-case Poor recognition
results **have not yet been calculated or reported**. Plate-layout
accuracy is not a substitute for legibility-stratified accuracy.

The previously reported CNN–ViT **24.57% (737/3,000)** on the
different **Poor-only** external subset is **not** a paired comparator
for this mixed-difficulty benchmark. CNN–ViT has now also been
evaluated on the exact **same frozen mixed** dataset (see direct
comparison below). Do not retune models or select crops based on
these observed predictions. This remains exploratory and
as-annotated due to Illegible annotation QA caveats.

## Direct fixed-checkpoint comparison on the SAME 3,000 crops

**Both F=1 inference runs are complete** on the identical verified
LMDB and frozen manifest SHA-256
`3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05`.
The following values come from the respective user-supplied local
`F1_bayes.json` reports, not from unpublished prediction CSV files.
Both used test mode `val`, F=1, no TTA/SWA, seed 42, and
deterministic cuDNN. CNN–ViT uses `cls_loss=CE` and
`experiments/revision_eval/submitted_model/last.pth`
(submitted frozen checkpoint directory). SVTRv2-AR uses
`cls_loss=SVTRV2_AR` and frozen
`student_acc_0.7500_ep_52.pth`.

| Aggregate metric | CNN–ViT | SVTRv2-AR | CNN–ViT minus SVTRv2-AR |
| --- | ---: | ---: | ---: |
| Full seven-character exact-match RR | **26.40% (792/3,000)** | 20.07% (602/3,000) | **+6.33 percentage points, +190 exact plates** |
| Character-position accuracy | **72.80% (15,287/21,000)** | 68.51% (14,387/21,000) | **+4.29 pp, +900 matching positions** |
| Character error rate (CER, lower is better) | **27.17% (5,705 edits/21,000)** | 31.40% (6,593 edits/21,000) | **−4.23 pp, 888 fewer edits** |
| At least 6 correct positions | **48.73% (1,462/3,000)** | 40.27% (1,208/3,000) | +8.47 pp, +254 plates |
| At least 5 correct positions | **67.27% (2,018/3,000)** | 59.90% (1,797/3,000) | +7.37 pp, +221 plates |
| Old Brazilian RR | **33.56% (195/581)** | 26.51% (154/581) | +7.06 pp, +41 plates |
| Mercosur RR | **24.68% (597/2,419)** | 18.52% (448/2,419) | +6.16 pp, +149 plates |

CNN–ViT records higher overall exact-match and character-slot
accuracy, lower CER, and higher observed exact-match accuracy in
both Brazilian plate formats. This is an **observed external-domain
advantage over this SVTRv2-AR baseline**, not a demonstrated causal
benefit of a single CNN–ViT component or guaranteed label validity.

**Pending analysis:** compute the Poor (2,158) versus Illegible
(842) per-cohort results from the frozen local prediction CSVs
using the script below; then calculate a **paired McNemar test**
from per-track discordances, rather than inferring significance
from aggregate 792/602 alone. A few source photographs contribute
multiple annotations, so sample dependence should be acknowledged.
Report this experiment as an **exploratory post-hoc geometry-filtered
as-annotated benchmark**, not as the previously frozen Poor-only
3,000-image result or an official representative LPLCv2 OCR test.

### Reproducible local Poor / Illegible stratification

The `test.py` evaluator writes `F1_bayes_predictions.csv` with
virtual test-set track IDs plus `correct`, `positional_matches`
and `edit_distance` fields. This has been cross-checked against
its implementation. The new
`scripts/summarize_mixed_legibility_predictions.py` script joins
tracks against the original frozen selection manifest (rather than
assuming CSV ordering), verifies OCR GT identities, recomputes slot
and edit-distance metrics and checks that recomputed totals match
the corresponding `F1_bayes.json` global counts. It prints only
aggregate Poor/Illegible/All statistics.

SVTRv2-AR (now ready to analyze):

```bash
cd ~/doc2025/project1
D=~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k
R=experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_aspect_svtrv2_ar

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/summarize_mixed_legibility_predictions.py \
  --dataset "$D" \
  --predictions "$R/F1_bayes_predictions.csv" \
  --metrics "$R/F1_bayes.json" \
  --expected-manifest-sha256 3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05 \
  --output-json "$R/legibility_aggregates.json"
```

Run the same script **now** for the completed CNN–ViT run,
changing `R` to
`experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_aspect_ours`.
Both F=1 prediction CSVs should already exist on daugman.
The `legibility_aggregates.json` file contains only aggregated
numeric metrics, not source plate strings or per-sample identities.
Keep the underlying restricted prediction CSV local.

## Frozen F=1 recognition invocations (after successful verification)

CNN–ViT (the submitted checkpoint and configuration snapshot):

```bash
cd ~/doc2025/project1
python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
  --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
  --checkpoints experiments/revision_eval/submitted_model \
  --split ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k \
  --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_aspect_ours \
  --frames 1 --fusions bayes
```

SVTRv2-AR: select the **original frozen LRLPR-26 baseline**
`student_acc_0.7500_ep_52.pth`. Its exact daugman path must be
verified; do **not** substitute another baseline checkpoint. A
one-file symlink directory prevents the runner's automatic
`*acc_*.pth` selection from choosing a different weight set.

```bash
cd ~/doc2025/project1
mapfile -t CKPTS < <(
  find "$HOME/doc2025" -type f -name 'student_acc_0.7500_ep_52.pth' -print
)
if (( ${#CKPTS[@]} != 1 )); then
  printf 'Expected exactly 1 frozen SVTR checkpoint, found %s\n' "${#CKPTS[@]}"
  printf '%s\n' "${CKPTS[@]}"
else
  SVTR_CKPT="${CKPTS[0]}"
  RESULT_DIR="$PWD/experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_aspect_svtrv2_ar"
  LINK_DIR="$RESULT_DIR/frozen_checkpoint_only"
  mkdir -p "$LINK_DIR"
  ln -sfn "$SVTR_CKPT" "$LINK_DIR/student_acc_0.7500_ep_52.pth"
  sha256sum "$SVTR_CKPT"

  python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
    --config baselines_configs/SVTRV2_AR_BASELINE.yaml \
    --checkpoints "$LINK_DIR" \
    --split ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k \
    --output-dir "$RESULT_DIR" \
    --frames 1 --fusions bayes
fi
```

Both evaluations must run **without TTA, SWA, training, sample filtering,
or checkpoint reselection**. Expected output in each isolated
directory: `F1_bayes.json`, `F1_bayes_predictions.csv`,
`F1_bayes.txt` and `matrix_manifest.json`.
The overall exact-match rate is not enough: also separately score
the 842 Illegible and 2,158 Poor annotations by joining predictions
against the frozen local `selection_manifest.json`. Because some
Illegible OCR labels may be unverifiable from the released still crop,
interpret as **exploratory as-annotated** recognition on a
post-hoc geometry-controlled subset; the original Poor-only score
remains unchanged. Do not commit restricted per-image prediction
records or plate labels without permission.
