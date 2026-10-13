# LPLCv2 mixed Poor/Illegible: native aspect-controlled 3,000-crop F=1 test

**Status:** aspect-filtered **dry-run completed successfully** on daugman.
The selected 3,000-crop population now contains **844 Illegible and 2,156 Poor**;
its original crop aspect ratio has minimum **1.7059**. The completed
LMDB build and `--verify-only` output have **not yet been supplied**.
This is an explicitly post-hoc, new test **variant**. Never overwrite
the previous mixed 905 Illegible + 2,095 Poor dataset, or replace
the earlier frozen Poor-only 3,000-case external evaluation.

## Selection rule (independent of recognizer predictions)

Start with the EXACT original 3,000-crop mixed selection reconstructed
deterministically by `prepare_lplcv2_mixed_3k.py` (seed 2026):

- 905 Illegible samples with valid Brazilian/Mercosur plate-format GT.
- 2,095 Poor samples drawn from the already frozen 3,000-crop
  non-motorcycle Poor LMDB, excluding every Poor source photograph
  or plate OCR string overlapping with the 905 Illegible pool.

Then use the **native** lossless crop width and height already
recorded in the source manifests:

1. Remove **every** initially selected crop with native
   width / native height **below 1.7**. This applies to
   both the Illegible and Poor portions; a crop with
   aspect ratio exactly 1.7 is retained.
2. Keep every initially selected crop with aspect ratio **>= 1.7**;
   do not preferentially discard other samples or change their GT.
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
dropping unusual-perspective plate crops. It therefore cannot be
reported as the same dataset or as an unbiased improvement of OCR
performance. Any comparison to the unfiltered mixed set is a
**post-hoc geometry sensitivity analysis**, and the prior
Poor-only external RR of 24.57% remains unchanged.

## Observed aspect-controlled dry-run — 13 October 2026

The user ran the builder's geometry-controlled preview on `daugman` and
supplied the following end-state and geometry figures. **These are
dry-run statistics; no completed dataset has been verified.**

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

Sixty-one Illegible samples were removed (905 → 844).
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
without padding. The crop filter applies **only the lower native
aspect bound (1.7)**; retained extreme wide crops above 3.5
are intentionally not removed. The 75-case reduction in
`>20%` native-aspect deviations is descriptive and does not
establish improved OCR accuracy or annotation correctness.

The existing baseline datasets and evaluation scores remain
unchanged. The next step is to run the non-dry command below in
a **new output directory**, verify all 3,000 crops, freeze the
resulting new selection manifest SHA-256, and run the frozen
models with independent result output paths.

## Phase 1: dry-run on daugman (do this FIRST)

```bash
cd ~/doc2025/project1
git pull --ff-only

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspectControlled_TEST_3k \
  --seed 2026 \
  --min-native-aspect 1.7 \
  --refill-aspect-min 2.5 \
  --refill-aspect-max 3.5 \
  --dry-run
```

Inspect `native_aspect_selection`:
`removed_counts`, `removed_total`,
`unused_near_ideal_poor_candidates`,
`added_near_ideal_poor`, and `final_counts`.
It will report the original sample identities and aspects for
every removal and replacement. It should retain exactly
3,000 crops with no native ratios below 1.7, but the
final number of Illegible examples is unknown until this run.

If the shortage safeguard triggers, inspect the actual inventory
before negotiating a documented alternative. Do NOT silently
change the agreed threshold/replacement interval to make it pass.

## Phase 2: build NEW frozen variant and verify

Only after reviewing the dry-run, repeat the command above
without `--dry-run`:

```bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspectControlled_TEST_3k \
  --seed 2026 \
  --min-native-aspect 1.7 \
  --refill-aspect-min 2.5 \
  --refill-aspect-max 3.5
```

```bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedAspectControlled_TEST_3k \
  --verify-only
```

Record the actual `selection_manifest.json` SHA-256 and
`validation.json` for publication provenance. The verifier
checks each raw LMDB image, model 96x32 geometry, crop dimensions,
category counts, original annotation uniqueness, absence of
cross-subset overlaps and selected ratio thresholds. All restricted
images/manifests/predictions stay local on daugman.

## Evaluation only after manifest freeze

Test both **frozen** models at F=1 against the identical new LMDB,
with dedicated `lplcv2_mixed_aspect_ours` and
`lplcv2_mixed_aspect_svtrv2_ar` result directories; never reuse
previous output folders. Report overall and class-stratified
recognition, CER, and per-position accuracy, including original
vs newly selected crop counts. Do not select or remove additional
samples based on predictions; complete the independent
source-target annotation review when feasible.
