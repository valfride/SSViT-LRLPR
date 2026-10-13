# LPLCv2 mixed difficulty F=1 external dataset: 905 Illegible + 2095 Poor

**Status:** frozen-seed dry-run completed on daugman with expected
905 Illegible + 2,095 Poor crop composition, zero cross-subset photo or
OCR-label overlap, and native aspect-ratio statistics recorded below.
**Full LMDB build and independent verification are not yet confirmed.**
This is a new exploratory mixed-difficulty test subset, not the
original 3,000-sample Poor-only benchmark.

## Scientific motivation and source-paper verification

Wojcik et al., *LPLCv2: An Expanded Dataset for Fine-Grained
License Plate Legibility Classification*, IJCNN 2026,
https://arxiv.org/pdf/2604.08741, Section III, pages 2–3,
explicitly state:

- LP annotation contains localization box, OCR text and legibility level;
- localization/OCR/legibility annotations were generated with
  YOLOv11/PARSeq-tiny/ResNet-50 and **subsequently manually reviewed**;
- OCR was also cross-validated against vehicle data;
- if severe degradation precluded reliable human validation, OCR
  characters were withheld;
- **some otherwise Illegible plates could be validated using camera zoom**;
- Illegible refers to substantial degradation preventing confident
  character interpretation under the standard view, except where
  zoom makes validation possible.

Consequently a valid OCR label on an Illegible plate is consistent
with the authors' documentation; it does NOT imply the characters
are independently readable from the released single image. The
local annotation audit found 1,189 seven-character Illegible OCR
*fields*, but 284 were the placeholder UNKNOWN. We retain the 905
Brazilian-format OCR annotations; the only invalid-geometry source
belonged to UNKNOWN.

## Prespecified composition

- **905 Illegible** (\`leg=0\`): include ALL 905 from the previously
  verified \`LPLCv2_IllegibleBrazilianOCR_TEST\` set. Preserve any
  repeated labels, source photographs, motorcycles and unknown
  vehicle types; do not deduplicate these 905.
- **2,095 Poor** (\`leg=1\`): choose uniformly at random with
  \`random.Random(2026)\` from the previously frozen 3,000-sample
  \`LPLCv2_PoorNonMotoUnique_TEST_3k\` test set, after excluding
  any Poor crops whose *source photographs* or *OCR strings*
  occur in the 905 Illegible population. Fail before building if
  fewer than 2,095 are eligible.
- **Total 3,000 plate crops** (30.17% Illegible, 69.83% Poor).
  Do not describe them as 3,000 distinct original photographs.
  Under a successful zero-cross-source-overlap draw, the 905
  Illegible entries come from 891 distinct source photos and
  the 2,095 Poor entries from 2,095 distinct photos, yielding
  **2,986 distinct source photographs**. 890 distinct Illegible
  labels plus 2,095 distinct Poor labels (no cross-class duplicates)
  yield **2,985 distinct OCR strings**.

This new mixed set is assembled **without reading CNN–ViT or
SVTRv2-AR predictions**. No reweighting or test-set subset choice
based on correctness/confidence, and no fine-tuning. Both parent
LMDBs and frozen evaluation outputs are left unchanged.

Important inclusion asymmetry: the Poor set was originally
restricted to known non-motorcycle vehicle types and no occlusion;
the Illegible set preserves known motorcycles (25), unknown type
(318) and any available occlusion. These different eligibility
criteria must be disclosed; the mixture is **not** a random sample
of the full LPLCv2 class distribution, nor the dataset authors'
official classification-test split. The 905/2095 composition was
chosen for a supplementary OCR stress test after earlier analyses,
not a prospectively established test distribution.

## Matching the original CNN–ViT / SVTRv2-AR aspect ratio

**Confirmed from committed code/configs:** both frozen evaluations
use `VSR_Sequence_collate_fn` with `imgW: 96`,
`imgH: 32`, `image_aspect_ratio: 3.0`, evaluation
`aug: false`, `test: true`, and `--frames 1`.
The actual wrapper calls:

```python
lr_img = cv2.resize(img_raw, (96, 32), interpolation=cv2.INTER_CUBIC)
```

So the **model-input aspect ratio** is always **3:1**. Crucially,
the wrapper does **not** keep native aspect ratio via padding:
original crops of 2.6:1, 3.2:1, 1.5:1 etc. are
resized *anisotropically* to 3:1. The `image_aspect_ratio`
configuration field is metadata in this path, not a signal
to use an aspect-preserving padding routine.

The mixed-LMDB builder **copies original lossless crop bytes
unchanged**, preserving the same pipeline as the prior LRLPR-26
and frozen LPLCv2 Poor/Illegible tests. It now verifies both model
configs have 96×32/3:1 and the same unaugmented wrapper,
simulates the exact `cv2.INTER_CUBIC` resizing for **every**
output image, and records in the preview and validation:

- Original native bounding-box width and height p10/median/p90
  separately for Poor, Illegible and combined.
- Original width/height aspect-ratio min, p10/median/p90, max.
- How many crops are naturally exactly 3:1, and how many
  differ from 3:1 by over 10% or 20%.
- The required 3×32×96 model-input tensor geometry, 3:1
  aspect, config hashes, bicubic resampling and no padding.

This allows us to measure geometric-warping risk without changing
our test distribution. If we later want **aspect-preserving padding**,
that must be a **different, clearly labelled sensitivity experiment**
applying the same new preprocessing to both models, not a
quiet change to this frozen protocol.

## Observed dry-run sample selection and original-crop geometry

The user provided the actual dry-run output from daugman
(13 October 2026). These are **pre-build selection statistics**, not
a completed LMDB validation:

| Sample selection | Count |
| --- | ---: |
| Illegible annotated plate crops | 905 |
| Selected Poor plate crops | 2,095 |
| Total samples | **3,000** |
| Distinct original source photographs | 2,986 |
| Distinct OCR strings | 2,985 |
| Distinct annotation identities | 3,000 |
| Poor candidates eligible after cross-subset exclusion | 2,929 |
| Poor candidates excluded due to Illegible photo overlap | 45 |
| Poor candidates excluded due to Illegible OCR-string overlap | 26 |
| Remaining cross-subset photo / OCR-string overlap | **0 / 0** |

Layout distribution: Poor-Mercosur 1,681; Poor-old 414;
Illegible-Mercosur 737; Illegible-old 168;
combined Mercosur 2,418 (80.6%) and old Brazilian 582 (19.4%).

**Original lossless crop aspect ratios** (width/height, before 96×32
bicubic input resizing):

| Measurement | All 3,000 | Illegible 905 | Poor 2,095 |
| --- | ---: | ---: | ---: |
| Aspect ratio p10 / median / p90 | 2.30 / 2.7222 / 3.00 | 2.14 / 2.6667 / 3.00 | 2.375 / 2.75 / 3.0343 |
| Aspect ratio min / max | 0.963 / 4.6176 | 1.0476 / 4.2222 | 0.963 / 4.6176 |
| Exactly 3:1 | 156 (5.2%) | 41 (4.5%) | 115 (5.5%) |
| Deviation from 3:1 over 10% | 1,419 (47.3%) | 515 (56.9%) | 904 (43.2%) |
| Deviation from 3:1 over 20% | 438 (14.6%) | 197 (21.8%) | 241 (11.5%) |
| Native width p10 / median / p90 | Not supplied | 27.4 / 37 / 48 | 36 / 45 / 74 |
| Native height p10 / median / p90 | Not supplied | 11 / 14 / 19 | 13 / 16 / 28 |

Interpretation: the **model input** remains rigorously fixed at 96×32
and 3:1 for both models, but almost half the raw crops are
more than 10% away from the target aspect ratio. Distortion induced
by direct resizing may be more consequential for Illegible crops,
which have both smaller native dimensions and more frequent aspect
deviation. This does NOT by itself establish wrong annotations or
causal performance effects.

For a fair frozen-protocol test, preserve all 3,000 selected instances,
retain their lossless original crop bytes and use the same direct
bicubic resize as prior evaluations. Do not retroactively discard
low-aspect/high-aspect instances. Independently examine suspicious
near-square crops in the QA review; any alternative
aspect-preserving padding should be a separately named sensitivity
study applied identically to both models. The actual native
aspect-ratio distribution of original LRLPR-26 crops has not
been established by this audit; do not claim a match beyond
the shared **model-input** geometry.

## Frozen parent manifests

Illegible 905:
\`7e74f46d991db9835739e7b99b5c01f1be6da7b21ba1b50b2c9bef848b7dfb0e\`

Poor 3000:
\`8a08180604d88b652c6d915853830b4b1fd0a9ff6bcceace597d8a57f199d408\`

The builder rejects mismatched parent hashes, unsupported plate
strings and duplicate annotation identities. The final manifest
records parent hashes, all copied source identities and subset
provenance, per-class plate-format counts, selection seed and
source/label collision exclusions. New images are the *exact*
lossless PNG bytes already present in the two source LMDBs.

## Preparation on daugman

Run the dry-run first, inspect eligible Poor count and
source/label overlap exclusions, then build a new output:

\`\`\`bash
cd ~/doc2025/project1
git pull --ff-only

python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedPoorIllegible_TEST_3k \
  --seed 2026 \
  --dry-run
\`\`\`

Once dry-run counts are correct, repeat **without** \`--dry-run\`:

\`\`\`bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedPoorIllegible_TEST_3k \
  --seed 2026
\`\`\`

Verify independently after build:

\`\`\`bash
python3 experiments/revision_eval/reviewer1/02_cross_dataset/scripts/prepare_lplcv2_mixed_3k.py \
  --illegible ~/doc2025/LMDB-Datasets/LPLCv2_IllegibleBrazilianOCR_TEST \
  --poor ~/doc2025/LMDB-Datasets/LPLCv2_PoorNonMotoUnique_TEST_3k \
  --output ~/doc2025/LMDB-Datasets/LPLCv2_MixedPoorIllegible_TEST_3k \
  --verify-only
\`\`\`

No source dataset is modified; new output directory must not
already exist. Expected outputs: \`data.mdb\`, \`metadata.pkl\`,
\`crops/\`, \`selection_manifest.json\`, \`dataset_info.json\` and
\`validation.json\`.

## Both frozen recognizers, only after validation

Both use the same new 3,000-crop LMDB at F=1. Keep separate output
folders and preserve the original experimental results.

CNN–ViT:

\`\`\`bash
python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
  --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
  --checkpoints experiments/revision_eval/submitted_model \
  --split ~/doc2025/LMDB-Datasets/LPLCv2_MixedPoorIllegible_TEST_3k \
  --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_ours \
  --frames 1 --fusions bayes
\`\`\`

SVTRv2-AR: use the same frozen \`student_acc_0.7500_ep_52.pth\`
baseline checkpoint already selected on LRLPR-26. The actual
checkpoint directory must be verified on daugman; do not
guess it or select a different checkpoint:

\`\`\`bash
python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
  --config baselines_configs/SVTRV2_AR_BASELINE.yaml \
  --checkpoints /ACTUAL/FROZEN/SVTRV2_AR/CHECKPOINT_DIRECTORY \
  --split ~/doc2025/LMDB-Datasets/LPLCv2_MixedPoorIllegible_TEST_3k \
  --output-dir experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_svtrv2_ar \
  --frames 1 --fusions bayes
\`\`\`

Report the final manifest SHA, exact-match RR, character-slot
accuracy and CER overall **and separately within Poor and
Illegible**. Because the parent subsets have different
selection rules and Illegible OCR labels may be visually
unverifiable, report this as an exploratory as-annotated
mixed-legibility stress test, not an official or representative
LPLCv2 OCR benchmark.

All source images and derived crops/annotation manifests are
request-restricted and stay on daugman unless explicit sharing
permission allows otherwise.
