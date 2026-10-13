# LPLCv2 mixed difficulty F=1 external dataset: 905 Illegible + 2095 Poor

**Status:** builder committed; **not yet run on daugman**.
This is a new, exploratory mixed-difficulty external test subset,
not the earlier 3,000-sample Poor-only benchmark.

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
