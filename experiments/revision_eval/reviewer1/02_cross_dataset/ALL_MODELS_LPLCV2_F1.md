# All Table-I models on frozen LPLCv2 (single frame)

**Status (13 October 2026):** frozen 3,000-crop LPLCv2 mixed Poor/Illegible dataset independently verified. Frozen CNN–ViT and SVTRv2-AR F=1 inference completed; the other **six** Table-I baseline evaluations are pending on daugman. No baseline scores should be invented or carried across from LRLPR-26 or from a different LPLCv2 subset.

## Protocol

- **Unchanged** dataset: \`~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k\`, 2,158 Poor + 842 Illegible, native crop aspect within [1.7, 4.0], selection seed 2026, prediction-blind replenishment.
- Manifest SHA-256: \`3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05\`.
- Evaluate **only F=1**, because there is one image per annotated license plate (not a five-frame sequence). Identical 96x32 bicubic input, no padding, no augmentation, no TTA, no SWA, no external adaptation.
- All **eight** architectures from the manuscript Table I: MDiff4STR, CPPD, IGTR, LISTER, SVTRv2 (CTC), OTE, SVTRv2-AR, and the submitted CNN–ViT.
- **Lock the exact prior LRLPR-26 validation-selected checkpoint for each model**. The baseline weights are not uploaded to GitHub. Do not select a checkpoint or modify any cropped image based on observed LPLCv2 performance. The checkpoint for SVTRv2-AR is the epoch-52 file (even though other checkpoints have equal rounded reported validation accuracy). Submitted CNN–ViT uses \`last.pth\` from its submitted snapshot folder.
- The output for each model must contain \`F1_bayes.json\`, \`F1_bayes_predictions.csv\`, \`F1_bayes.txt\`, and \`matrix_manifest.json\`. Keep per-image predictions/labels on daugman; only aggregated results are suitable for publication after review. The local \`.gitignore\` prevents accidental addition of this directory.

## Local checkpoint-locking and inference

Pull the existing evaluation branch first:

\`\`\`bash
cd ~/doc2025/project1
git pull --ff-only

RUNNER=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/run_lplcv2_all_models.py
python3 "$RUNNER" --inventory
\`\`\`

The inventory prints local baseline \`student_weights\` directories and candidate files, **without choosing one**. Check the checkpoint recorded for each baseline in the **original** LRLPR-26 validation experiment, not which file has the highest accuracy on the external test set. Inspect the actual checkpoint source directory and the old Table-I evaluation log if two validation-selected candidates tie.

Create the **local-only** checkpoint selection template:

\`\`\`bash
python3 "$RUNNER" --write-lock-template
\`\`\`

This writes \`experiments/revision_eval/reviewer1/02_cross_dataset/FROZEN_LPLCV2_CHECKPOINTS.local.json\`.
It prepopulates the two known checkpoints and leaves six paths blank. Replace the empty strings with the verified exact file paths on daugman, using paths relative to the project root or absolute paths. The local file is ignored by Git and should not be committed as an authoritative model selection before validation.

After confirming all eight checkpoint choices, run:

\`\`\`bash
python3 "$RUNNER" --run
\`\`\`

This checks the frozen dataset SHA and exact sample composition, verifies the model-specific loss routing, validation wrapper and 96x32 geometry, displays the SHA-256 of each selected checkpoint, and evaluates any missing models through the existing \`run_eval_matrix.py\`/ \`test.py\` F=1 codepath. Existing, complete CNN–ViT and SVTRv2-AR outputs are checked and skipped; **the runner does not overwrite nonempty incomplete result folders**. It stops on errors. It does not run training.

## Collect the manuscript's Table-I-style metrics

\`\`\`bash
python3 "$RUNNER" --table
\`\`\`

The output has the exact manuscript columns **Model / Frames / 7/7 / >=6 / >=5**, using one-decimal precision and reporting only metrics actually available in each frozen F=1 JSON. Once all eight models are complete, this prints LaTeX rows with per-column best results highlighted.

| Model | F | 7/7 (%) | >=6 (%) | >=5 (%) | Current status |
|---|---:|---:|---:|---:|---|
| MDiff4STR | 1 | — | — | — | Pending |
| CPPD | 1 | — | — | — | Pending |
| IGTR | 1 | — | — | — | Pending |
| LISTER | 1 | — | — | — | Pending |
| SVTRv2 | 1 | — | — | — | Pending |
| OTE | 1 | — | — | — | Pending |
| SVTRv2-AR | 1 | 20.1 | 40.3 | 59.9 | Complete |
| Ours | 1 | 26.4 | 48.7 | 67.3 | Complete |

**Scientific limitations:** This is a post-hoc geometry-controlled exploratory subset, not an official LPLCv2 recognition distribution. Some OCR-labelled Illegible crops may not be visually decipherable. Original crop ratios still vary; all models receive a direct aspect-distorting 96x32 bicubic resize to match their **frozen** LRLPR-26 protocol. Individual annotations do not guarantee fully independent source photographs. Before a final manuscript conclusion, obtain Poor/Illegible-stratified metrics, the per-track paired McNemar comparison, and ideally source/annotation quality review; do not treat a difference between marginal accuracies alone as a significance test.

The manuscript's dedicated \`Cross-Dataset Experimental Results\` subsection lives in \`valfride/CNN-ViT-LRLPR-paper\` on \`revision/ojits-major-revision\`. Its placeholders must be replaced **only after the corresponding verified model results are available**.
