# All Table-I models on frozen LPLCv2 (single frame)

**Status (13 October 2026):** frozen 3,000-crop LPLCv2 mixed Poor/Illegible dataset independently verified. Frozen CNN–ViT and SVTRv2-AR F=1 inference completed; the other **six** Table-I baseline evaluations are pending on daugman. No baseline scores should be invented or carried across from LRLPR-26 or from a different LPLCv2 subset.

## Protocol

- **Unchanged** dataset: `~/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k`, 2,158 Poor + 842 Illegible, native crop aspect within [1.7, 4.0], selection seed 2026, prediction-blind replenishment.
- Manifest SHA-256: `3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05`.
- Evaluate **only F=1**, because there is one image per annotated license plate (not a five-frame sequence). Identical 96x32 bicubic input, no padding, no augmentation, no TTA, no SWA, no external adaptation.
- All **eight** architectures from the manuscript Table I: MDiff4STR, CPPD, IGTR, LISTER, SVTRv2 (CTC), OTE, SVTRv2-AR, and the submitted CNN–ViT.
- **Lock the exact prior LRLPR-26 validation-selected checkpoint for each model**. The baseline weights are not uploaded to GitHub. Do not select a checkpoint or modify any cropped image based on observed LPLCv2 performance. The checkpoint for SVTRv2-AR is the epoch-52 file (even though other checkpoints have equal rounded reported validation accuracy). Submitted CNN–ViT uses `last.pth` from its submitted snapshot folder.
- The output for each model must contain `F1_bayes.json`, `F1_bayes_predictions.csv`, `F1_bayes.txt`, and `matrix_manifest.json`. Keep per-image predictions/labels on daugman; only aggregated results are suitable for publication after review. The local `.gitignore` prevents accidental addition of this directory.

## Identified original-baseline checkpoint candidates (daugman inventory)

The user confirmed the following available local checkpoint files.
The candidates correspond to the highest validation accuracy in each
baseline's saved `*acc_*.pth` filenames. The repository `test.py`
evaluator sorts paths alphabetically, then stably sorts by descending
validation accuracy; **ties thus select the earlier filename**.
Confirm these choices with the original LRLPR-26 Table-I evaluation
logs if independently recorded.

| Model | Selected checkpoint candidate | Validation acc. in filename |
| --- | --- | ---: |
| MDiff4STR | `MDIFF_BASELINE_14-04-2026/model_acc_0.6040_ep_21.pth` | 0.6040 |
| CPPD | `CPPD_BASELINE_14-04-2026/model_acc_0.6880_ep_77.pth` | 0.6880 |
| IGTR | `IGTR_BASELINE_14-04-2026/model_acc_0.6600_ep_61.pth` | 0.6600 |
| LISTER | `LISTER_BASELINE_14-04-2026/model_acc_0.6830_ep_103.pth` | 0.6830 (tie with ep 113) |
| SVTRv2 | `SVTRV2_BASELINE_14-04-2026/model_acc_0.7030_ep_146.pth` | 0.7030 (tie with ep 156) |
| OTE | `OTE_BASELINE_14-04-2026/model_acc_0.7430_ep_88.pth` | 0.7430 |
| SVTRv2-AR | `student_weights/student_acc_0.7500_ep_52.pth` | 0.7500 (tie with ep 85, 94) |
| Ours | `submitted_model/last.pth` | submitted frozen checkpoint |

Full paths are prepopulated in the runner's JSON lock template.
The apparent top-checkpoint accuracy is from the original
LRLPR-26 validation process, **not from LPLCv2**.
We have not independently re-read the local Table-I checkpoint-selection
logs; filenames and `test.py` selection mechanics justify candidates
but not an unconditional claim about previously executed Table-I runs.

## Local checkpoint-locking and inference

Pull the existing evaluation branch and inspect the candidates:

```bash
cd ~/doc2025/project1
git pull --ff-only

RUNNER=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/run_lplcv2_all_models.py
python3 -m py_compile "$RUNNER"
python3 "$RUNNER" --inventory
```

The inventory includes both direct `model_acc_*.pth` files and
SVTRv2-AR's `student_weights/` structure. Review tie selections against
original validation logs as needed. Create a **local-only** selection
file (will refuse to overwrite an existing one):

```bash
python3 "$RUNNER" --write-lock-template
cat experiments/revision_eval/reviewer1/02_cross_dataset/FROZEN_LPLCV2_CHECKPOINTS.local.json
```

All eight paths are prepopulated from the user's directory inventory.
If the original evaluation used a different selected checkpoint,
correct the path in this local JSON **before evaluating LPLCv2**.
Do not commit this local-only file.

After confirming the eight paths, first check them and their SHA-256s
*without invoking inference*:

```bash
python3 "$RUNNER" --preflight
```

If preflight succeeds, launch the missing six models:

```bash
python3 "$RUNNER" --run
```

This checks the frozen dataset SHA and sample composition, verifies
the loss routing, validation wrapper and 96x32 geometry, records
checkpoint file hashes in terminal output, and evaluates missing
models with the existing `run_eval_matrix.py` / `test.py` F=1
codepath. Completed CNN–ViT and SVTRv2-AR results are validated
and skipped without overwriting them; **nonempty incomplete output
directories are not overwritten**. The runner stops on any error.
No training occurs.

## Collect the manuscript's Table-I-style metrics

```bash
python3 "$RUNNER" --table
```

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

The manuscript's dedicated `Cross-Dataset Experimental Results` subsection lives in `valfride/CNN-ViT-LRLPR-paper` on `revision/ojits-major-revision`. Its placeholders must be replaced **only after the corresponding verified model results are available**.
