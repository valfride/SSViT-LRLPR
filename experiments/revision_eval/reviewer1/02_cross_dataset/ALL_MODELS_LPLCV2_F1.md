# All Table-I models on frozen LPLCv2 (single frame)

**Status (13 October 2026):** all **eight** Table-I architectures completed F=1 inference on the same independently verified frozen 3,000-crop mixed Poor/Illegible LMDB; the user's `--table` command confirmed `Completed: 8/8`. Exact-match and partial-recognition metrics below are copied from that verified local summary. Character accuracy and CER are already present in each local `F1_bayes.json` and may be reported without rerunning inference.

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

## Independent preflight confirmed on daugman (13 October 2026)

The user ran `--write-lock-template`, then `--preflight` and
`--run`. Both preflights passed, validating the frozen manifest
hash, all eight checkpoint paths, and the presence/protocol of the
two existing aggregate evaluations. The `--run` call **started
MDiff4STR**; its completion and the remaining six-run metrics
are not yet confirmed. Do not claim all eight runs completed.

The following are the actual SHA-256 values printed by `--preflight`
for the local frozen checkpoint files (not GitHub uploads):

| Model | Selected checkpoint SHA-256 |
| --- | --- |
| MDiff4STR | `e7ec4524f397a15fe1ce7629b40c7cf7415028434eec2a3be0d092c9d64f5143` |
| CPPD | `1de6f9cd0d424e714cb2c7721747ea296a82760f8a786f1849793fe5c1e7e89f` |
| IGTR | `5d6a367dbb888a65e540b3491779cfa1fefa8dd69e33048c35d3faaa8d9d75d0` |
| LISTER | `6cfa22a0707f03d5cd35d3df3f99b343d978212c990b038e5c7b4bd1fbe578c5` |
| SVTRv2 | `3dc8d3af7664dd554aa8a02816380326fac2bbb838d1ede0b3398f7c02018e69` |
| OTE | `2a64e2fcee16e1aaa1ebd2fdfa93a140d55e31231d8c44f4608b831fe47408a1` |
| SVTRv2-AR | `c6f89c74deed70ab9dcc7ca46d7dd0f7c21cb0b3074a4dd15313689eb8692838` |
| Ours | `1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a` |

The output also confirmed:
`SVTRv2-AR: complete, will skip`;
`Ours: complete, will skip`; and
the other six models pending at the time of launch.

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
| MDiff4STR | 1 | 12.0 | 29.5 | 49.3 | Complete |
| CPPD | 1 | **27.2** | 50.9 | 67.9 | Complete |
| IGTR | 1 | 26.7 | **52.2** | **71.6** | Complete |
| LISTER | 1 | 18.1 | 37.0 | 53.3 | Complete |
| SVTRv2 | 1 | 16.5 | 34.6 | 54.5 | Complete |
| OTE | 1 | 12.5 | 27.9 | 43.8 | Complete |
| SVTRv2-AR | 1 | 20.1 | 40.3 | 59.9 | Complete |
| Ours (best EMA) | 1 | 26.4 | 48.7 | 67.3 | Complete |

**Scientific limitations:** This is a post-hoc geometry-controlled exploratory subset, not an official LPLCv2 recognition distribution. Some OCR-labelled Illegible crops may not be visually decipherable. Original crop ratios still vary; all models receive a direct aspect-distorting 96x32 bicubic resize to match their **frozen** LRLPR-26 protocol. Individual annotations do not guarantee fully independent source photographs. Before a final manuscript conclusion, obtain Poor/Illegible-stratified metrics, the per-track paired McNemar comparison, and ideally source/annotation quality review; do not treat a difference between marginal accuracies alone as a significance test.

The manuscript's dedicated `Cross-Dataset Experimental Results` subsection lives in `valfride/CNN-ViT-LRLPR-paper` on `revision/ojits-major-revision`. Its eight-model table is populated with these values, pending additional character-level and legibility analyses.


## Submitted EMA checkpoint identity confirmed on daugman

The user independently verified these two files have **the same SHA-256**
(`1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a`):

- `experiments/revision_eval/submitted_model/last.pth`
- `experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/ghost_weights/ghost_acc_0.7660_ep_58.pth`

Both report checkpoint epoch **58**, and independently comparing every
`model_ghost_sd` parameter with `torch.equal` returned **True**.
Thus the submitted `last.pth` evaluated in the LPLCv2 full matrix is
**byte-identical to the best original-validation EMA checkpoint**; it is
not merely the last chronological EMA training state.
The existing 26.4% F=1 LPLCv2 result and the unchanged original Table-I
F=1/F=3/F=5 results use that same frozen EMA checkpoint. No additional
CNN--ViT rerun or LPLCv2-driven model reselection is required.

## Character-level diagnostic table

`test.py` already computes character-position accuracy (matching the
seven fixed GT positions) and CER (Levenshtein substitutions, insertions
and deletions divided by GT length) for each of the eight frozen runs.
Their different error definitions mean `character accuracy + CER`
does **not** necessarily equal 100%.

```bash
cd ~/doc2025/project1
git pull --ff-only
RUNNER=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/run_lplcv2_all_models.py
python3 "$RUNNER" --character-table
```

This new no-inference command reads each local `F1_bayes.json`,
requires 3,000 plates and 21,000 GT/slot character positions, and
cross-checks the computed character metrics against integer counts.
Keep the main Table-I-style 7/7, >=6, >=5 table and consider adding
a compact character accuracy/CER table or corresponding discussion
**after inspecting the full eight-model diagnostic output**.
No new scores are assumed here.

