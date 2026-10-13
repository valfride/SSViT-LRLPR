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

**Scientific limitations:** This is a post-hoc geometry-controlled exploratory subset, not an official LPLCv2 recognition distribution. Some OCR-labelled Illegible crops may not be visually decipherable. Original crop ratios still vary; all models receive a direct aspect-distorting 96x32 bicubic resize to match their **frozen** LRLPR-26 protocol. Individual annotations do not guarantee fully independent source photographs. Poor/Illegible-stratified descriptive metrics are now available below; the per-track paired McNemar comparison, crop-size sensitivity analysis, and ideally source/annotation quality review remain outstanding. Do not treat differences between marginal accuracies alone as significance tests.

The manuscript's dedicated `Cross-Dataset Experimental Results` subsection lives in `valfride/CNN-ViT-LRLPR-paper` on `revision/ojits-major-revision`. Its three cross-dataset tables are populated with the eight-model exact/partial-recognition, character-level, and Poor/Illegible-stratified results. Paired significance and native-crop-size conditioning remain outstanding.


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
The user ran the character diagnostic successfully, yielding all
eight rows below. These rounded percentages were checked locally by
`--character-table` against exact integer numerator/denominator counts.

| Model | Exact-match RR (%) ↑ | Char. accuracy (%) ↑ | CER (%) ↓ |
|---|---:|---:|---:|
| MDiff4STR | 12.03 | 62.18 | 37.75 |
| CPPD | **27.17** | 73.10 | 26.86 |
| IGTR | 26.73 | **75.11** | **24.86** |
| LISTER | 18.13 | 64.04 | 35.39 |
| SVTRv2 | 16.53 | 64.44 | 31.97 |
| OTE | 12.47 | 57.86 | 42.02 |
| SVTRv2-AR | 20.07 | 68.51 | 31.40 |
| Ours (best EMA) | 26.40 | 72.80 | 27.17 |

The full-plate and character-level rankings are not identical:
CPPD leads exact-match RR, while IGTR leads char. accuracy and CER.
The best-validation EMA CNN--ViT is third in all three metrics
on this fixed 3,000-crop subset; relative to CPPD, it trails by
0.77 percentage points in RR and 0.30 percentage points in char.
accuracy, with 0.31 percentage points higher CER.
No significance or distinct Poor/Illegible cohort claims follow from
these aggregate metrics alone.

The manuscript retains the Table-I-style 7/7, >=6, >=5 main table
and adds a compact character-accuracy/CER diagnostic table with
these values, under the dedicated Cross-Dataset Experimental Results
subsection.


## Poor versus Illegible subgroup analysis (all eight models)

**Purpose:** determine whether the external ranking changes with the
frozen LPLCv2 source **legibility annotations**. This partitions exactly
the already-verified 3,000 samples into 2,158 Poor and 842 OCR-labelled
Illegible plate crops. It does not rerun recognition or introduce new
test examples. For every model, **both cohorts come from the same F=1
mixed-aspect selection**. Do not compare these subgroup results to the
separately constructed historical Poor-only 3,000 subset.

Run on daugman:

```bash
cd ~/doc2025/project1
git pull --ff-only
RUNNER=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/run_lplcv2_all_models.py
python3 -m py_compile "$RUNNER"
python3 "$RUNNER" --legibility-table
```

This prints **two Markdown tables**, Poor (n=2,158) and Illegible
(n=842), for all eight models, with 7/7 exact-match RR,
>=6 and >=5 partial recognition, seven-slot character accuracy,
Levenshtein CER, and **exact correct/total counts**.
It does not print any plate GT, prediction, source filename or
sample ID. Save its textual output privately if desired.

For each model, it uses the existing
`summarize_mixed_legibility_predictions.py` verification code:
joins predictions to the frozen `selection_manifest.json` by
virtual track ID, checks the frozen SHA-256
`3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05`,
checks sample identity/GT, independently computes every character
match and edit distance, requires complete 3,000-prediction coverage,
and verifies recomputed global integer counts against the original
`F1_bayes.json`. The new eight-model command additionally ensures
the Poor and Illegible subgroup numerators/denominators sum to the
verified global totals. Any mismatch stops the command.

**Scientific interpretation:** an OCR-labelled *Illegible* target is
not necessarily human-readable from the released crop, so these
are as-annotated recognition rates, not certified perceptual
legibility performance. Compare rankings **within each cohort**;
different class sizes and label quality preclude simple causal
claims. Small between-model differences still require paired
inference on the same local prediction identities, preferably
accounting for duplicate source photographs. The main manuscript
table remains frozen until real subgroup aggregate outputs
are available.


## Completed 8/8 Poor versus Illegible analysis (13 October 2026)

The user ran `python3 "$RUNNER" --legibility-table` after a
fast-forward pull. The command printed
`Verified: 8/8 models; all 3,000 predictions per model match the frozen
manifest and original global results.` The original frozen selection-manifest
SHA-256 was `3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05`.
No models were retrained or re-evaluated: these are derived from
the original eight per-image predictions, grouped by source
`source_legibility` in the manifest.

### Poor-labelled cohort (n=2,158)

| Model | 7/7 (%) | >=6 (%) | >=5 (%) | Char. acc. (%) | CER (%) | Exact correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDiff4STR | 12.28 | 29.98 | 50.60 | 63.27 | 36.66 | 265/2158 |
| CPPD | **28.04** | 52.59 | 70.30 | 74.59 | 25.39 | 605/2158 |
| IGTR | 26.27 | **52.78** | **73.77** | **76.16** | **23.82** | 567/2158 |
| LISTER | 17.52 | 37.26 | 54.17 | 64.70 | 34.75 | 378/2158 |
| SVTRv2 | 14.87 | 33.55 | 54.08 | 64.25 | 31.99 | 321/2158 |
| OTE | 11.35 | 26.83 | 42.91 | 57.66 | 42.22 | 245/2158 |
| SVTRv2-AR | 20.02 | 41.01 | 61.58 | 69.54 | 30.35 | 432/2158 |
| Ours (best EMA) | 25.58 | 48.10 | 67.10 | 72.87 | 27.09 | 552/2158 |

### Illegible-labelled cohort (n=842)

| Model | 7/7 (%) | >=6 (%) | >=5 (%) | Char. acc. (%) | CER (%) | Exact correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDiff4STR | 11.40 | 28.27 | 45.84 | 59.38 | 40.55 | 96/842 |
| CPPD | 24.94 | 46.67 | 61.76 | 69.27 | 30.64 | 210/842 |
| IGTR | 27.91 | **50.59** | 65.91 | 72.45 | 27.52 | 235/842 |
| LISTER | 19.71 | 36.34 | 50.95 | 62.33 | 37.00 | 166/842 |
| SVTRv2 | 20.78 | 37.29 | 55.58 | 64.93 | 31.93 | 175/842 |
| OTE | 15.32 | 30.76 | 46.20 | 58.36 | 41.52 | 129/842 |
| SVTRv2-AR | 20.19 | 38.36 | 55.58 | 65.86 | 34.07 | 170/842 |
| Ours (best EMA) | **28.50** | 50.36 | **67.70** | **72.62** | **27.37** | 240/842 |

The existing direct-subgroup totals cross-check the original
global exact correct counts: MDiff4STR 265+96=361, CPPD
605+210=815, IGTR 567+235=802, LISTER 378+166=544,
SVTRv2 321+175=496, OTE 245+129=374, SVTRv2-AR
432+170=602, and Ours 552+240=792.

**Descriptive ranking reversal:** On Poor, CPPD leads in 7/7
RR (28.04%), followed by IGTR (26.27%) and Ours (25.58%).
On Illegible-labelled crops, Ours leads in 7/7 RR (28.50%),
followed by IGTR (27.91%) and CPPD (24.94%).
Relative to CPPD, Ours has **53 fewer** exact plates on Poor
and **30 more** on Illegible, yielding an overall difference
of **23 fewer**. Relative to IGTR, the differences are **15 fewer**
and **5 more**, yielding **10 fewer** overall.

On Poor, IGTR leads in character accuracy (76.16%) and lowest
CER (23.82%); on Illegible-labelled crops, Ours narrowly leads
in character accuracy (72.62% versus IGTR's 72.45%) and CER
(27.37% versus 27.52%). On Illegible, IGTR still leads in the
>=6 partial metric (50.59% versus Ours 50.36%).

**Avoid overinterpretation:** several models exhibit higher
exact-match RR in the *Illegible*-labelled cohort than in the
*Poor* cohort; this is not proof that visually illegible plates
are intrinsically easier. The original Illegible and Poor pools
have different sourcing/selection paths, post-hoc aspect filtering
removes unusual shapes, and some Illegible GT strings have
unverified visual recoverability. The reported labels are
annotation categories, not a controlled ordinal difficulty axis.
Differences of five plates between Ours and IGTR within
Illegible are particularly small; paired tests and source-level
dependency handling are needed before significance claims.

**Next diagnostic work:** paired correctness-discordance McNemar
comparisons for Ours/CPPD and Ours/IGTR within each subgroup,
and optional source-crop-height/area-conditioned performance.
Neither should require running the neural networks again,
provided the local manifests and eight prediction CSVs are retained.
