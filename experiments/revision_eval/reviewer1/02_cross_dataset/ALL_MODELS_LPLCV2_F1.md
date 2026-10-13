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


## Check the best-validation EMA checkpoint before freezing the paper

**Important provenance correction (13 October 2026).** The frozen
`experiments/revision_eval/submitted_model/last.pth` is an **EMA
checkpoint**: the evaluator's original submitted-model F=1 log confirms
`model_ghost_sd` was loaded. However, the name `last.pth` is **not
proof it contains the best-validation epoch**. The user has identified
another existing, *original-LRLPR-validation-selected candidate*:

```text
experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/ghost_weights/ghost_acc_0.7660_ep_58.pth
```

Its filename indicates 76.60% LRLPR validation recognition at epoch 58;
it has **not yet been checked on daugman** against the submitted
`last.pth` tensor values or original Table-I test results. The eight
cross-dataset results above still include the **26.4%** submitted-model
`last.pth` run. Do **not** replace that result or describe it as
the independently best-validation EMA checkpoint until identity is
established. Selecting a checkpoint on **original LRLPR-26 validation**
(rather than on LPLCv2 test) is the correct fair comparison rule.

First compare SHA-256 and exact model tensor contents locally:

```bash
cd ~/doc2025/project1
OLD=experiments/revision_eval/submitted_model/last.pth
BEST=experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/ghost_weights/ghost_acc_0.7660_ep_58.pth
sha256sum "$OLD" "$BEST"

python3 - <<'PY'
import torch
old = torch.load('experiments/revision_eval/submitted_model/last.pth',
                 map_location='cpu', weights_only=False)
best = torch.load(
    'experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/ghost_weights/ghost_acc_0.7660_ep_58.pth',
    map_location='cpu', weights_only=False)
print('Old epoch:', old.get('epoch'), 'best epoch:', best.get('epoch'))
print('Old state keys:', sorted(k for k in old if k.endswith('_sd')))
print('Best state keys:', sorted(k for k in best if k.endswith('_sd')))
a, b = old['model_ghost_sd'], best['model_ghost_sd']
print('Same keys:', a.keys() == b.keys())
print('EMA tensors byte-equivalent:',
      a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a))
PY
```

Before evaluating a different file, verify it uses the same frozen
architecture and recognition configuration as the submitted checkpoint:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
root = Path('experiments')
paths = [
    root/'revision_eval/submitted_model/config_snapshot.yaml',
    root/'ablations/ce_sfb/ce_sfb_13-05-2026-final/config_snapshot.yaml'
]
configs = [yaml.safe_load(p.read_text()) for p in paths]
for key in ('model_g', 'cls_loss', 'alphabet'):
    same = configs[0].get(key) == configs[1].get(key)
    print(f'{key}: {"MATCH" if same else "DIFFERENT"}')
    if not same:
        raise SystemExit(f'STOP: mismatch in {key}')
print('Architecture/classifier configuration confirmed.')
PY
```

Other training-only YAML fields may differ; this check protects
against evaluating different network architectures with the same
checkpoint loader's permissive `strict=False` setting.

If the EMA weights are identical, **no new inference is required**;
keep the existing 26.4% result. If different, run the candidate on
the **same frozen LPLCv2 dataset with a new directory**:

```bash
cd ~/doc2025/project1
BEST="$PWD/experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final/ghost_weights/ghost_acc_0.7660_ep_58.pth"
OUT="$PWD/experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_mixed_aspect_ours_best_ema"
CKPT_DIR="$OUT/frozen_checkpoint_only"
DATA="$HOME/doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k"

test -f "$BEST" && test -f "$DATA/metadata.pkl" || exit 1
mkdir -p "$CKPT_DIR"
ln -sfn "$BEST" "$CKPT_DIR/$(basename "$BEST")"

python3 experiments/revision_eval/shared/scripts/run_eval_matrix.py \
  --config experiments/revision_eval/submitted_model/config_snapshot.yaml \
  --checkpoints "$CKPT_DIR" \
  --split "$DATA" \
  --output-dir "$OUT" \
  --frames 1 --fusions bayes

cat "$OUT/F1_bayes.json"
```

The saved `F1_bayes.txt` must say it loaded Ghost EMA weights
and `F1_bayes.json` must list the exact
`ghost_acc_0.7660_ep_58.pth` filename.
Do not mutate, overwrite, or rename the previous `ours` result
or any baseline output.

**Consistency with Table I:** the committed Table-I 60.6% F=1
(and 74.2% F=3, 78.5% F=5) was reproduced using
`submitted_model/last.pth`. If epoch-58 weights differ and the
best-validation checkpoint becomes the primary submitted choice,
also evaluate the same checkpoint on the untouched original
LRLPR-26 test set for F=1/3/5; reconcile Table I, its text, and
the checkpoint provenance. This prevents reporting different
model selections for the two tables without saying so.

Do not rank the newly selected checkpoint based on external LPLCv2
results. The full-matrix runner's `--table` intentionally still
reports the originally frozen `ours` results until the primary
checkpoint decision and revalidation are complete.
