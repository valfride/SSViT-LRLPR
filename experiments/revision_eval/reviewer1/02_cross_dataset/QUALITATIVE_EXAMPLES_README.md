# LPLCv2 qualitative comparison: eight selected Illegible-labelled samples

The companion script, `scripts/select_lplcv2_qualitative_examples.py`, reads the **already completed frozen 7,023-sample padded LPLCv2 evaluation** and selects eight examples for visual review. It never runs inference, changes predictions, or modifies the frozen source dataset.

The three displayed recognizers are the best three on **Illegible-labelled exact-match recognition**: **Ours (43.55%)**, **SVTRv2 (29.37%)**, and **SVTRv2-AR (25.80%)**. This ranking differs from the overall 7,023-image ranking.

## Selection rules

The selection is deliberately illustrative, not representative of the test distribution:

- **4 exact matches:** Ours is correct in all seven positions; each comparison model has at least one incorrect position.
- **2 one-position errors:** Ours matches exactly six of seven positions; each comparison model matches no more than five.
- **2 larger partial errors:** Ours matches four or five positions (two or three wrong); each comparison model matches no more than three.

The script uses the evaluation's **position-wise match definition**, not edit distance. It avoids repeating any original source photograph or annotated text among the eight selected cases. By default it ranks eligible candidates by the margin between Ours and the better comparison model, with seed `2026` for reproducibility. `--selection random` draws instead from eligible candidates, while retaining uniqueness requirements.

All cases come from the original **Illegible source annotation category**; this is **not** an independent determination that characters are unreadable to a human. A cropped sample whose annotation cannot be visually verified requires editorial scrutiny, not a post-hoc change to the quantitative evaluation.

## Run on daugman

```bash
conda activate yolov11
cd ~/doc2025/project1
git checkout revision/reviewer-eval-tooling
git pull --ff-only
python -m pip install Pillow  # only if PIL is not already installed

SCRIPT=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/select_lplcv2_qualitative_examples.py
python "$SCRIPT" --dry-run
python "$SCRIPT"
```

By default, the script uses:

- Dataset: `~/doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Padded20pct_7023`.
- Three prediction directories: `experiments/revision_eval/reviewer1/02_cross_dataset/results/lplcv2_acq7023_padded_{ours,svtrv2,svtrv2_ar}/`.
- Output directory: `experiments/revision_eval/reviewer1/02_cross_dataset/results/qualitative_illegible_7023/`.

No new GPU inference or checkpoint loading is necessary. The script verifies the frozen manifest fingerprint, row IDs, ground-truth transcriptions, per-sample positional-match counts, completed aggregate metrics, and the source dataset in each metric file. It refuses to overwrite an existing output directory.

### Outputs

- `contact_sheet.png` — a 4-column by 2-row overview of all eight panels.
- `panel_01_full.png` to `panel_08_two_three_errors.png` — eight independently readable panels, with the actual filenames reflecting the selected categories.
- `selection_private.json` — local-only provenance, including source-image identity, ground truth, and predictions; **do not publish**.
- `README_REVIEW_FIRST.txt` — restrictions and the interpretation of the selection.

Each panel contains the original padded crop enlarged with nearest-neighbor interpolation **for display** (not a pixel-identical rendering of the network's 96×32 bicubic/normalized input). The ground truth and all three outputs are printed below it; red characters disagree with the annotated position and green characters agree. The figure contains real license plate text.

### Regenerating after visual review

If you decide not to use an image (e.g., OCR text appears incorrect, a crop is clipped, or the panel is not visually informative), create a **local-only** exclusions file, one virtual track ID per line, for example:

```bash
printf '%s\n' track_000123 track_000456 > /tmp/qualitative_exclude.txt
python "$SCRIPT" --exclude /tmp/qualitative_exclude.txt \
    --output experiments/revision_eval/reviewer1/02_cross_dataset/results/qualitative_illegible_7023_v2
```

If a category cannot be filled with the default constraints, the script raises an error displaying the number of eligible candidates. You may adjust `--hard-max-peer` (default `3`) or `--partial-max-peer` (default `5`) to relax the comparison constraints, but any relaxed criteria should be reported faithfully when choosing the final figure.

## Before showing examples in the paper or pushing images to GitHub

The output directory is beneath `results/`, which is Git-ignored by this repository. **Do not force-add the results directory.** Review dataset redistribution terms, journal policies, and applicable privacy requirements before publishing identifiable license plate imagery and transcriptions. Do not push source photographs, full annotation manifests, the 7,023 prediction CSVs, checkpoints, or `selection_private.json`.

Once the images are cleared for sharing, copy **only the selected contact sheet and panel PNGs** into a separately reviewed publication folder. Alternatively, attach just the overview image in this chat for visual feedback without publishing the underlying dataset files in a repository.

A suitable manuscript description is: *"Eight illustrative examples selected from the Illegible-labelled LPLCv2 evaluation set; model-performance-conditioned selection is not representative of the full dataset."* The quantitative comparisons must continue to use all 7,023 eligible samples, unchanged.

## LaTeX figure export (same eight already selected images)

The new exporter produces a *native LaTeX* 4-by-2 comparison, using the
**existing fixed selection**, without loading checkpoints, rerunning inference,
reselecting samples, or altering any source image. LaTeX typesets the labels
and model predictions under each original padded crop, with green/red
per-position comparisons.

On daugman, from the experiment repository root:

```bash
conda activate yolov11
cd ~/doc2025/project1
git pull --ff-only
EXPORT=experiments/revision_eval/reviewer1/02_cross_dataset/scripts/export_lplcv2_qualitative_latex.py
python "$EXPORT" --dry-run
python "$EXPORT"
cd experiments/revision_eval/reviewer1/02_cross_dataset/results/qualitative_illegible_7023/latex_figure
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
```

The script reads the local `selection_private.json`, validates all eight
entries against the frozen padded dataset and all three original prediction
CSVs, and verifies their 4+2+2 categories and selection constraints.
It refuses to overwrite an existing output directory.

A new ignored `latex_figure/` directory is created alongside the
existing PNG panels. It contains:

- `figure.tex`: a two-column-width `figure*`, four examples per row,
  with original crop, source legibility label, ground truth, the three
  model predictions and their matching-position counts.
- `crops/example_01.png` through `example_08.png`: byte-for-byte
  copies of the native padded license plate crops, with no resizing or text.
- `preview.tex`: a standalone preview requiring standard LaTeX
  packages `graphicx`, `xcolor`, and `geometry`.
- `REVIEW_FIRST.txt`: a reminder about permission and review.

The generated figure caption uses the paper's `\VN{...}` revision
convention. The source itself defines `\LPLQualPath` to the default
manuscript-relative path `4-experiments/qualitative_lplcv2`.
The standalone preview overrides that path to the local directory.

After **independently checking publication/redistribution permission,
privacy, the transcription quality, and the visual crops**, copy only
the reviewed image files and generated figure source to the paper
repository:

```bash
cd ~/doc2025/project1
SOURCE=experiments/revision_eval/reviewer1/02_cross_dataset/results/qualitative_illegible_7023/latex_figure
PAPER=~/doc2025/CNN-ViT-LRLPR-paper
mkdir -p "$PAPER/4-experiments/qualitative_lplcv2/crops"
cp "$SOURCE/figure.tex" "$PAPER/4-experiments/qualitative_lplcv2/"
cp "$SOURCE"/crops/*.png "$PAPER/4-experiments/qualitative_lplcv2/crops/"
```

Then add this line in the Cross-Dataset Experimental Results subsection:

```tex
\input{4-experiments/qualitative_lplcv2/figure.tex}
```

Neither repository's manuscript source is edited automatically. Compile the
paper normally after you insert the figure. If images are not cleared
for public redistribution, keep the generated package local or share
it privately for visual review. In particular, **the generated LaTeX
contains real transcriptions and predictions**, so do not force-add it,
the crops, the selection metadata, or the full prediction CSVs to a
public repository without clearance.

To export a different vetted selection, supply `--selection PATH`.
To create another output without overwriting, use `--output NEW_PATH`.
