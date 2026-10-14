#!/usr/bin/env python3
"""Export the ALREADY SELECTED eight LPLCv2 examples as a native LaTeX figure.

Reads selection_private.json from select_lplcv2_qualitative_examples.py,
revalidates every selected ground truth and all three frozen predictions,
and copies ONLY the eight original padded PNG crops into a local, Git-ignored
LaTeX package. No image annotation, resampling, inference, or reselection.

RESTRICTED: Generated TeX contains real transcriptions and predictions.
Do not commit the generated package before image/privacy/license review.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from select_lplcv2_qualitative_examples import (
    DEFAULT_DATASET, DEFAULT_OUTPUT, MANIFEST_SHA256, MODEL_KEYS,
    MODEL_ORDER, normalize, position_matches, read_manifest, read_predictions,
    sha256,
)

DEFAULT_SELECTION = DEFAULT_OUTPUT / "selection_private.json"
DEFAULT_LATEX_OUTPUT = DEFAULT_OUTPUT / "latex_figure"
EXPECTED_CATEGORIES = {"full": 4, "one_error": 2, "two_three_errors": 2}
PEER_LIMIT_KEYS = {
    "full": "full_max_peer",
    "one_error": "partial_max_peer",
    "two_three_errors": "hard_max_peer",
}
CATEGORY_TITLES = {
    "full": "Exact match",
    "one_error": "One wrong position",
    "two_three_errors": "Two--three wrong",
}
# Set this to the repository-relative destination when copying cleared assets
# into the manuscript. Override in LaTeX with \renewcommand if required.
PAPER_IMAGE_FOLDER = "4-experiments/qualitative_lplcv2"


def verify_selection(selection: Path, dataset: Path):
    doc = json.loads(selection.read_text(encoding="utf-8"))
    examples = doc.get("examples")
    if doc.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("Selection belongs to a different frozen manifest")
    if not isinstance(examples, list) or len(examples) != 8:
        raise ValueError("Expected exactly eight previously selected examples")
    if Counter(e.get("category") for e in examples) != EXPECTED_CATEGORIES:
        raise ValueError("Selection does not have the original 4+2+2 categories")

    by_track = read_manifest(dataset)
    predictions = {
        model: read_predictions(dataset, by_track, key)
        for model, key in MODEL_KEYS.items()
    }
    thresholds = doc.get("thresholds", {})
    for name in PEER_LIMIT_KEYS.values():
        if type(thresholds.get(name)) is not int:
            raise ValueError("Missing original selection threshold: " + name)

    seen_tracks, seen_photos, seen_labels = set(), set(), set()
    verified = []
    for i, item in enumerate(examples, 1):
        tid, category = item["track_id"], item["category"]
        if tid not in by_track or tid in seen_tracks:
            raise ValueError("Unknown or repeated selected track: " + tid)
        row = by_track[tid]
        truth = row["ocr"]
        if (row["source_legibility"] != "Illegible"
                or item["ground_truth"] != truth
                or item["source_image"] != row["source_image"]
                or item["annotation_index"] != row["annotation_index"]):
            raise ValueError("Selection source metadata has changed: " + tid)
        if row["source_image"] in seen_photos or truth in seen_labels:
            raise ValueError("Selected samples share a source photograph or label")
        seen_tracks.add(tid)
        seen_photos.add(row["source_image"])
        seen_labels.add(truth)

        checked = {}
        for model in MODEL_ORDER:
            predicted = predictions[model][tid]
            stored = item["predictions"][model]
            if (stored["text"] != predicted.text
                    or stored["matching_positions"] != predicted.matches
                    or position_matches(predicted.text, truth) != predicted.matches):
                raise ValueError("Prediction drift: " + model + " / " + tid)
            checked[model] = predicted

        ours = checked["Ours"].matches
        peers = max(checked[m].matches for m in MODEL_ORDER[1:])
        if (category == "full" and ours != 7
                or category == "one_error" and ours != 6
                or category == "two_three_errors" and ours not in (4, 5)
                or peers > thresholds[PEER_LIMIT_KEYS[category]]):
            raise ValueError("Selection category rules no longer hold: " + tid)

        crop = dataset / "crops" / tid / "lr-001.png"
        if not crop.is_file():
            raise FileNotFoundError(crop)
        with Image.open(crop) as im:
            if im.format != "PNG" or im.width < 1 or im.height < 1:
                raise ValueError("Invalid original PNG crop: " + str(crop))
            size = im.size
            im.verify()
        verified.append((i, category, crop, size, truth, checked))
    return verified


def colored_prediction(pred: str, truth: str) -> str:
    """Monospace per-position colors; safely handles empty/variable-length OCR."""
    if not pred:
        return r"\textcolor{red!75!black}{\ensuremath{\varnothing}}"
    if normalize(pred) != pred:
        raise ValueError("Predictions must be normalized alphanumeric text")
    pieces = []
    for i, c in enumerate(pred):
        color = "green!45!black" if i < len(truth) and c == truth[i] else "red!75!black"
        pieces.append(r"\textcolor{" + color + "}{" + c + "}")
    return r"{\ttfamily " + "".join(pieces) + "}"


def latex_card(example) -> str:
    number, category, _crop, size, truth, checked = example
    letter = chr(ord("a") + number - 1)
    category_title = (f"{7 - checked['Ours'].matches} wrong positions"\n                      if category == "two_three_errors"\n                      else CATEGORY_TITLES[category])
    parts = [
        r"\begin{minipage}[t]{0.235\textwidth}",
        r"\vspace{0pt}\centering",
        r"{\fontsize{7.5}{9}\selectfont\bfseries (" + letter + ") " + category_title + r"}\par\vspace{1mm}",
        r"{\setlength{\fboxsep}{1pt}"
        r"\fbox{\includegraphics[width=\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax,"
        r"height=0.73in,keepaspectratio]{\LPLQualPath/crops/example_"
        + f"{number:02d}" + r".png}}}\par",
        r"{\fontsize{6.5}{7.7}\selectfont\color{gray}"
        + "Illegible; native " + str(size[0]) + r"$\times$" + str(size[1])
        + r" px}\par\vspace{0.7mm}",
        r"{\fontsize{8}{9.8}\selectfont",
        r"\begin{tabular}{@{}l@{\hspace{2pt}}l@{\hspace{2pt}}r@{}}",
        r"\textbf{GT} & {\ttfamily " + truth + r"} & --\\",
    ]
    for model in MODEL_ORDER:
        pred = checked[model]
        label = r"\textbf{Ours}" if model == "Ours" else model
        parts.append(label + " & " + colored_prediction(pred.text, truth)
                     + " & " + str(pred.matches) + r"/7\\")
    parts.extend([r"\end{tabular}}", r"\end{minipage}"])
    return "\n".join(parts)


def render_figure(examples) -> str:
    cards = [latex_card(e) for e in examples]\n    last_matches = [e[5]["Ours"].matches for e in examples[6:]]\n    last_clause = (\n        f"{last_matches[0]} in both (g) and (h)."\n        if last_matches[0] == last_matches[1]\n        else f"{last_matches[0]} in (g) and {last_matches[1]} in (h)."\n    )
    grid = (" &\n".join(cards[:4]) + r" \\[2.5mm]" + "\n"
            + " &\n".join(cards[4:]) + r" \\" + "\n")
    return "\n".join([
        "% GENERATED FROM A PRIVATE, FROZEN LPLCv2 SELECTION. NOT CLEARED FOR PUBLICATION.",
        "% Copy figure.tex and crops/ together to the paper's destination folder.",
        r"\providecommand{\LPLQualPath}{" + PAPER_IMAGE_FOLDER + "}",
        r"\begin{figure*}[!t]",
        r"\centering",
        r"\begingroup",
        r"\setlength{\tabcolsep}{0pt}",
        r"\begin{tabular}{@{}c@{\hspace{1.5mm}}c@{\hspace{1.5mm}}c@{\hspace{1.5mm}}c@{}}",
        grid.rstrip("\n"),
        r"\end{tabular}",
        r"\endgroup",
        r"\caption{\VN{Eight performance-selected examples from the LPLCv2",
        r"subset labelled Illegible ($F=1$). In (a)--(d), our model matches all",
        r"seven annotated characters while both comparison models do not; it",
        r"matches six in (e)--(f) and " + last_clause,
        r"Green and red characters indicate position-wise agreement and",
        r"disagreement with the source transcription, respectively.",
        r"Panels show native-resolution padded crops enlarged for",
        r"visualization; all recognizers were evaluated on the corresponding",
        r"crops bicubically resized to $96\times32$.",
        r"Illegible is a source annotation, not independently verified",
        r"human unreadability. These deliberately selected examples are not",
        r"representative of the full evaluation set.}}",
        r"\label{fig:lplcv2_qualitative}",
        r"\end{figure*}",
        "",
    ])


def preview_source() -> str:
    return "\n".join([
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=17mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\providecommand{\VN}[2][]{#2}",
        r"\newcommand{\LPLQualPath}{.}",
        r"\begin{document}",
        r"\input{figure.tex}",
        r"\end{document}",
        "",
    ])


def write_package(output: Path, examples):
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing output: " + str(output))
    # Destination is private by default; no file is written outside output.
    output.mkdir(parents=True, exist_ok=False)
    crops = output / "crops"
    crops.mkdir()
    for i, _, path, _, _, _ in examples:
        destination = crops / f"example_{i:02d}.png"
        shutil.copyfile(path, destination)  # preserve original crop bytes
        if sha256(path) != sha256(destination):
            raise RuntimeError("Copied image differs from the frozen source")
    (output / "figure.tex").write_text(render_figure(examples), encoding="utf-8")
    (output / "preview.tex").write_text(preview_source(), encoding="utf-8")
    (output / "REVIEW_FIRST.txt").write_text(
        "PRIVATE QUALITATIVE FIGURE: figure.tex reveals full transcriptions "
        "and individual model predictions. Crops contain real license plate imagery.\n"
        "Review visual correctness, redistribution rights, and privacy before "
        "publishing. Do NOT commit selection_private.json or full result CSVs.\n"
        "Preview: cd this directory && pdflatex -interaction=nonstopmode "
        "-halt-on-error preview.tex\n"
        "For manuscript integration, copy the reviewed figure.tex + crops/ "
        "to 4-experiments/qualitative_lplcv2/ within the paper repository; "
        "then add \\input{4-experiments/qualitative_lplcv2/figure.tex} "
        "to the Cross-Dataset Results subsection.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="Previously generated, local selection_private.json")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_LATEX_OUTPUT,
                        help="New local output directory; must not already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate saved selection and all predictions, without writing")
    args = parser.parse_args()
    selection = args.selection.expanduser().resolve(strict=True)
    dataset = args.dataset.expanduser().resolve(strict=True)
    examples = verify_selection(selection, dataset)
    counts = Counter(category for _, category, *_ in examples)
    print("Verified existing eight examples:", dict(counts))
    print("Original padded PNG crops and all three model outputs match the frozen results.")
    if args.dry_run:
        return
    output = args.output.expanduser().absolute()
    write_package(output, examples)
    print("LaTeX package saved:", output)
    print("Preview: cd " + str(output) +
          " && pdflatex -interaction=nonstopmode -halt-on-error preview.tex")
    print("Review rights and the images before copying any assets to the paper repository.")


if __name__ == "__main__":
    main()
