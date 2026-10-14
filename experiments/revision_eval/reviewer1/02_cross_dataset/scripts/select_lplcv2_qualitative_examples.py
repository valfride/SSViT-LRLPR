#!/usr/bin/env python3
"""Select eight reproducible, illustrative LPLCv2 Illegible-labelled examples.

Reads *existing* verified padded-7,023 F=1 prediction CSVs and frozen crop PNGs.
No inference, training, image editing, or changes to the evaluation dataset.

Top three on Illegible-labelled exact-match recognition: Ours, SVTRv2,
SVTRv2-AR. Examples are intentionally performance-conditioned and must NOT be
presented as representative or used for statistical evaluation.

RESTRICTED OUTPUT: Panels reveal license-plate text. Keep outputs local until
image redistribution, dataset license, and privacy conditions are reviewed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path.home() / "doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Padded20pct_7023"
DEFAULT_OUTPUT = BASE / "results/qualitative_illegible_7023"
MANIFEST_SHA256 = "107d53e3c2f48ecfe118bb222a828080f6750632b7c9b51a523ef186347ffb48"
EXPECTED_COUNT = 7023
EXPECTED_GROUPS = {"Poor": 6240, "Illegible": 783}
MODEL_KEYS = {"Ours": "ours", "SVTRv2": "svtrv2", "SVTRv2-AR": "svtrv2_ar"}
MODEL_ORDER = tuple(MODEL_KEYS)
PANEL_W, PANEL_H = 560, 430
BG = "#f7f8fa"
GREEN = "#237047"
RED = "#b73831"
MUTED = "#596274"
DARK = "#18212b"


@dataclass(frozen=True)
class Prediction:
    text: str
    matches: int


def normalize(value: str) -> str:
    return "".join(c for c in str(value or "").upper() if c.isalnum())


def position_matches(pred: str, gt: str) -> int:
    return sum(i < len(pred) and pred[i] == gt[i] for i in range(7))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(dataset: Path) -> dict[str, dict]:
    source = dataset / "selection_manifest.json"
    actual = sha256(source)
    if actual != MANIFEST_SHA256:
        raise ValueError(f"Unexpected frozen manifest SHA256: {actual}")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    groups = Counter(row["source_legibility"] for row in samples)
    if len(samples) != EXPECTED_COUNT or dict(groups) != EXPECTED_GROUPS:
        raise ValueError(f"Unexpected sample count / categories: {len(samples)}, {groups}")
    by_track = {row["track_id"]: row for row in samples}
    if len(by_track) != EXPECTED_COUNT:
        raise ValueError("Duplicate track IDs in the frozen manifest")
    for row in samples:
        if len(row["ocr"]) != 7 or row["source_legibility"] not in EXPECTED_GROUPS:
            raise ValueError("Invalid source transcription or category")
    info = json.loads((dataset / "dataset_info.json").read_text(encoding="utf-8"))
    if info.get("variant") != "padded" or info.get("manifest_sha256") != actual:
        raise ValueError("Dataset-info record does not match the padded evaluation set")
    return by_track


def read_predictions(dataset: Path, by_track: dict, key: str) -> dict[str, Prediction]:
    directory = BASE / "results" / f"lplcv2_acq7023_padded_{key}"
    path = directory / "F1_bayes_predictions.csv"
    metrics_path = directory / "F1_bayes.json"
    if not path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing completed prediction/metrics files in {directory}")
    metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    if Path(metrics_doc["dataset"]).expanduser().resolve() != dataset:
        raise ValueError(f"Wrong source dataset in {metrics_path}")
    reported = metrics_doc["metrics"]
    if int(reported["sequence_total"]) != EXPECTED_COUNT:
        raise ValueError(f"Wrong evaluation sample count in {metrics_path}")
    out: dict[str, Prediction] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"track", "prediction", "ground_truth", "positional_matches", "correct"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV columns missing from {path}")
        for row in reader:
            tid = row["track"]
            if tid not in by_track or tid in out:
                raise ValueError(f"Unknown or duplicated track in {path}: {tid}")
            truth = normalize(row["ground_truth"])
            if truth != by_track[tid]["ocr"]:
                raise ValueError(f"GT disagrees with frozen manifest: {tid}")
            pred = normalize(row["prediction"])
            matches = position_matches(pred, truth)
            if int(row["positional_matches"]) != matches:
                raise ValueError(f"Positional accuracy disagreement: {key}, {tid}")
            if row["correct"].lower() not in {"true", "false"}:
                raise ValueError(f"Unexpected exact-match flag: {key}, {tid}")
            if (row["correct"].lower() == "true") != (pred == truth):
                raise ValueError(f"Exact-match flag disagrees: {key}, {tid}")
            out[tid] = Prediction(pred, matches)
    if set(out) != set(by_track):
        raise ValueError(f"Incomplete {key} results ({len(out)} / {len(by_track)})")
    actual_correct = sum(out[tid].text == by_track[tid]["ocr"] for tid in out)
    if actual_correct != int(reported["sequence_correct"]):
        raise ValueError(f"Aggregate total mismatch for {key}")
    return out


def choose(by_track: dict, results: dict, excluded: set[str], seed: int,
           full_max: int, partial_max: int, hard_max: int, mode: str):
    buckets = {"full": [], "one_error": [], "two_three_errors": []}
    for tid, row in by_track.items():
        if row["source_legibility"] != "Illegible" or tid in excluded:
            continue
        ours = results["Ours"][tid].matches
        peers = [results[name][tid].matches for name in MODEL_ORDER[1:]]
        peer_max = max(peers)
        if ours == 7 and peer_max <= full_max:
            buckets["full"].append(tid)
        if ours == 6 and peer_max <= partial_max:
            buckets["one_error"].append(tid)
        if ours in {4, 5} and peer_max <= hard_max:
            buckets["two_three_errors"].append(tid)

    counts = {name: len(items) for name, items in buckets.items()}
    requirements = {"full": 4, "one_error": 2, "two_three_errors": 2}
    if any(counts[name] < required for name, required in requirements.items()):
        raise ValueError(
            f"Insufficient eligible examples: {counts}, required {requirements}. "
            "Try higher --full-max-peer / --partial-max-peer / --hard-max-peer thresholds."
        )

    rng = random.Random(seed)
    chosen = []
    seen_photos, seen_labels = set(), set()
    for category, amount in requirements.items():
        candidates = list(buckets[category])
        rng.shuffle(candidates)  # reproducible tie-breaking
        if mode == "contrast":
            candidates.sort(key=lambda tid: (
                results["Ours"][tid].matches - max(results[n][tid].matches for n in MODEL_ORDER[1:]),
                -sum(results[n][tid].matches for n in MODEL_ORDER[1:]),
            ), reverse=True)
        picked = []
        for tid in candidates:
            row = by_track[tid]
            if row["source_image"] in seen_photos or row["ocr"] in seen_labels:
                continue
            picked.append((category, tid))
            seen_photos.add(row["source_image"])
            seen_labels.add(row["ocr"])
            if len(picked) == amount:
                break
        if len(picked) < amount:
            raise ValueError(f"Not enough independent source-photo/label examples for {category}: "
                             f"found {len(picked)} of {amount}, despite {counts[category]} candidates")
        chosen.extend(picked)
    return chosen, counts


def font(size: int, bold=False):
    files = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf"]
        if bold else ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                      "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"]
    )
    for path in files:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, f) -> int:
    return round(draw.textlength(text, font=f))


def colored_prediction(draw, x: int, y: int, pred: str, gt: str, f):
    if not pred:
        draw.text((x, y), "<empty>", font=f, fill=RED)
        return
    # Individual characters are red when they disagree at that position.
    for i, char in enumerate(pred):
        draw.text((x, y), char, font=f, fill=GREEN if i < 7 and char == gt[i] else RED)
        x += text_width(draw, char, f)


def render_panel(label: str, example_no: int, row: dict, preds: dict,
                 crop: Path) -> Image.Image:
    if not crop.is_file():
        raise FileNotFoundError(crop)
    with Image.open(crop) as fp:
        original = fp.convert("RGB")
    panel = Image.new("RGB", (PANEL_W, PANEL_H), BG)
    d = ImageDraw.Draw(panel)
    d.rounded_rectangle((5, 5, PANEL_W-6, PANEL_H-6), radius=12, outline="#bfc7d1", width=2)
    headers = {"full": "Exact match (both baselines wrong)",
               "one_error": "Our model: one wrong position",
               "two_three_errors": "Our model: 2-3 wrong positions"}
    d.text((18, 14), f"{example_no:02d}  {headers[label]}", font=font(19, True), fill=DARK)

    frame = (18, 52, PANEL_W-18, 215)
    d.rounded_rectangle(frame, radius=8, fill="#e6e9ed")
    fitted = ImageOps.contain(original, (frame[2]-frame[0]-10, frame[3]-frame[1]-10),
                              method=Image.Resampling.NEAREST)
    px = frame[0] + (frame[2]-frame[0]-fitted.width)//2
    py = frame[1] + (frame[3]-frame[1]-fitted.height)//2
    panel.paste(fitted, (px, py))
    d.text((20, 219), f"Source label: Illegible   |   native crop: {original.width} x {original.height} px",
           font=font(13), fill=MUTED)

    gt = row["ocr"]
    f_label = font(19, True)
    f_pred = font(25, True)
    d.text((25, 250), "Ground truth", font=f_label, fill=DARK)
    d.text((270, 246), gt, font=f_pred, fill=DARK)
    d.line((20, 281, PANEL_W-20, 281), fill="#d8dde4", width=1)
    for j, model in enumerate(MODEL_ORDER):
        y = 289 + j*43
        p = preds[model]
        d.text((25, y+3), model, font=f_label, fill=DARK)
        colored_prediction(d, 270, y, p.text, gt, f_pred)
        d.text((480, y+4), f"{p.matches}/7", font=font(18, True),
               fill=GREEN if p.matches==7 else RED)
    return panel


def render_outputs(args, chosen, by_track, predictions, counts, dataset):
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for _, tid in chosen:
        crop = dataset / "crops" / tid / "lr-001.png"
        if not crop.is_file():
            raise FileNotFoundError(crop)
    output.mkdir(parents=True, exist_ok=False)
    cards=[]
    private=[]
    for i,(category,tid) in enumerate(chosen, 1):
        row=by_track[tid]
        pred={name:predictions[name][tid] for name in MODEL_ORDER}
        card=render_panel(category,i,row,pred,dataset / "crops" / tid / "lr-001.png")
        filename=f"panel_{i:02d}_{category}.png"
        card.save(output / filename, optimize=True)
        cards.append(card)
        private.append({"panel":filename,"category":category,"track_id":tid,
                        "source_image":row["source_image"],"annotation_index":row["annotation_index"],
                        "source_legibility":row["source_legibility"],"ground_truth":row["ocr"],
                        "predictions":{name:{"text":pred[name].text,"matching_positions":pred[name].matches}
                                       for name in MODEL_ORDER}})
    spacing=16
    margin=24
    header=90
    W=4*PANEL_W+3*spacing+2*margin
    H=header+2*PANEL_H+spacing+margin
    sheet=Image.new("RGB",(W,H),"white")
    d=ImageDraw.Draw(sheet)
    d.text((margin,15),"LPLCv2 — selected Illegible-labelled examples",
           font=font(30,True),fill=DARK)
    d.text((margin,55),"Illustrative, accuracy-conditioned examples; not representative of the full evaluation set.",
           font=font(17),fill=MUTED)
    for i,card in enumerate(cards):
        sheet.paste(card,(margin+(i%4)*(PANEL_W+spacing),header+(i//4)*(PANEL_H+spacing)))
    sheet.save(output / "contact_sheet.png",optimize=True)
    (output/"selection_private.json").write_text(json.dumps({
        "purpose":"Human-review selection only; not a statistical sample",
        "selected_from":"frozen padded LPLCv2 7,023 annotations; source legibility Illegible",
        "manifest_sha256":MANIFEST_SHA256,"seed":args.seed,"ranking":args.selection,
        "thresholds":{"full_max_peer":args.full_max_peer,"partial_max_peer":args.partial_max_peer,
                      "hard_max_peer":args.hard_max_peer},"eligible_counts":counts,
        "examples":private},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (output/"README_REVIEW_FIRST.txt").write_text(
        "RESTRICTED REVIEW OUTPUT — do not publish as-is.\n"
        "Eight selected LPLCv2 source-labelled Illegible images; top three models by "
        "Illegible-subset exact-match accuracy: Ours, SVTRv2, SVTRv2-AR.\n"
        "contact_sheet.png and panel_*.png contain visible license plate imagery, "
        "GT transcription, and predictions.\n"
        "selection_private.json includes source-image identities and must NOT be published.\n"
        "Check image redistribution rights, privacy conditions, OCR label correctness, "
        "and human readability independently BEFORE selecting panels for a paper.\n"
        "The examples were selected by model performance and are not representative; "
        "do not infer rates, statistical significance, or verified human readability.\n"
        "Original full evaluation predictions, datasets, and checkpoint selections were unchanged.\n",
        encoding="utf-8")
    print(f"Saved {len(chosen)} panels + combined contact sheet to: {output}")
    print("Review panels first; do not upload the private selection metadata or full result CSVs.")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",type=Path,default=DEFAULT_DATASET)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--selection",choices=("contrast","random"),default="contrast")
    parser.add_argument("--full-max-peer",type=int,choices=range(0,7),default=6,
                        help="Four fully correct examples: each other model has <= this many matching positions")
    parser.add_argument("--partial-max-peer",type=int,choices=range(0,7),default=5,
                        help="Two one-error examples: each other model has <= this many matching positions")
    parser.add_argument("--hard-max-peer",type=int,choices=range(0,7),default=3,
                        help="Two 2-3-error examples: each other model has <= this many matching positions")
    parser.add_argument("--exclude",type=Path,help="Optional local text file with one track ID per line to skip after visual review")
    parser.add_argument("--dry-run",action="store_true",help="Validate results and show candidate counts; create no files")
    args=parser.parse_args()
    dataset=args.dataset.expanduser().resolve(strict=True)
    by_track=read_manifest(dataset)
    predictions={name:read_predictions(dataset,by_track,key) for name,key in MODEL_KEYS.items()}
    excluded=set()
    if args.exclude:
        excluded={line.strip() for line in args.exclude.expanduser().read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")}
        unknown=excluded-set(by_track)
        if unknown:
            raise ValueError(f"Unrecognized excluded track IDs: {sorted(unknown)[:5]}")
    chosen,counts=choose(by_track,predictions,excluded,args.seed,
                         args.full_max_peer,args.partial_max_peer,args.hard_max_peer,args.selection)
    print("Eligible candidates by category:",counts)
    print("Selection:",[(group,tid) for group,tid in chosen])
    if not args.dry_run:
        render_outputs(args,chosen,by_track,predictions,counts,dataset)


if __name__=="__main__":
    main()