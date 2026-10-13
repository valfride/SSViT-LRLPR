#!/usr/bin/env python3
"""Make a reproducible prediction-blind, stratified manual annotation QA sample.

Draw random annotated images from auto-flagged and unflagged strata, render
review sheets, and save a review CSV with blank reviewer fields. The user
annotates the CSV manually. --summarize reports actual reviewed status counts,
with stratum-weighted proportions *only when every drawn row is reviewed*.

No model predictions are read and the original LMDB/manifest is untouched.
Restricted dataset images and per-plate labels stay in local output folders.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from audit_lplcv2_annotation_quality import FIELDS, render_pages


STATUSES = (
    "valid_target",
    "unverifiable_text",
    "wrong_box_or_target",
    "wrong_ocr_label",
    "uncertain",
)
REVIEW_FIELDS = ["review_stratum", *FIELDS]


def records_from_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Unexpected QA CSV fields: {path}")
        result = list(reader)
    for row in result:
        for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "crop_width", "crop_height"):
            row[key] = int(row[key])
    return result


def write_csv(path, items):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(items)


def prepare(args):
    source = args.qa_dir.expanduser().resolve()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Won't overwrite existing review: {output}")
    summary = json.loads((source / "qa_summary.json").read_text(encoding="utf-8"))
    population = records_from_csv(source / "qa_samples.csv")
    if not population or len(population) != summary["samples"]:
        raise ValueError("QA CSV count differs from full annotation QA summary")
    if len({r["track_id"] for r in population}) != len(population):
        raise ValueError("Duplicate track IDs in QA population")
    if len({(r["source_image"], r["annotation_index"]) for r in population}) != len(population):
        raise ValueError("Duplicate source-image annotation identities in QA population")
    if not (root / "images").is_dir():
        raise FileNotFoundError("Expected source photograph images/ directory")
    flagged = [dict(x) for x in population if x["flags"]]
    unflagged = [dict(x) for x in population if not x["flags"]]
    if len(flagged) != summary["unique_flagged_samples"]:
        raise ValueError("Flag-stratum count differs from full QA summary")
    if args.flagged > len(flagged) or args.unflagged > len(unflagged):
        raise ValueError(f"Requested {args.flagged}/{args.unflagged} samples, "
                         f"only {len(flagged)}/{len(unflagged)} available")
    n_flagged = args.flagged
    n_unflagged = args.unflagged
    rng = random.Random(args.seed)
    selected_flagged = rng.sample(flagged, n_flagged)
    selected_unflagged = rng.sample(unflagged, n_unflagged)
    for row in selected_flagged:
        row["review_stratum"] = "flagged"
        row["manual_status"] = ""
        row["manual_note"] = ""
    for row in selected_unflagged:
        row["review_stratum"] = "unflagged"
        row["manual_status"] = ""
        row["manual_note"] = ""
    output.mkdir(parents=True)
    review = selected_flagged + selected_unflagged
    if len({r["track_id"] for r in review}) != len(review):
        raise RuntimeError("Duplicate sample IDs")
    write_csv(output / "blind_review.csv", review)
    images = root / "images"
    render_pages(output, "flagged_sample_pages", selected_flagged, images, args.page_size)
    render_pages(output, "unflagged_sample_pages", selected_unflagged, images, args.page_size)
    meta = {
        "protocol": "prediction-blind random stratified annotation-quality review",
        "original_manifest_sha256": summary["manifest_sha256"],
        "qa_sample_count": len(population),
        "seed": args.seed,
        "population": {"flagged": len(flagged), "unflagged": len(unflagged)},
        "draw": {"flagged": n_flagged, "unflagged": n_unflagged},
        "statuses": list(STATUSES),
        "guidance": {
            "valid_target": "Box contains the intended plate, AND the OCR transcription is independently plausible/confirmable from the image.",
            "unverifiable_text": "Plate box is plausibly correct, but characters are too degraded to validate the exact OCR string; not automatically a wrong annotation.",
            "wrong_box_or_target": "Clearly incorrect target: box excludes annotated plate or encloses a non-plate object; do not confuse with a legitimate secondary/partial vehicle.",
            "wrong_ocr_label": "Visible in-box plate transcription clearly disagrees with OCR ground truth; must be independently readable.",
            "uncertain": "Insufficient evidence to classify.",
        },
        "note": ("Flags are review priorities, not proof of error. The evaluator's "
                 "predictions and confidences must not be used to classify cases. "
                 "Never alter frozen datasets or predictions."),
    }
    (output / "review_protocol.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("Review blind_review.csv while viewing pages in the two *_sample_pages folders.")


def summarize(args):
    review_dir = args.output_dir.expanduser().resolve()
    protocol = json.loads((review_dir / "review_protocol.json").read_text(encoding="utf-8"))
    rows = records_from_csv(review_dir / "blind_review.csv")
    expected = protocol["draw"]
    seen = set()
    tallies = {k: Counter() for k in ("flagged", "unflagged")}
    for row in rows:
        track_id = row["track_id"]
        if track_id in seen:
            raise ValueError("Duplicate review record: " + track_id)
        seen.add(track_id)
        stratum = row.get("review_stratum")
        if stratum not in tallies:
            raise ValueError("Invalid review stratum for " + track_id)
        status = row["manual_status"].strip()
        if status and status not in STATUSES:
            raise ValueError("Invalid manual_status for " + track_id + ": " + status)
        tallies[stratum][status or "unreviewed"] += 1
    if any(sum(tallies[s].values()) != expected[s] for s in tallies):
        raise ValueError("Stratum sample counts do not match frozen review protocol")
    if len(rows) != sum(expected.values()):
        raise ValueError("Review sample count changed since protocol creation")
    results = {
        "statuses": list(STATUSES),
        "reviewed_by_stratum": {
            key: {"counts": dict(counts),
                  "reviewed": sum(v for k, v in counts.items() if k != "unreviewed"),
                  "total": expected[key]}
            for key, counts in tallies.items()},
        "complete": all(tallies[s]["unreviewed"] == 0 for s in tallies),
    }
    if results["complete"]:
        if any(expected[k] == 0 for k in tallies):
            raise ValueError("Cannot weight an empty review stratum")
        total_population = sum(protocol["population"].values())
        weighted = {}
        for status in STATUSES:
            weighted[status] = sum(
                protocol["population"][s] / total_population *
                tallies[s][status] / expected[s]
                for s in tallies
            ) * 100
        results["estimated_population_status_percent"] = {
            key: round(value, 3) for key, value in weighted.items()
        }
        results["interpretation"] = (
            "Descriptive stratum-weighted estimates from manually reviewed samples; "
            "they depend on reviewer classification quality and should not be treated "
            "as verified population-level counts. Uncertainty intervals not computed."
        )
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", type=Path,
                        help="Existing full annotation-QA folder with qa_samples.csv")
    parser.add_argument("--dataset-root", type=Path,
                        help="LPLC_dataset folder, used to render review sheets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flagged", type=int, default=100)
    parser.add_argument("--unflagged", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--summarize", action="store_true",
                        help="Read manually edited blind_review.csv; no image rendering")
    args = parser.parse_args()
    if args.summarize:
        summarize(args)
    else:
        if args.qa_dir is None or args.dataset_root is None:
            parser.error("--qa-dir and --dataset-root required for preparation")
        if args.flagged < 1 or args.unflagged < 1:
            parser.error("Both stratum sample sizes must be >=1")
        if not (2 <= args.page_size <= 24 and args.page_size % 2 == 0):
            parser.error("--page-size must be even and between 2 and 24")
        prepare(args)


if __name__ == "__main__":
    main()
