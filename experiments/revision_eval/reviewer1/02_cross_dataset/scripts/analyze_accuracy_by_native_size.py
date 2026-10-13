#!/usr/bin/env python3
"""Post-hoc descriptive analysis: F=1 OCR accuracy by native plate crop size.

Read-only comparison of frozen sample manifest and model prediction CSV.
Does not alter selection, crop data or model predictions. Prespecified,
disjoint pixel-size groups; bin results are exploratory, not causal.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


WIDTH_BINS = (
    ("<=32", lambda n: n <= 32),
    ("33-48", lambda n: 33 <= n <= 48),
    ("49-64", lambda n: 49 <= n <= 64),
    ("65-96", lambda n: 65 <= n <= 96),
    (">96", lambda n: n > 96),
)
HEIGHT_BINS = (
    ("<=12", lambda n: n <= 12),
    ("13-16", lambda n: 13 <= n <= 16),
    ("17-24", lambda n: 17 <= n <= 24),
    ("25-32", lambda n: 25 <= n <= 32),
    (">32", lambda n: n > 32),
)


def distribution(rows, kind):
    bins = WIDTH_BINS if kind == "width" else HEIGHT_BINS
    value_key = "crop_width" if kind == "width" else "crop_height"
    result = []
    total = sum(1 for _ in rows)
    for label, fn in bins:
        chunk = [x for x in rows if fn(x[value_key])]
        n = len(chunk)
        matched = sum(x["correct"] for x in chunk)
        char_total = 7 * n
        slot_correct = sum(x["positional_matches"] for x in chunk)
        edit_errors = sum(x["edit_distance"] for x in chunk)
        result.append({
            "range_px": label,
            "n": n,
            "coverage_percent": round(100 * n / total, 3) if total else None,
            "rr_percent": round(100 * matched / n, 3) if n else None,
            "character_accuracy_percent": round(100 * slot_correct / char_total, 3) if char_total else None,
            "cer_percent": round(100 * edit_errors / char_total, 3) if char_total else None,
            "correct": matched,
        })
    if sum(x["n"] for x in result) != total:
        raise RuntimeError("Native-size bins do not partition the full dataset")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", type=Path, help="Optional new JSON summary path")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) == 0:
        raise ValueError("Empty selection manifest")
    lookup = {}
    for sample in samples:
        track = sample["track_id"]
        if track in lookup:
            raise ValueError("Duplicate track in selection manifest: " + track)
        lookup[track] = sample
    aligned = []
    seen = set()
    with args.predictions.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"track", "ground_truth", "prediction", "correct",
                    "positional_matches", "edit_distance", "layout"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Prediction CSV missing fields: " +
                             str(sorted(required - set(reader.fieldnames or []))))
        for item in reader:
            track = item["track"]
            if track not in lookup:
                raise ValueError("Unexpected prediction track: " + track)
            if track in seen:
                raise ValueError("Duplicate prediction track: " + track)
            seen.add(track)
            original = lookup[track]
            if item["ground_truth"] != original["ocr"]:
                raise ValueError("GT mismatch for " + track)
            correct = item["prediction"] == original["ocr"]
            if item["correct"].strip().lower() != str(correct).lower():
                raise ValueError("Correctness mismatch for " + track)
            aligned.append({
                **original,
                "correct": int(correct),
                "positional_matches": int(item["positional_matches"]),
                "edit_distance": int(item["edit_distance"]),
                "layout": item["layout"],
            })
    if len(aligned) != len(samples):
        missing = sorted(set(lookup) - seen)
        raise ValueError(f"Missing {len(missing)} predictions; first: {missing[:5]}")
    if not all(len(x["ocr"]) == 7 for x in aligned):
        raise ValueError("Not all OCR labels have seven characters")
    n = len(aligned)
    correct = sum(x["correct"] for x in aligned)
    overview = {
        "count": n, "correct": correct,
        "rr_percent": round(100 * correct / n, 3),
        "native_width_le_96": sum(x["crop_width"] <= 96 for x in aligned),
        "native_height_le_32": sum(x["crop_height"] <= 32 for x in aligned),
        "both_width_le_96_height_le_32": sum(
            x["crop_width"] <= 96 and x["crop_height"] <= 32 for x in aligned),
        "layout_counts": dict(Counter(x["layout"] for x in aligned)),
    }
    result = {
        "analysis": "post-hoc descriptive accuracy by original plate bounding box size",
        "manifest": str(args.manifest),
        "predictions": str(args.predictions),
        "overall": overview,
        "width_groups": distribution(aligned, "width"),
        "height_groups": distribution(aligned, "height"),
        "caveat": ("Original crop dimensions are not proof of sensor-native resolution; "
                   "different legibility, layout, camera, and lighting conditions "
                   "can confound these descriptive size-stratified results."),
    }
    content = json.dumps(result, indent=2) + "\n"
    print(content, end="")
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite existing analysis output: " +
                                  str(args.output))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
