#!/usr/bin/env python3
"""Summarize F=1 predictions by Poor / Illegible without releasing plate text.

Join each prediction to the frozen mixed-LPLCv2 selection manifest by
virtual track ID. Validate every annotation identity, GT string, slot-count,
edit-distance, total sample count, and reported aggregate metrics. Print and
optionally write ONLY aggregate results, never per-image predictions or GT.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_plate(s):
    return "".join(ch for ch in str(s or "").upper() if ch.isalnum())


def positional_matches(pred, gt):
    return sum(
        1 for i in range(7)
        if i < len(pred) and i < len(gt) and pred[i] == gt[i]
    )


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + int(char != other)))
        previous = current
    return previous[-1]


def stats(records):
    n = len(records)
    correct = sum(r["correct"] for r in records)
    matches = sum(r["matches"] for r in records)
    edits = sum(r["edits"] for r in records)
    total_chars = sum(len(r["gt"]) for r in records)
    return {
        "samples": n,
        "sequence_correct": correct,
        "sequence_accuracy_percent": 100.0 * correct / n if n else None,
        "character_correct": matches,
        "character_total": 7 * n,
        "character_accuracy_percent": 100.0 * matches / (7 * n) if n else None,
        "edit_errors": edits,
        "ground_truth_characters": total_chars,
        "cer_percent": 100.0 * edits / total_chars if total_chars else None,
        "partial_6_correct": sum(r["matches"] >= 6 for r in records),
        "partial_5_correct": sum(r["matches"] >= 5 for r in records),
    }


def summarize(dataset, predictions_path, metrics_path, expected_hash):
    dataset = dataset.expanduser().resolve()
    manifest_path = dataset / "selection_manifest.json"
    found_hash = sha256(manifest_path)
    if expected_hash and found_hash != expected_hash:
        raise ValueError("Frozen selection-manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_rows = manifest["samples"]
    if len(source_rows) != 3000:
        raise ValueError("This analysis requires exactly 3,000 frozen manifest rows")
    source_by_track = {row["track_id"]: row for row in source_rows}
    if len(source_by_track) != 3000:
        raise ValueError("Duplicate track IDs in source manifest")
    actual_classes = Counter(row["source_legibility"] for row in source_rows)
    if dict(actual_classes) != {"Poor": 2158, "Illegible": 842}:
        raise ValueError(f"Unexpected source composition: {actual_classes}")

    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    if Path(metadata["dataset"]).expanduser().resolve() != dataset:
        raise ValueError("Metrics JSON evaluated another dataset path")
    if metadata["frames"] != 1 or metadata["mode"] != "val":
        raise ValueError("Metrics JSON is not an F=1 validation result")
    if metadata["tta"] or metadata["swa"]:
        raise ValueError("Unexpected TTA/SWA: use the frozen protocol")
    if metadata["metrics"]["sequence_total"] != 3000:
        raise ValueError("Overall metrics do not report 3,000 samples")

    seen = set()
    by_kind = {"Poor": [], "Illegible": [], "All": []}
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"track", "prediction", "ground_truth", "correct",
                    "positional_matches", "edit_distance"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {required - set(reader.fieldnames or [])}")
        for line_number, entry in enumerate(reader, start=2):
            track = entry["track"]
            if track not in source_by_track:
                raise ValueError(f"Unrecognized track at CSV line {line_number}: {track!r}")
            if track in seen:
                raise ValueError(f"Duplicate prediction for {track}")
            seen.add(track)
            src = source_by_track[track]
            ground_truth = normalize_plate(entry["ground_truth"])
            expected_gt = normalize_plate(src["ocr"])
            if ground_truth != expected_gt or len(ground_truth) != 7:
                raise ValueError(f"GT/manifest mismatch for {track}")
            pred = normalize_plate(entry["prediction"])
            correct = pred == expected_gt
            matches = positional_matches(pred, expected_gt)
            edits = edit_distance(pred, expected_gt)
            if entry["correct"].lower() not in ("true", "false"):
                raise ValueError(f"Invalid CSV correct flag for {track}")
            if correct != (entry["correct"].lower() == "true"):
                raise ValueError(f"Correct flag mismatch for {track}")
            if matches != int(entry["positional_matches"]):
                raise ValueError(f"Positional match count mismatch for {track}")
            if edits != int(entry["edit_distance"]):
                raise ValueError(f"Edit distance mismatch for {track}")
            record = {"correct": correct, "matches": matches, "edits": edits,
                      "gt": expected_gt}
            by_kind[src["source_legibility"]].append(record)
            by_kind["All"].append(record)

    if len(seen) != 3000 or seen != set(source_by_track):
        raise ValueError(f"Missing predictions for {3000 - len(seen)} tracks")
    observed = {kind: stats(by_kind[kind]) for kind in ("All", "Poor", "Illegible")}
    if observed["Poor"]["samples"] != 2158 or observed["Illegible"]["samples"] != 842:
        raise ValueError("Legibility sample counts inconsistent with frozen selection")

    global_metrics = metadata["metrics"]
    totals = observed["All"]
    for name in ("sequence_correct", "sequence_total", "character_correct",
                 "character_total", "edit_errors", "ground_truth_characters",
                 "partial_5_correct", "partial_6_correct"):
        from_report = global_metrics[name]
        local_name = "samples" if name == "sequence_total" else name
        if totals[local_name] != from_report:
            raise ValueError(f"Overall {name} differs from supplied metrics JSON")
    return {
        "frozen_manifest_sha256": found_hash,
        "input_metric_model": metadata["cls_loss"],
        "checkpoint_files": [Path(x).name for x in metadata["checkpoint_files"]],
        "frames": metadata["frames"],
        "cohorts": observed,
        "interpretation": (
            "Exploratory post-hoc LPLCv2 aspect-controlled subset, as-annotated. "
            "No per-image predictions or OCR ground truths in this summary."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--output-json", type=Path,
                   help="Optional LOCAL aggregate-only JSON summary")
    args = p.parse_args()
    summary = summarize(args.dataset, args.predictions, args.metrics,
                        args.expected_manifest_sha256)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
        print(f"Aggregate-only result saved to {args.output_json}")


if __name__ == "__main__":
    main()
