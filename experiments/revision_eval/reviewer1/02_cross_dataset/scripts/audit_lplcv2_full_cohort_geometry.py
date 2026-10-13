#!/usr/bin/env python3
"""Read-only, prediction-blind inventory of ALL eligible LPLCv2 Poor/Illegible crops.

Uses ORIGINAL annotations_v2.json + images/, not the historically capped Poor-3k
or mixed-3k LMDBs. Does not write crops, labels, predictions or manifests.
Reuses the same floor/ceil bounding-rectangle convention as the existing
LPLCv2 dataset builders. Counts annotation instances, not unique plate strings.
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

FORMATS = (
    re.compile(r"^[A-Z]{3}[0-9]{4}$"),
    re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"),
)
COHORT = {0: "Illegible", 1: "Poor"}
HEIGHT_BANDS = (4, 8, 10, 12, 14, 16, 20, 24, 32)
WIDTH_BANDS = (8, 16, 24, 32, 48, 64, 96)
PERCENTILES = (10, 25, 50, 75, 90)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox(xy):
    if not isinstance(xy, (list, tuple)) or len(xy) != 8:
        return None
    try:
        coords = [float(n) for n in xy]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(n) for n in coords):
        return None
    x1, x2 = math.floor(min(coords[0::2])), math.ceil(max(coords[0::2]))
    y1, y2 = math.floor(min(coords[1::2])), math.ceil(max(coords[1::2]))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def percentile(data, p):
    if not data:
        return None
    seq = sorted(data)
    k = (len(seq) - 1) * p / 100.0
    a, b = math.floor(k), math.ceil(k)
    return round(seq[a] + (seq[b] - seq[a]) * (k - a), 4)


def describe(rows):
    if not rows:
        return {"samples": 0}
    widths = [r["width"] for r in rows]
    heights = [r["height"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    border_margins = [r["border_margin"] for r in rows]
    sources = Counter(r["source"] for r in rows)
    labels = Counter(r["label"] for r in rows)
    return {
        "samples": len(rows),
        "distinct_source_images": len(sources),
        "distinct_ocr_labels": len(labels),
        "extra_annotations_sharing_source_image": len(rows) - len(sources),
        "extra_annotations_sharing_ocr_label": len(rows) - len(labels),
        "native_width_min_p10_p25_median_p75_p90_max": [
            min(widths), *[percentile(widths, p) for p in PERCENTILES], max(widths)
        ],
        "native_height_min_p10_p25_median_p75_p90_max": [
            min(heights), *[percentile(heights, p) for p in PERCENTILES], max(heights)
        ],
        "native_aspect_min_p10_p25_median_p75_p90_max": [
            round(min(ratios), 4),
            *[percentile(ratios, p) for p in PERCENTILES],
            round(max(ratios), 4),
        ],
        "occluded_count": sum(r["occluded"] for r in rows),
        "type_missing_count": sum(r["missing_type"] for r in rows),
        # Diagnostic only: the ROI margin setting on the curator's
        # detector ROI is not known to be enforced, and the LPLCv2
        # box is a separate annotation type. Never silently discard.
        "native_annotation_border_margin_px_min_p10_median_p90": [
            min(border_margins), percentile(border_margins, 10),
            percentile(border_margins, 50), percentile(border_margins, 90),
        ],
        "annotations_at_source_image_border": sum(v == 0 for v in border_margins),
        "annotations_below_2px_source_image_margin": sum(v < 2 for v in border_margins),
        "annotations_at_least_2px_source_image_margin": sum(v >= 2 for v in border_margins),
        "height_at_least_px": {
            str(value): sum(r["height"] >= value for r in rows)
            for value in HEIGHT_BANDS
        },
        "width_at_least_px": {
            str(value): sum(r["width"] >= value for r in rows)
            for value in WIDTH_BANDS
        },
        "joint_size_thresholds_after_aspect": {
            f"w>={w}_h>={h}": sum(r["width"] >= w and r["height"] >= h for r in rows)
            for w, h in ((8, 4), (24, 8), (30, 10), (36, 12), (42, 14), (48, 16))
        },
    }


def inventory(args):
    if not (0 < args.min_aspect <= args.max_aspect):
        raise ValueError("Aspect bounds must be positive and ordered")
    if args.min_width < 1 or args.min_height < 1:
        raise ValueError("Minimum size must be positive")
    root = args.dataset_root.expanduser().resolve(strict=True)
    image_dir = (root / "images").resolve(strict=True)
    ann_path = root / "annotations_v2.json"
    source_hash = fingerprint(ann_path)
    annotations = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict):
        raise TypeError("Expected annotation map keyed by source image")

    counts = {name: Counter() for name in COHORT.values()}
    valid_geometry = defaultdict(list)
    selected = defaultdict(list)
    for filename, info in sorted(annotations.items()):
        if not isinstance(info, dict):
            continue
        anns = info.get("anns", [])
        if not isinstance(anns, list):
            continue
        candidates = [(a, COHORT[a["leg"]]) for a in anns
                      if isinstance(a, dict) and type(a.get("leg")) is int
                      and a["leg"] in COHORT]
        if not candidates:
            continue

        # Reject path traversal, and check source only after candidate OCR/box screening.
        image_path = (image_dir / filename).resolve()
        if not image_path.is_relative_to(image_dir):
            raise ValueError("Annotation source image resolved outside images/")
        img = None
        decoded = False
        for ann, kind in candidates:
            ctr = counts[kind]
            ctr["all_legibility_annotations"] += 1
            label = ann.get("ocr")
            if not isinstance(label, str) or not any(p.fullmatch(label) for p in FORMATS):
                ctr["excluded_invalid_brazilian_ocr_format"] += 1
                continue
            ctr["valid_ocr_format"] += 1
            box = bbox(ann.get("xy"))
            if box is None:
                ctr["excluded_invalid_bbox"] += 1
                continue
            if not decoded:
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.is_file() else None
                decoded = True
            if img is None:
                ctr["excluded_missing_or_undecodable_image"] += 1
                continue
            ih, iw = img.shape[:2]
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= iw and 0 <= y1 < y2 <= ih):
                ctr["excluded_out_of_bounds_bbox"] += 1
                continue
            w, h = x2 - x1, y2 - y1
            row = {
                "width": w, "height": h, "ratio": w / h,
                "border_margin": min(x1, y1, iw - x2, ih - y2),
                "source": filename, "label": label,
                "occluded": bool(ann.get("occ", False)),
                "missing_type": not (
                    isinstance(ann.get("car"), dict) and ann["car"].get("type")
                ),
            }
            valid_geometry[kind].append(row)
            ctr["valid_ocr_and_in_bounds_crop"] += 1
            if not args.min_aspect <= row["ratio"] <= args.max_aspect:
                ctr["excluded_native_aspect_outside_bounds"] += 1
                continue
            ctr["within_native_aspect_bounds"] += 1
            if w < args.min_width or h < args.min_height:
                ctr["excluded_native_size_below_minimum"] += 1
                continue
            ctr["selected_by_geometry"] += 1
            selected[kind].append(row)

    # Never print plate strings, filenames or per-annotation identities.
    output = {
        "scope": "original annotations_v2.json, all Poor and Illegible entries",
        "source_annotations_sha256": source_hash,
        "policy": {
            "native_aspect_ratio_inclusive": [args.min_aspect, args.max_aspect],
            "minimum_native_crop_width_px": args.min_width,
            "minimum_native_crop_height_px": args.min_height,
            "ocr": "old Brazilian or Mercosur seven-character structure",
            "bbox": "floor/ceil axis-aligned envelope of original four corners, fully in bounds",
            "border_2px_rule": "reported diagnostically, NOT an exclusion filter",
            "vehicle_types": "all, including motorcycles and missing vehicle types",
            "occlusion": "retained",
            "duplicates": "annotation instances retained; source/label repeats counted",
            "sampling": "none; include every qualifying original annotation",
            "model_predictions_used": False,
        },
        "classes": {},
    }
    for kind in ("Poor", "Illegible"):
        output["classes"][kind] = {
            "filter_counts": dict(sorted(counts[kind].items())),
            "all_valid_geometry_before_ratio_filter": describe(valid_geometry[kind]),
            "retained_after_ratio_and_size_filters": describe(selected[kind]),
        }
    output["total_selected"] = sum(len(x) for x in selected.values())
    if not output["total_selected"]:
        raise RuntimeError("No eligible annotations survive filters; investigate thresholds")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--min-aspect", type=float, default=2.5)
    p.add_argument("--max-aspect", type=float, default=3.5)
    p.add_argument("--min-width", type=int, default=8)
    p.add_argument("--min-height", type=int, default=4)
    p.add_argument("--output-json", type=Path,
                   help="Optional local-only aggregate JSON; refuses overwrite")
    args = p.parse_args()
    report = inventory(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output_json:
        destination = args.output_json.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(rendered)


if __name__ == "__main__":
    main()
