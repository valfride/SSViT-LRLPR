#!/usr/bin/env python3
"""Prediction-blind, read-only visual QA of a frozen LPLCv2 evaluation subset.

Generate pages for ALL samples showing the source image + selected box,
a context zoom, and the original-pixel annotated crop. Separately render
priority-flagged crops (image-edge proximity, low texture, unusual shape).
Flags indicate items for *manual review*, NOT invalid annotation labels.

Produces qa_samples.csv with manual_status/manual_note columns for review;
does not change annotations, crops, metadata, sample selection, or predictions.
Keep output local: LPLCv2 has request-based redistribution restrictions.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "track_id", "source_image", "annotation_index", "ocr", "camera_id",
    "crop_width", "crop_height", "source_width", "source_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "near_source_boundary", "low_gray_variation", "unusual_aspect_ratio",
    "flags", "source_read_status", "manual_status", "manual_note",
]
COLS = 2
TILE_W, TILE_H = 760, 240
PAGE_ROWS = 12
CANVAS_W, CANVAS_H = COLS * TILE_W, PAGE_ROWS * TILE_H
PANEL_W, PANEL_H = 235, 149


def open_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit(img, max_w, max_h, enlarge=True):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if not enlarge:
        scale = min(1., scale)
    width = max(1, round(w * scale))
    height = max(1, round(h * scale))
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    return cv2.resize(img, (width, height), interpolation=method)


def context_image(img, box):
    height, width = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Context centered on annotated box; offset is clipped to image.
    margin = max(25, 2 * max(bw, bh))
    left = max(0, x1 - margin)
    right = min(width, x2 + margin)
    top = max(0, y1 - margin)
    bottom = min(height, y2 + margin)
    region = img[top:bottom, left:right].copy()
    cv2.rectangle(region, (x1 - left, y1 - top),
                  (x2 - left - 1, y2 - top - 1), (0, 0, 255), 2)
    return region


def draw_panel(canvas, img, left, top, name):
    bg = (238, 238, 238)
    cv2.rectangle(canvas, (left, top), (left + PANEL_W, top + PANEL_H), bg, -1)
    if img is not None and img.size:
        tile = fit(img, PANEL_W - 6, PANEL_H - 6, enlarge=True)
        ox = left + (PANEL_W - tile.shape[1]) // 2
        oy = top + (PANEL_H - tile.shape[0]) // 2
        canvas[oy:oy + tile.shape[0], ox:ox + tile.shape[1]] = tile
    cv2.putText(canvas, name, (left + 2, top + PANEL_H + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)


def sample_features(row, full, low_std):
    x1, y1, x2, y2 = map(int, row["crop_box_xyxy"])
    ih, iw = full.shape[:2]
    if not (0 <= x1 < x2 <= iw and 0 <= y1 < y2 <= ih):
        raise ValueError(f"Invalid/inconsistent bbox: {row['track_id']}: {row['crop_box_xyxy']}")
    crop = full[y1:y2, x1:x2]
    gray_std = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())
    w, h = x2 - x1, y2 - y1
    aspect = w / h
    # Bounds expressed relative to source frame, *not* model input size.
    edge = min(x1 / iw, y1 / ih, (iw - x2) / iw, (ih - y2) / ih) < 0.035
    uniform = gray_std < low_std
    aspect_issue = aspect < 1.7 or aspect > 4.0
    flags = []
    if edge:
        flags.append("NEAR_FRAME_EDGE")
    if uniform:
        flags.append("LOW_GRAY_STD")
    if aspect_issue:
        flags.append("UNUSUAL_ASPECT_RATIO")
    result = {
        "near_source_boundary": edge,
        "low_gray_variation": uniform,
        "unusual_aspect_ratio": aspect_issue,
        "flags": ",".join(flags),
    }
    return crop, result


def page_path(out, section, page_number):
    folder = out / section
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"page_{page_number:03d}.jpg"


def render_pages(out, section, records, images, page_size):
    result_paths = []
    if not records:
        return result_paths
    for base in range(0, len(records), page_size):
        batch = records[base:base + page_size]
        rows_used = math.ceil(len(batch) / COLS)
        canvas = np.full((rows_used * TILE_H, CANVAS_W, 3), 255, dtype=np.uint8)
        for index, record in enumerate(batch):
            local_x = (index % COLS) * TILE_W
            local_y = (index // COLS) * TILE_H
            header = f"{record['track_id']} GT={record['ocr']}  {record['crop_width']}x{record['crop_height']} px"
            if record["flags"]:
                header += "  ! " + record["flags"]
            cv2.putText(canvas, header[:94], (local_x + 5, local_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, record["source_image"][:26], (local_x + 5, local_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
            if record["source_read_status"] != "ok":
                cv2.putText(canvas, record["source_read_status"], (local_x + 10, local_y + 104),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 210), 1, cv2.LINE_AA)
                continue
            full = cv2.imread(str(images / record["source_image"]), cv2.IMREAD_COLOR)
            if full is None:
                raise RuntimeError("Decode changed during sheet rendering: " + record["source_image"])
            box = (record["bbox_x1"], record["bbox_y1"],
                   record["bbox_x2"], record["bbox_y2"])
            overview = full.copy()
            thickness = max(2, round(max(full.shape[:2]) / 500))
            cv2.rectangle(overview, (box[0], box[1]),
                          (box[2] - 1, box[3] - 1), (0, 0, 255), thickness)
            context = context_image(full, box)
            crop = full[box[1]:box[3], box[0]:box[2]]
            draw_panel(canvas, overview, local_x + 4, local_y + 47, "SOURCE + BBOX")
            draw_panel(canvas, context, local_x + 248, local_y + 47, "BBOX CONTEXT")
            draw_panel(canvas, crop, local_x + 492, local_y + 47, "ANNOTATED CROP (enlarged)")
            cv2.putText(canvas, "Visual review only; flags do NOT imply invalid ground truth",
                        (local_x + 6, local_y + 228),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80, 80, 80), 1, cv2.LINE_AA)
        page = page_path(out, section, base // page_size + 1)
        if not cv2.imwrite(str(page), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError("Failed to write QA page: " + str(page))
        result_paths.append(str(page.relative_to(out)))
        if (base // page_size + 1) % 10 == 0:
            print(f"{section}: wrote {base // page_size + 1} pages", flush=True)
    return result_paths


def audit(args):
    root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not (root / "images").is_dir():
        raise FileNotFoundError("Dataset root must contain images/")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {out}")
    out.mkdir(parents=True)
    original = open_json(manifest_path)
    rows = original.get("samples")
    if not isinstance(rows, list) or len(rows) != 3000:
        raise ValueError("Expected frozen 3,000-sample selection manifest")
    image_root = root / "images"
    records = []
    seen_labels = set()
    seen_images = set()
    counts = Counter()
    for n, row in enumerate(rows):
        label, filename = row["ocr"], row["source_image"]
        if label in seen_labels or filename in seen_images:
            raise ValueError(f"Frozen selection not unique at index {n}")
        seen_labels.add(label)
        seen_images.add(filename)
        rec = {k: row.get(k, "") for k in FIELDS if k not in {
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "flags",
            "source_read_status", "near_source_boundary",
            "low_gray_variation", "unusual_aspect_ratio",
            "manual_status", "manual_note"}}
        rec["manual_status"] = ""
        rec["manual_note"] = ""
        box = row["crop_box_xyxy"]
        rec.update(dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box)))
        image_path = image_root / filename
        full = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if full is None:
            rec.update({"source_read_status": "missing_or_unreadable",
                        "flags": "UNREADABLE_SOURCE",
                        "near_source_boundary": "",
                        "low_gray_variation": "",
                        "unusual_aspect_ratio": ""})
            counts["unreadable_source"] += 1
        else:
            h, w = full.shape[:2]
            if h != row["source_height"] or w != row["source_width"]:
                raise ValueError(f"Source geometry changed: {filename} {w}x{h}")
            crop, features = sample_features(row, full, args.low_std)
            if crop.shape[1] != row["crop_width"] or crop.shape[0] != row["crop_height"]:
                raise ValueError(f"Crop bbox mismatch: {row['track_id']}")
            rec.update(features)
            rec["source_read_status"] = "ok"
            counts["ok_decoded"] += 1
            for name in rec["flags"].split(","):
                if name:
                    counts[name] += 1
            if not rec["flags"]:
                counts["not_flagged"] += 1
        records.append(rec)
        if (n + 1) % 500 == 0:
            print(f"Inspected {n+1}/{len(rows)} source images", flush=True)

    csv_path = out / "qa_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        wr = csv.DictWriter(stream, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(records)

    flagged = [r for r in records if r["flags"]]
    flagged.sort(key=lambda r: (
        not bool(r["near_source_boundary"] is True),
        not bool(r["low_gray_variation"] is True),
        r["track_id"],
    ))
    all_pages = render_pages(out, "all_pages", records, image_root, args.page_size)
    flagged_pages = render_pages(out, "priority_pages", flagged, image_root, args.page_size)
    result = {
        "dataset_root": str(root), "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "samples": len(records), "flags_are_review_priorities_not_exclusions": True,
        "criteria": {
            "near_source_boundary": "bbox within 3.5% of any source-image edge",
            "low_gray_variation": f"grayscale standard deviation < {args.low_std}",
            "unusual_aspect_ratio": "bbox width/height outside [1.7, 4.0]",
        },
        "counts": dict(counts), "unique_flagged_samples": len(flagged),
        "all_pages": len(all_pages), "priority_pages": len(flagged_pages),
        "csv": "qa_samples.csv",
        "note": ("Image inspection alone cannot prove OCR ground-truth text incorrect; "
                 "a difficult/partial plate is not automatically a crop error. "
                 "Review labels and annotation target independently of model predictions."),
    }
    (out / "qa_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Review flagged pages first; then examine all-pages to estimate false negatives.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--page-size", type=int, default=24,
                   help="Number of samples per contact sheet; must be even, at most 24")
    p.add_argument("--low-std", type=float, default=16.0,
                   help="Gray-value crop standard deviation below which to prioritize review")
    args = p.parse_args()
    if not (2 <= args.page_size <= 24 and args.page_size % 2 == 0):
        p.error("--page-size must be an even number between 2 and 24")
    if args.low_std <= 0:
        p.error("--low-std must be positive")
    audit(args)


if __name__ == "__main__":
    main()
