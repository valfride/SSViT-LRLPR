#!/usr/bin/env python3
"""Build all OCR-labelled Illegible (leg=0) LPLCv2 plate crops as an F=1 LMDB.

All annotations with an EXACT [A-Z0-9]{7} OCR string are included, even if
labels or source photographs repeat. Does not use model outputs or randomness
to choose samples. The source photographs and annotations are never modified.

This is a diagnostic, annotation-availability subset; Illegible OCR labels
may be unverifiable from the image itself, so accuracy is not automatically
a trustworthy recognition benchmark. Keep outputs local under dataset terms.
"""
import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import lmdb
import numpy as np

ALPHANUMERIC_7 = re.compile(r"^[A-Z0-9]{7}$")
LEG_ILLEGIBLE = 0
MOTORCYCLE_TYPES = frozenset({
    "MOTOCICLETA", "MOTONETA", "MOTO", "CICLOMOTOR",
    "SCOOTER", "TRICICLO", "QUADRICICLO",
})
MANIFEST_FIELDS = (
    "track_id", "source_image", "annotation_index", "ocr", "legibility",
    "vehicle_type", "known_motorcycle", "occluded",
    "camera_id", "time", "day", "rain", "faulty",
    "xy", "raw_annotation_box_xyxy", "crop_box_xyxy",
    "bbox_clipped_to_image", "annotation_area_inside_image",
    "source_width", "source_height",
    "crop_width", "crop_height", "crop_path",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vehicle_type(ann):
    car = ann.get("car")
    raw = car.get("type") if isinstance(car, dict) else None
    if not isinstance(raw, str):
        return ""
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", raw.upper()).split())


def bounding_box(xy):
    if not isinstance(xy, (tuple, list)) or len(xy) != 8:
        return None
    try:
        coords = list(map(float, xy))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in coords):
        return None
    x1, y1 = math.floor(min(coords[::2])), math.floor(min(coords[1::2]))
    x2, y2 = math.ceil(max(coords[::2])), math.ceil(max(coords[1::2]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def inventory(annotations, exclude_known_motorcycles):
    stats = Counter()
    selected = []
    for filename, info in sorted(annotations.items()):
        if not isinstance(info, dict):
            continue
        for idx, ann in enumerate(info.get("anns", [])):
            if not isinstance(ann, dict) or ann.get("leg") != LEG_ILLEGIBLE:
                continue
            stats["illegible_total"] += 1
            label = ann.get("ocr")
            if isinstance(label, str) and label.strip():
                stats["nonempty_ocr"] += 1
            if not isinstance(label, str) or ALPHANUMERIC_7.fullmatch(label) is None:
                stats["non_7char_ocr"] += 1
                continue
            stats["valid_7char_ocr"] += 1
            typ = vehicle_type(ann)
            is_moto = typ in MOTORCYCLE_TYPES
            if is_moto:
                stats["known_motorcycles_among_valid_ocr"] += 1
            if not typ:
                stats["missing_vehicle_type_among_valid_ocr"] += 1
            if bool(ann.get("occ", False)):
                stats["occluded_among_valid_ocr"] += 1
            if exclude_known_motorcycles and is_moto:
                stats["excluded_known_motorcycles"] += 1
                continue
            box = bounding_box(ann.get("xy"))
            if box is None:
                stats["invalid_geometry_among_valid_ocr"] += 1
            selected.append({
                "source_image": filename,
                "annotation_index": idx,
                "ocr": label, "xy": ann.get("xy"), "bbox": box,
                "vehicle_type": typ, "known_motorcycle": is_moto,
                "occluded": bool(ann.get("occ", False)),
                "cam": info.get("cam"), "time": info.get("time"),
                "day": info.get("day"), "rain": info.get("rain"),
                "faulty": info.get("faulty"),
            })
    stats["selected_annotation_candidates"] = len(selected)
    stats["unique_candidate_labels"] = len({x["ocr"] for x in selected})
    stats["unique_candidate_source_images"] = len(
        {x["source_image"] for x in selected})
    stats["repeated_labels"] = len(selected) - stats["unique_candidate_labels"]
    stats["repeated_source_images"] = len(selected) - stats["unique_candidate_source_images"]
    return selected, stats


def preflight_geometry(candidates, photos):
    """Check every annotation against *decoded* source-image dimensions.

    No crops are written. Distinguishes minor overflow from annotations
    completely outside the image; provides details for independent QA.
    """
    stats = Counter()
    problems = []
    for candidate in candidates:
        name = candidate["source_image"]
        item = {"source_image": name,
                "annotation_index": candidate["annotation_index"],
                "ocr": candidate["ocr"],
                "raw_bbox": candidate["bbox"]}
        if candidate["bbox"] is None:
            item["problem"] = "invalid_or_degenerate_box"
            problems.append(item)
            stats["invalid_or_degenerate_box"] += 1
            continue
        img = cv2.imread(str(photos / name), cv2.IMREAD_COLOR)
        if img is None:
            item["problem"] = "missing_or_undecodable_source_image"
            problems.append(item)
            stats["missing_or_undecodable_source_image"] += 1
            continue
        height, width = img.shape[:2]
        x1, y1, x2, y2 = candidate["bbox"]
        # Clamp every coordinate; negative maxima and positive out-of-image
        # minima must be handled explicitly to detect zero intersection.
        cx1, cy1 = max(0, min(width, x1)), max(0, min(height, y1))
        cx2, cy2 = max(0, min(width, x2)), max(0, min(height, y2))
        area = (x2 - x1) * (y2 - y1)
        inside = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        fraction = inside / area
        candidate["source_dimensions"] = [width, height]
        candidate["clipped_bbox"] = [cx1, cy1, cx2, cy2]
        candidate["inside_fraction"] = fraction
        if inside == 0:
            problem = "no_in_frame_intersection"
        elif [cx1, cy1, cx2, cy2] != candidate["bbox"]:
            problem = "partially_outside_image"
        else:
            stats["fully_in_bounds"] += 1
            continue
        stats[problem] += 1
        problems.append({
            **item,
            "problem": problem, "image_width": width, "image_height": height,
            "clipped_bbox": [cx1, cy1, cx2, cy2],
            "inside_fraction": round(fraction, 6),
            "area_lost_percent": round(100 * (1 - fraction), 3),
        })
    return {
        "candidate_count": len(candidates),
        "counts": dict(stats),
        "problem_count": len(problems),
        "problems": problems,
        "note": ("An out-of-frame annotation may target an intentionally "
                 "partially visible plate or be incorrect. Clipping recovers "
                 "ONLY the visible intersection, not missing characters."),
    }


def make_sheet(entries, output, num, seed):
    if num <= 0:
        return
    chosen = random.Random(seed).sample(entries, min(num, len(entries)))
    cell_w, cell_h, columns = 210, 115, 5
    rows = math.ceil(len(chosen) / columns)
    sheet = np.full((rows * cell_h, columns * cell_w, 3), 245, dtype=np.uint8)
    for i, row in enumerate(chosen):
        img = cv2.imread(str(output / row["crop_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Contact-sheet crop cannot be decoded")
        ih, iw = img.shape[:2]
        scale = min(194 / iw, 78 / ih)
        width, height = max(1, round(iw * scale)), max(1, round(ih * scale))
        interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
        img = cv2.resize(img, (width, height), interpolation=interp)
        x, y = i % columns, i // columns
        left = x * cell_w + (cell_w - width) // 2
        top = y * cell_h + (78 - height) // 2
        sheet[top:top + height, left:left + width] = img
        note = f"{row['track_id']} {row['ocr']} {iw}x{ih}"
        cv2.putText(sheet, note, (x * cell_w + 5, y * cell_h + 99),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (35, 35, 35),
                    1, cv2.LINE_AA)
    if not cv2.imwrite(str(output / "contact_sheet.jpg"), sheet):
        raise RuntimeError("Cannot write contact sheet")


def validate(output, expected=None):
    manifest = json.loads((output / "selection_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["samples"]
    with (output / "metadata.pkl").open("rb") as inp:
        metadata = pickle.load(inp)  # Locally generated file only.
    if len(rows) != len(metadata) or not rows:
        raise RuntimeError("Manifest count disagrees with LMDB metadata count")
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"Expected {expected} items; got {len(rows)}")
    if manifest["legibility_id"] != LEG_ILLEGIBLE or any(
            r["legibility"] != LEG_ILLEGIBLE for r in rows):
        raise RuntimeError("A non-Illegible plate was included")
    if any(ALPHANUMERIC_7.fullmatch(r["ocr"]) is None for r in rows):
        raise RuntimeError("Manifest contains an unsupported OCR label")
    id_pairs = [(r["source_image"], r["annotation_index"]) for r in rows]
    if len(set(id_pairs)) != len(rows):
        raise RuntimeError("Duplicate annotation identity in manifest")
    env = lmdb.open(str(output), readonly=True, lock=False, readahead=False,
                    max_readers=4)
    try:
        with env.begin() as txn:
            for row, meta in zip(rows, metadata):
                if meta["gt"] != row["ocr"] or meta["split"] != "test":
                    raise RuntimeError("LMDB ground truth or split mismatch")
                key = meta["lmdb_key"]
                if not isinstance(key, bytes):
                    raise RuntimeError("LMDB key is not encoded as bytes")
                expected_suffix = "/" + row["track_id"] + "/lr-001.png"
                if not meta["original_path"].replace("\\", "/").endswith(expected_suffix):
                    raise RuntimeError("LMDB sequence path mismatch")
                raw = txn.get(key)
                if raw is None:
                    raise RuntimeError("Missing LMDB image bytes")
                decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[:2] != (
                        row["crop_height"], row["crop_width"]):
                    raise RuntimeError("LMDB decode/crop-size mismatch")
                saved = output / row["crop_path"]
                if not saved.is_file() or saved.read_bytes() != raw:
                    raise RuntimeError("Saved crop differs from LMDB image")
    finally:
        env.close()
    return {
        "verified_samples": len(rows),
        "unique_annotation_identities": len(set(id_pairs)),
        "unique_ocr_labels": len({r["ocr"] for r in rows}),
        "unique_source_images": len({r["source_image"] for r in rows}),
        "known_motorcycles": sum(r["known_motorcycle"] for r in rows),
        "missing_vehicle_type": sum(not r["vehicle_type"] for r in rows),
        "occluded": sum(r["occluded"] for r in rows),
        "bbox_clipped_to_image": sum(r.get("bbox_clipped_to_image", False) for r in rows),
        "all_legibility": "Illegible (leg=0)", "split": "test", "frames": 1,
    }


def build(args):
    root = args.dataset_root.expanduser().resolve()
    json_path = root / "annotations_v2.json"
    photos = root / "images"
    if not json_path.is_file() or not photos.is_dir():
        raise FileNotFoundError("Dataset root must contain annotations_v2.json and images/")
    annotations = json.loads(json_path.read_text(encoding="utf-8"))
    candidates, stats = inventory(annotations, args.exclude_known_motorcycles)
    print("Illegible annotation inventory:\n" + json.dumps(stats, indent=2))
    geometric = preflight_geometry(candidates, photos)
    print("Image/annotation geometry preflight:")
    print(json.dumps({**geometric, "problems": geometric["problems"][:30]},
                     indent=2))
    if len(geometric["problems"]) > 30:
        print("Showing first 30 of", len(geometric["problems"]), "issues")
    if args.audit_report:
        report_path = args.audit_report.expanduser().resolve()
        if report_path.exists():
            raise FileExistsError("Refusing to overwrite geometry audit: " + str(report_path))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(geometric, indent=2) + "\n", encoding="utf-8")
        print("Full geometry report:", report_path)
    if args.audit_only:
        return
    if args.output is None:
        raise ValueError("--output is required unless --audit-only")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pre-existing output: {output}")
    if not args.exclude_known_motorcycles and stats["valid_7char_ocr"] != args.expect_all:
        raise RuntimeError(
            f"Local annotation version has {stats['valid_7char_ocr']} seven-character "
            f"Illegible labels, expected {args.expect_all}. Re-audit before proceeding.")
    # Preflight every instance BEFORE staging an LMDB. No candidate is silently
    # discarded. An explicitly requested clipping policy is permitted only
    # for intersecting boxes above the chosen retained-area threshold.
    blockers = [x for x in geometric["problems"]
                if x["problem"] != "partially_outside_image"]
    overflowing = [x for x in geometric["problems"]
                   if x["problem"] == "partially_outside_image"]
    severely_outside = [x for x in overflowing
                        if x["inside_fraction"] < args.min_inside_fraction]
    if blockers or (overflowing and not args.clip_out_of_bounds) or (
            args.clip_out_of_bounds and severely_outside):
        raise RuntimeError(
            f"Geometry preflight failed: {len(blockers)} missing/invalid/"
            f"nonintersecting annotations; {len(overflowing)} partially outside; "
            f"{len(severely_outside)} below min inside fraction "
            f"{args.min_inside_fraction:.3f}. "
            "Inspect report; use --clip-out-of-bounds ONLY after manual "
            "review of cases and never claim those crops are fully intact.")
    output.parent.mkdir(parents=True, exist_ok=True)
    items = []
    metadata = []
    with tempfile.TemporaryDirectory(prefix=output.name + ".staging-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "crops").mkdir()
        env = lmdb.open(str(stage), map_size=2 * (1024 ** 3), subdir=True, sync=True)
        try:
            # Commit in batches; deterministic order is source filename + annotation index.
            with env.begin(write=True) as txn:
                for n, cand in enumerate(candidates):
                    img = cv2.imread(str(photos / cand["source_image"]), cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError("Cannot decode original image: " + cand["source_image"])
                    height, width = img.shape[:2]
                    if [width, height] != cand["source_dimensions"]:
                        raise RuntimeError("Image geometry changed after preflight")
                    originally_outside = cand["bbox"] != cand["clipped_bbox"]
                    x1, y1, x2, y2 = (cand["clipped_bbox"] if args.clip_out_of_bounds
                                      else cand["bbox"])
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise RuntimeError("Bounding box failed the geometry preflight")
                    crop = img[y1:y2, x1:x2]
                    ok, encoded = cv2.imencode(".png", crop)
                    if not ok:
                        raise RuntimeError("Lossless PNG encode failed")
                    bytes_data = encoded.tobytes()
                    track_id = f"track_{n:06d}"
                    crop_path = f"crops/{track_id}/lr-001.png"
                    dest = stage / crop_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(bytes_data)
                    key = f"image-{n:08d}".encode("ascii")
                    if not txn.put(key, bytes_data, overwrite=False):
                        raise RuntimeError("Duplicate LMDB key")
                    metadata.append({
                        "lmdb_key": key, "gt": cand["ocr"], "split": "test",
                        "original_path": str(output / crop_path),
                    })
                    items.append({
                        "track_id": track_id,
                        "source_image": cand["source_image"],
                        "annotation_index": cand["annotation_index"],
                        "ocr": cand["ocr"],
                        "legibility": LEG_ILLEGIBLE,
                        "vehicle_type": cand["vehicle_type"],
                        "known_motorcycle": cand["known_motorcycle"],
                        "occluded": cand["occluded"],
                        "camera_id": cand["cam"], "time": cand["time"],
                        "day": cand["day"], "rain": cand["rain"],
                        "faulty": cand["faulty"], "xy": cand["xy"],
                        "raw_annotation_box_xyxy": cand["bbox"],
                        "crop_box_xyxy": [x1, y1, x2, y2],
                        "bbox_clipped_to_image": originally_outside,
                        "annotation_area_inside_image": round(cand["inside_fraction"], 6),
                        "source_width": width, "source_height": height,
                        "crop_width": crop.shape[1],
                        "crop_height": crop.shape[0],
                        "crop_path": crop_path,
                    })
                    if (n + 1) % 200 == 0:
                        print(f"Prepared {n + 1}/{len(candidates)} Illegible crops", flush=True)
        finally:
            env.sync()
            env.close()

        with (stage / "metadata.pkl").open("wb") as f:
            pickle.dump(metadata, f, protocol=4)
        frozen = {
            "dataset": "LPLCv2", "subset": "all OCR-labelled Illegible annotations",
            "legibility": "Illegible", "legibility_id": LEG_ILLEGIBLE,
            "annotations": str(json_path), "annotations_sha256": digest(json_path),
            "selection": "all exact [A-Z0-9]{7} OCR annotations; deterministic file/index order",
            "duplicate_ocr_and_source_image_allowed": True,
            "excluded_known_motorcycles": bool(args.exclude_known_motorcycles),
            "includes_occluded": True,
            "geometry_policy": ("explicit clip partially out-of-bounds" if args.clip_out_of_bounds
                                else "strict no clipping"),
            "min_annotation_area_inside_image": args.min_inside_fraction,
            "geometry_preflight_counts": geometric["counts"],
            "crop": "original visible bbox pixels; lossless PNG; evaluator resizes to 96x32",
            "samples": items,
        }
        manifest_path = stage / "selection_manifest.json"
        manifest_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        with (stage / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for record in items:
                writer.writerow({k: record.get(k, "") for k in MANIFEST_FIELDS})
        make_sheet(items, stage, args.contact_sheet, args.seed)
        summary = {
            "inventory": dict(stats),
            "selected": len(items),
            "unique_ocr_labels": len({x["ocr"] for x in items}),
            "unique_source_images": len({x["source_image"] for x in items}),
            "known_motorcycles_included": sum(x["known_motorcycle"] for x in items),
            "unknown_vehicle_types_included": sum(not x["vehicle_type"] for x in items),
            "occluded_included": sum(x["occluded"] for x in items),
            "partially_outside_and_clipped": sum(x["bbox_clipped_to_image"] for x in items),
            "geometry_preflight": geometric["counts"],
            "vehicle_types": dict(Counter(x["vehicle_type"] or "MISSING" for x in items)),
            "native_crop_width": [min(x["crop_width"] for x in items),
                                  max(x["crop_width"] for x in items)],
            "native_crop_height": [min(x["crop_height"] for x in items),
                                   max(x["crop_height"] for x in items)],
            "selection_manifest_sha256": digest(manifest_path),
        }
        (stage / "dataset_info.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        result = validate(stage, expected=len(candidates))
        (stage / "validation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print("SUCCESS: all compatible LPLCv2 Illegible annotations prepared")
    print(f"LMDB: {output}")
    print(f"Contact sheet: {output / 'contact_sheet.jpg'}")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path,
                   help="LPLC_dataset with annotations_v2.json and images/")
    p.add_argument("--output", type=Path,
                   help="New LMDB directory; must not already exist")
    p.add_argument("--audit-only", action="store_true",
                   help="Count all Illegible OCR annotations and check all image box bounds")
    p.add_argument("--audit-report", type=Path,
                   help="Save complete geometry issue report as JSON (new file only)")
    p.add_argument("--clip-out-of-bounds", action="store_true",
                   help="Explicitly clip intersecting out-of-frame boxes to image bounds; "
                        "preserve raw coordinates and mark clipped items")
    p.add_argument("--min-inside-fraction", type=float, default=0.95,
                   help="When clipping, require at least this proportion of original "
                        "annotation bbox area within image (default: 0.95)")
    p.add_argument("--verify-only", action="store_true",
                   help="Revalidate the existing output without modifying it")
    p.add_argument("--expect-all", type=int, default=1189,
                   help="Expected total of seven-character Illegible annotations")
    p.add_argument("--exclude-known-motorcycles", action="store_true",
                   help="Optional DISTINCT sensitivity version, not the all-available set")
    p.add_argument("--contact-sheet", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026,
                   help="Only controls randomized QA contact-sheet examples")
    args = p.parse_args()
    if args.contact_sheet < 0:
        p.error("--contact-sheet must be >= 0")
    if not 0 < args.min_inside_fraction <= 1:
        p.error("--min-inside-fraction must be in (0, 1]")
    if args.verify_only:
        if args.output is None:
            p.error("--verify-only requires --output")
        result = validate(args.output.expanduser().resolve())
        print(json.dumps(result, indent=2))
    else:
        if args.dataset_root is None:
            p.error("--dataset-root required unless --verify-only")
        build(args)


if __name__ == "__main__":
    main()
