#!/usr/bin/env python3
"""Prepare a frozen, label-unique LPLCv2 Poor-only OCR test LMDB.

Read-only access to the original dataset. Generates 3,000 lossless plate crops,
a selection manifest, a contact sheet and an LMDB accepted by test.py F=1.
LPLCv2 images/annotations are request-only: never commit generated outputs.
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
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import lmdb
import numpy as np

LABEL_RE = re.compile(r"^[A-Z0-9]{7}$")
LEG_POOR = 1

# Accept only explicitly labeled non-motorcycle vehicle classes. Exclude
# unknown and missing classes rather than risk including unlabelled motorcycles.
ALLOWED_NON_MOTORCYCLE_TYPES = frozenset({
    "AUTOMOVEL", "CAMINHONETE", "CAMIONETA", "CAMINHAO",
    "CAMINHAO TRATOR", "ONIBUS", "MICROONIBUS", "UTILITARIO",
    "JIPE", "VAN", "FURGAO", "PICAPE", "SUV", "MISTO",
    "REBOQUE", "SEMI REBOQUE",
})


def normalize_vehicle_type(value):
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", ascii_text.upper()).split())


def vehicle_type_status(ann):
    car = ann.get("car")
    value = car.get("type") if isinstance(car, dict) else None
    normalized = normalize_vehicle_type(value)
    if not normalized:
        return "missing_vehicle_type", normalized
    if any(token in normalized for token in (
            "MOTO", "CICLOMOTOR", "SCOOTER", "TRICICLO", "QUADRICICLO")):
        return "motorcycle_excluded", normalized
    if normalized not in ALLOWED_NON_MOTORCYCLE_TYPES:
        return "unrecognized_vehicle_type", normalized
    return "allowed", normalized

FIELDS = (
    "track_id", "source_image", "annotation_index", "ocr", "legibility",
    "camera_id", "time", "day", "rain", "faulty", "vehicle_type", "xy",
    "crop_box_xyxy", "source_width", "source_height",
    "crop_width", "crop_height", "crop_path",
)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_box(coords):
    if not isinstance(coords, (list, tuple)) or len(coords) != 8:
        return None
    try:
        v = [float(x) for x in coords]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in v):
        return None
    xlo, xhi = math.floor(min(v[::2])), math.ceil(max(v[::2]))
    ylo, yhi = math.floor(min(v[1::2])), math.ceil(max(v[1::2]))
    if xhi <= xlo or yhi <= ylo:
        return None
    return xlo, ylo, xhi, yhi


def gather(annotations, image_dir, excluded_labels):
    stats = Counter()
    pool = []
    for source_name, info in sorted(annotations.items()):
        if not isinstance(info, dict):
            stats["invalid_record"] += 1
            continue
        for idx, ann in enumerate(info.get("anns", [])):
            if not isinstance(ann, dict) or ann.get("leg") != LEG_POOR:
                continue
            stats["poor_total"] += 1
            label = ann.get("ocr")
            if not isinstance(label, str) or LABEL_RE.fullmatch(label) is None:
                stats["invalid_ocr"] += 1
                continue
            stats["valid_ocr"] += 1
            if bool(ann.get("occ", False)):
                stats["occluded"] += 1
                continue
            status, vehicle_type = vehicle_type_status(ann)
            stats["type_" + (vehicle_type or "MISSING")] += 1
            if status != "allowed":
                stats[status] += 1
                continue
            stats["allowed_non_motorcycle_type"] += 1
            if label in excluded_labels:
                stats["overlap_excluded"] += 1
                continue
            box = resolve_box(ann.get("xy"))
            if box is None:
                stats["invalid_xy"] += 1
                continue
            source_path = image_dir / source_name
            if not source_path.is_file():
                stats["missing_source_image"] += 1
                continue
            pool.append({
                "source_image": source_name, "annotation_index": idx,
                "ocr": label, "box": box, "xy": ann["xy"],
                "vehicle_type": vehicle_type,
                "cam": info.get("cam"), "time": info.get("time"),
                "day": info.get("day"), "rain": info.get("rain"),
                "faulty": info.get("faulty"),
            })
    stats["eligible_before_image_decode"] = len(pool)
    stats["unique_labels_before_image_decode"] = len({p["ocr"] for p in pool})
    stats["unique_source_images_before_image_decode"] = len(
        {p["source_image"] for p in pool}
    )
    return pool, stats


def read_excluded_labels(paths):
    labels = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            path = path / "metadata.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Overlap metadata missing: {path}")
        # Read pickle ONLY from your own trusted LRLPR experiment environment.
        with path.open("rb") as handle:
            items = pickle.load(handle)
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"Unexpected metadata format at {path}")
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("gt"), str):
                labels.add(item["gt"].strip().upper())
        print(f"Overlap exclusion metadata: {path}, labels accumulated={len(labels)}")
    return labels


def produce_sheet(rows, folder, count, seed):
    if not rows or count <= 0:
        return
    chosen = random.Random(seed + 1).sample(rows, min(count, len(rows)))
    cols, cell_w, cell_h = 6, 185, 105
    nrows = math.ceil(len(chosen) / cols)
    sheet = np.full((nrows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for i, item in enumerate(chosen):
        img = cv2.imread(str(folder / item["crop_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Saved crop cannot be decoded: {item['crop_path']}")
        y, x = divmod(i, cols)
        scale = min((cell_w - 12) / img.shape[1], 70 / img.shape[0])
        nw = max(1, int(img.shape[1] * scale))
        nh = max(1, int(img.shape[0] * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        thumb = cv2.resize(img, (nw, nh), interpolation=interp)
        left = x * cell_w + (cell_w - nw) // 2
        top = y * cell_h + (75 - nh) // 2
        sheet[top:top + nh, left:left + nw] = thumb
        cv2.putText(sheet, f"{item['track_id']}  {item['ocr']}",
                    (x * cell_w + 6, y * cell_h + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    sheet_path = folder / "contact_sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError("Failed to save contact sheet")


def validate_prepared(outdir, expected):
    manifest = load_json(outdir / "selection_manifest.json")
    metadata_path = outdir / "metadata.pkl"
    # Pickle from this locally generated output only.
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    rows = manifest["samples"]
    if len(rows) != expected or len(metadata) != expected:
        raise RuntimeError("Selection or metadata count is not the requested count")
    if len({r["ocr"] for r in rows}) != expected:
        raise RuntimeError("Duplicate OCR label in selection")
    if len({r["source_image"] for r in rows}) != expected:
        raise RuntimeError("Repeated source image in selection")
    if not all(r["legibility"] == 1 for r in rows):
        raise RuntimeError("Non-Poor sample found")
    if not all(r.get("vehicle_type") in ALLOWED_NON_MOTORCYCLE_TYPES for r in rows):
        raise RuntimeError("An unapproved or motorcycle vehicle type was found")
    env = lmdb.open(str(outdir), readonly=True, lock=False, readahead=False,
                    max_readers=4)
    try:
        with env.begin() as txn:
            for row, meta in zip(rows, metadata):
                if meta["gt"] != row["ocr"] or meta["split"] != "test":
                    raise RuntimeError(f"Bad metadata: {row['track_id']}")
                if f"/{row['track_id']}/lr-001.png" not in meta["original_path"].replace("\\", "/"):
                    raise RuntimeError(f"Unexpected virtual path: {meta['original_path']}")
                key = meta["lmdb_key"]
                if not isinstance(key, bytes):
                    raise RuntimeError("LMDB key must be bytes")
                encoded = txn.get(key)
                if not encoded:
                    raise RuntimeError(f"Missing LMDB sample {key!r}")
                crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if crop is None or crop.shape[0] != row["crop_height"] or crop.shape[1] != row["crop_width"]:
                    raise RuntimeError(f"Invalid crop in LMDB {row['track_id']}")
                if not (outdir / row["crop_path"]).is_file():
                    raise RuntimeError(f"Crop file missing for {row['track_id']}")
    finally:
        env.close()
    return {"verified_samples": expected, "unique_labels": expected,
            "unique_source_images": expected, "all_legibility": "Poor (leg=1)",
            "lmdb_decoded": expected, "split": "test", "frames": 1}


def prepare(args):
    root = args.dataset_root.expanduser().resolve()
    image_dir = root / "images"
    ann_path = args.annotations.expanduser().resolve() if args.annotations else root / "annotations_v2.json"
    out = args.output.expanduser().resolve() if args.output else None
    if not ann_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError("Expected annotation JSON and images/ under --dataset-root")
    if not args.audit_only:
        if out is None:
            raise ValueError("--output is required unless --audit-only")
        if out.exists():
            raise FileExistsError(f"Frozen output already exists, refusing to overwrite: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)

    anns = load_json(ann_path)
    excluded_labels = read_excluded_labels(args.exclude_labels_metadata)
    candidates, stats = gather(anns, image_dir, excluded_labels)
    if args.audit_only:
        print(json.dumps({"candidate_audit": dict(stats),
                          "allowed_vehicle_types": sorted(ALLOWED_NON_MOTORCYCLE_TYPES)},
                         indent=2, ensure_ascii=False))
        return
    if stats["unique_labels_before_image_decode"] < args.count:
        raise RuntimeError(f"Only {stats['unique_labels_before_image_decode']} unique labels available")

    # Draw plate labels uniformly rather than favoring labels with more images.
    candidates.sort(key=lambda r: (r["ocr"], r["source_image"], r["annotation_index"]))
    by_label = defaultdict(list)
    for row in candidates:
        by_label[row["ocr"]].append(row)
    rng = random.Random(args.seed)
    labels = sorted(by_label)
    rng.shuffle(labels)
    for choices in by_label.values():
        rng.shuffle(choices)
    print("Candidate audit:", json.dumps(stats, indent=2))
    print(f"Sampling {args.count} distinct labels with seed={args.seed}; "
          "also requiring distinct source images")

    selected_labels, selected_sources = set(), set()
    final_rows = []
    skipped = Counter()
    # Write under a temporary directory. Rename only after all samples pass checks.
    with tempfile.TemporaryDirectory(prefix=out.name + ".staging-", dir=out.parent) as workdir:
        stage = Path(workdir)
        (stage / "crops").mkdir()
        metadata = []
        env = lmdb.open(str(stage), map_size=2 * 1024**3, subdir=True,
                        max_dbs=1, writemap=False, sync=True)
        try:
            with env.begin(write=True) as txn:
                for label in labels:
                    if len(final_rows) == args.count:
                        break
                    for row in by_label[label]:
                        if row["source_image"] in selected_sources:
                            skipped["duplicate_source_image"] += 1
                            continue
                        image_path = image_dir / row["source_image"]
                        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                        if img is None:
                            skipped["image_decode_failed"] += 1
                            continue
                        h, w = img.shape[:2]
                        x1, y1, x2, y2 = row["box"]
                        # Reject out-of-bounds boxes; do not clip away unknown plate content.
                        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                            skipped["out_of_bounds_bbox"] += 1
                            continue
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            skipped["empty_crop"] += 1
                            continue
                        ok, encoded = cv2.imencode(".png", crop)
                        if not ok:
                            skipped["crop_encode_failed"] += 1
                            continue

                        track_id = f"track_{len(final_rows):06d}"
                        crop_relpath = f"crops/{track_id}/lr-001.png"
                        crop_path = stage / crop_relpath
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        crop_path.write_bytes(encoded.tobytes())
                        key = f"image-{len(final_rows):08d}".encode("ascii")
                        txn.put(key, encoded.tobytes(), overwrite=False)
                        meta = {
                            "lmdb_key": key, "gt": row["ocr"], "split": "test",
                            # The path is used by the evaluator to GROUP images, not to load bytes.
                            "original_path": str(out / crop_relpath),
                        }
                        metadata.append(meta)
                        final_rows.append({
                            "track_id": track_id,
                            "source_image": row["source_image"],
                            "annotation_index": row["annotation_index"],
                            "ocr": row["ocr"],
                            "legibility": LEG_POOR,
                            "camera_id": row["cam"], "time": row["time"],
                            "day": row["day"], "vehicle_type": row["vehicle_type"],
                            "rain": row["rain"],
                            "faulty": row["faulty"], "xy": row["xy"],
                            "crop_box_xyxy": [x1, y1, x2, y2],
                            "source_width": w, "source_height": h,
                            "crop_width": crop.shape[1], "crop_height": crop.shape[0],
                            "crop_path": crop_relpath,
                        })
                        selected_labels.add(row["ocr"])
                        selected_sources.add(row["source_image"])
                        if len(final_rows) % 500 == 0:
                            print(f"Prepared {len(final_rows)}/{args.count} unique crops")
                        break  # Stop after the first viable image for this label.
        finally:
            env.sync()
            env.close()
        if len(final_rows) != args.count:
            raise RuntimeError(
                f"Could produce only {len(final_rows)} valid, unique plate crops; "
                f"needed {args.count}. Rejection counts: {dict(skipped)}"
            )

        with (stage / "metadata.pkl").open("wb") as handle:
            pickle.dump(metadata, handle, protocol=4)
        manifest = {
            "dataset": "LPLCv2", "source_annotations": str(ann_path),
            "source_annotations_sha256": digest(ann_path),
            "legibility": "Poor", "legibility_id": LEG_POOR,
            "vehicle_type_policy": "strict allowed non-motorcycle types",
            "allowed_vehicle_types": sorted(ALLOWED_NON_MOTORCYCLE_TYPES),
            "count": args.count, "seed": args.seed,
            "selection_method": "seeded permutation of unique OCR labels; shuffled "
                                "candidates within each label; first image not already used",
            "eligibility": "leg=1, exact [A-Z0-9]{7}, non-occluded, allowed "
                           "non-motorcycle car.type, valid in-bounds xy, source image "
                           "decodable; optional training/test label exclusion",
            "crop": "axis-aligned min/max envelope of 4 xy points; original pixels; "
                    "lossless PNG; no resizing until evaluator F=1 bicubic preprocessing",
            "excluded_label_sources": [str(Path(s)) for s in args.exclude_labels_metadata],
            "excluded_label_count": len(excluded_labels),
            "samples": final_rows,
        }
        (stage / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (stage / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in final_rows:
                writer.writerow(row)
        summary = {
            "total_candidates": dict(stats),
            "selection_rejections": dict(skipped),
            "selected": len(final_rows),
            "unique_labels": len(selected_labels),
            "unique_source_images": len(selected_sources),
            "distinct_cameras": len({r["camera_id"] for r in final_rows}),
            "vehicle_types": dict(Counter(r["vehicle_type"] for r in final_rows)),
            "time_of_day": dict(Counter(str(r["time"]) for r in final_rows)),
            "day": dict(Counter(str(r["day"]) for r in final_rows)),
            "faulty_cameras_flagged": sum(bool(r["faulty"]) for r in final_rows),
            "rain_flagged": sum(bool(r["rain"]) for r in final_rows),
            "crop_sizes_min_max": {
                "width": [min(r["crop_width"] for r in final_rows),
                          max(r["crop_width"] for r in final_rows)],
                "height": [min(r["crop_height"] for r in final_rows),
                           max(r["crop_height"] for r in final_rows)],
            },
            "selection_manifest_sha256": digest(stage / "selection_manifest.json"),
        }
        (stage / "dataset_info.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        produce_sheet(final_rows, stage, args.contact_sheet, args.seed)
        # Validate stage with temporary paths (virtual LMDB metadata targets final out).
        validation = validate_prepared(stage, args.count)
        (stage / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(out)

    print("SUCCESS: frozen LMDB and local crops ready")
    print(f"LMDB: {out}")
    print(f"Manifest: {out / 'selection_manifest.json'}")
    print(f"Contact sheet: {out / 'contact_sheet.jpg'}")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Existing LPLC_dataset directory (must contain images/)")
    p.add_argument("--annotations", type=Path, default=None,
                   help="Defaults to DATASET_ROOT/annotations_v2.json")
    p.add_argument("--output", type=Path,
                   help="New output LMDB directory; must not exist")
    p.add_argument("--audit-only", action="store_true",
                   help="Count candidate vehicle types and exclusions without writing anything")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--contact-sheet", type=int, default=60,
                   help="Number of sampled crops pictured locally for manual checking")
    p.add_argument("--exclude-labels-metadata", nargs="*", default=[],
                   help="Optional locally trusted LRLPR metadata.pkl files/LMDB dirs "
                        "whose OCR labels must be excluded")
    p.add_argument("--verify-only", action="store_true",
                   help="Verify output LMDB and uniqueness without modifying it")
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be >=1")
    if args.verify_only:
        if args.output is None:
            p.error("--verify-only requires --output")
        out = args.output.expanduser().resolve()
        result = validate_prepared(out, args.count)
        print(json.dumps(result, indent=2))
    else:
        prepare(args)


if __name__ == "__main__":
    main()
