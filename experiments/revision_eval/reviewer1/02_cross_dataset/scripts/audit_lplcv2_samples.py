#!/usr/bin/env python3
"""Read-only LPLCv2 OCR availability and native-resolution visual audit.

Outputs class-wise counts and contact sheets comparing native annotated crop,
model-sized 96x32 input, and source-photo context. Does not change the LMDB,
the frozen manifest, or image/plate labels. Keep sheets local (restricted data).
"""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

LABEL = re.compile(r"^[A-Z0-9]{7}$")
CLASSES = {0: "Illegible", 1: "Poor", 2: "Good", 3: "Perfect"}


def get_box(pts):
    if not isinstance(pts, (list, tuple)) or len(pts) != 8:
        return None
    try:
        coords = [float(v) for v in pts]
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in coords):
        return None
    box = (math.floor(min(coords[::2])), math.floor(min(coords[1::2])),
           math.ceil(max(coords[::2])), math.ceil(max(coords[1::2])))
    return box if box[2] > box[0] and box[3] > box[1] else None


def percentile(values):
    if not values:
        return {}
    return {key: round(float(np.percentile(values, q)), 1) for key, q in
            (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100))}


def scale_fit(img, width, height, upsample=False):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    if not upsample:
        scale = min(scale, 1.0)
    ow, oh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    return cv2.resize(img, (ow, oh), interpolation=interp)


def render_contact_sheet(samples, images, outfile, cols=2):
    cell_w, cell_h = 660, 255
    rows = math.ceil(len(samples) / cols)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 247, dtype=np.uint8)
    cache = {}
    failures = []
    for i, record in enumerate(samples):
        name = record["image"]
        if name not in cache:
            if len(cache) >= 4:
                cache.clear()
            cache[name] = cv2.imread(str(images / name), cv2.IMREAD_COLOR)
        full = cache[name]
        if full is None:
            failures.append({"image": name, "reason": "decode_failed"})
            continue
        h, w = full.shape[:2]
        box = get_box(record["xy"])
        if box is None:
            failures.append({"image": name, "reason": "invalid_xy"})
            continue
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            failures.append({"image": name, "reason": "out_of_bounds"})
            continue
        crop = full[y1:y2, x1:x2]
        input96 = cv2.resize(crop, (96, 32), interpolation=cv2.INTER_CUBIC)
        # This second resize is only for DISPLAY. The model input is 96x32.
        display96 = cv2.resize(input96, (192, 64), interpolation=cv2.INTER_NEAREST)
        context = full.copy()
        cv2.rectangle(context, (x1, y1), (x2-1, y2-1), (0, 0, 255), max(1, round(w / 850)))
        context = scale_fit(context, 196, 151)
        # Native pixel crop, displayed with nearest-neighbor magnification
        # when enlarged. The annotation reports its true dimensions.
        native = scale_fit(crop, 195, 151, upsample=True)
        cell_x, cell_y = i % cols * cell_w, i // cols * cell_h
        plate_text = record.get("ocr") or "[unlabeled]"
        desc = (f"{CLASSES.get(record['leg'], '?')}  {plate_text}"
                f"  {x2-x1}x{y2-y1}px  {record['image'][:10]}")
        cv2.putText(canvas, desc[:82], (cell_x+8, cell_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
        for center_x, tile, label in ((105, context, "source image"),
                                      (319, native, "native crop (display-scaled)"),
                                      (545, display96, "96x32 input (shown 2x)")):
            left = cell_x + center_x - tile.shape[1] // 2
            top = cell_y + 34 + (155 - tile.shape[0]) // 2
            canvas[top:top+tile.shape[0], left:left+tile.shape[1]] = tile
            cv2.putText(canvas, label, (cell_x+center_x-100, cell_y+212),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"annotation #{record['idx']}, camera {record.get('cam')}",
                    (cell_x+8, cell_y+237), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (70, 70, 70), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(outfile), canvas):
        raise RuntimeError(f"Could not save {outfile}")
    return failures


def analyze(args):
    root = args.dataset_root.expanduser().resolve()
    annpath = root / "annotations_v2.json"
    imagedir = root / "images"
    if not annpath.is_file() or not imagedir.is_dir():
        raise FileNotFoundError("Need annotations_v2.json and images/ in dataset root")
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {out}")
    out.mkdir(parents=True)
    annotations = json.loads(annpath.read_text(encoding="utf-8"))
    counts = defaultdict(Counter)
    geom = defaultdict(lambda: defaultdict(list))
    pools = defaultdict(list)
    per_type = defaultdict(Counter)
    for image, meta in sorted(annotations.items()):
        if not isinstance(meta, dict):
            continue
        for index, ann in enumerate(meta.get("anns", [])):
            if not isinstance(ann, dict) or ann.get("leg") not in CLASSES:
                continue
            cls = CLASSES[ann["leg"]]
            c = counts[cls]
            c["annotations"] += 1
            label = ann.get("ocr")
            if isinstance(label, str) and label.strip():
                c["nonempty_ocr"] += 1
            else:
                c["empty_or_missing_ocr"] += 1
            if isinstance(label, str) and LABEL.fullmatch(label):
                c["valid_7char_ocr"] += 1
            else:
                c["not_valid_7char"] += 1
            if bool(ann.get("occ", False)):
                c["occluded"] += 1
            typ = ann.get("car")
            per_type[cls][str(typ.get("type")) if isinstance(typ, dict) else "MISSING"] += 1
            box = get_box(ann.get("xy"))
            if box is None:
                c["invalid_bbox"] += 1
                continue
            x1, y1, x2, y2 = box
            width, height = x2-x1, y2-y1
            geom[cls]["widths"].append(width)
            geom[cls]["heights"].append(height)
            geom[cls]["aspect_ratios"].append(width / height)
            c["bbox_width_at_most_96"] += width <= 96
            c["bbox_width_at_most_48"] += width <= 48
            c["bbox_height_at_most_32"] += height <= 32
            pools[cls].append({"image": image, "idx": index, "leg": ann["leg"],
                               "ocr": label, "xy": ann["xy"], "cam": meta.get("cam")})
    report = {"dataset": str(root), "classes": {}, "selected": None,
              "note": "Pixel dimensions alone do NOT establish blur, native degradation, or perceptual quality."}
    for cls in ("Perfect", "Good", "Poor", "Illegible"):
        report["classes"][cls] = {
            "counts": dict(counts[cls]),
            "width_px": percentile(geom[cls]["widths"]),
            "height_px": percentile(geom[cls]["heights"]),
            "aspect_ratio": percentile(geom[cls]["aspect_ratios"]),
            "vehicle_types_top": per_type[cls].most_common(12),
        }
    rng = random.Random(args.seed)
    failures = {}
    for cls in ("Perfect", "Good", "Poor", "Illegible"):
        candidates = pools[cls]
        chosen = rng.sample(candidates, min(args.per_class, len(candidates)))
        failures[cls] = render_contact_sheet(chosen, imagedir,
                                              out / ("legibility_" + cls.lower() + ".jpg"))
    if args.manifest:
        manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
        rows = manifest["samples"]
        if len(rows) != 3000:
            raise ValueError(f"Unexpected selected set size: {len(rows)}")
        selected = []
        for row in rows:
            name, idx = row["source_image"], row["annotation_index"]
            ann = annotations[name]["anns"][idx]
            if ann["ocr"] != row["ocr"]:
                raise RuntimeError(f"Mismatch in frozen manifest: {name}[{idx}]")
            selected.append({"image": name, "idx": idx, "leg": ann["leg"],
                             "ocr": ann["ocr"], "xy": ann["xy"],
                             "cam": annotations[name].get("cam"),
                             "width": row["crop_width"], "height": row["crop_height"]})
        widths = [r["width"] for r in selected]
        heights = [r["height"] for r in selected]
        report["selected"] = {
            "total": len(selected), "width_px": percentile(widths),
            "height_px": percentile(heights),
            "n_native_width_at_most_96": sum(w <= 96 for w in widths),
            "n_native_width_at_most_48": sum(w <= 48 for w in widths),
            "n_native_height_at_most_32": sum(h <= 32 for h in heights),
            "n_native_width_over_96": sum(w > 96 for w in widths),
            "n_native_height_over_32": sum(h > 32 for h in heights),
            "original_manifest_sha256": hashlib.sha256(
                args.manifest.expanduser().read_bytes()).hexdigest(),
        }
        groups = {
            "selected_random": rng.sample(selected, min(args.selected_count, len(selected))),
            "selected_smallest_heights": sorted(selected, key=lambda r: (r["height"], r["width"]))[:args.selected_count],
            "selected_largest_widths": sorted(selected, key=lambda r: (-r["width"], -r["height"]))[:args.selected_count],
        }
        for label, chosen in groups.items():
            failures[label] = render_contact_sheet(chosen, imagedir, out / (label + ".jpg"))
    report["image_failures"] = failures
    (out / "audit_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                                             encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRead-only QA images and audit saved under: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   help="Optional frozen external selection_manifest.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--per-class", type=int, default=12)
    p.add_argument("--selected-count", type=int, default=24)
    args = p.parse_args()
    if args.per_class < 1 or args.selected_count < 1:
        p.error("Counts must be >=1")
    analyze(args)


if __name__ == "__main__":
    main()
