#!/usr/bin/env python3
"""Build or verify matched 7023-sample tight and 20%-padded LPLCv2 F=1 LMDBs.

Freeze: original annotations SHA, seven-character Brazilian formats, tight
native width/height >=24x8, tight aspect [2.1,3.5], and *full padded box*
at least 2 original-frame pixels inside each edge. Reuse the same 7023
original annotations, order, labels and track IDs for both LMDBs.

No sampling, inference, training, checkpoint selection, or output overwriting.
User-private source paths, crops, sample manifest and predictions stay local.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import lmdb
import numpy as np

from audit_lplcv2_full_cohort_geometry import FORMATS, bbox, fingerprint

EXPECTED_SOURCE_SHA = "2da194f7393970b7ddf11085a9bb5adb3d6780eca49d9d7389eed0ca06d49a49"
EXPECTED_COUNTS = {"Poor": 6240, "Illegible": 783}
TOTAL = 7023
LEG = {0: "Illegible", 1: "Poor"}
RATIO_MIN = 2.1
RATIO_MAX = 3.5
MIN_WIDTH = 24
MIN_HEIGHT = 8
PAD = 0.2
FRAME_MARGIN = 2
BATCH = 200


def select(root):
    root = root.expanduser().resolve(strict=True)
    source = root / "annotations_v2.json"
    image_dir = (root / "images").resolve(strict=True)
    actual_sha = fingerprint(source)
    if actual_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Source annotations differ from the audited frozen SHA256: " + actual_sha)
    anns = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(anns, dict):
        raise ValueError("Expected source-image-keyed annotation dict")
    rows, counts = [], Counter()
    for filename, image_info in sorted(anns.items()):
        if not isinstance(image_info, dict) or not isinstance(image_info.get("anns"), list):
            continue
        path = (image_dir / filename).resolve()
        if not path.is_relative_to(image_dir):
            raise ValueError("Source annotation resolves outside images/")
        image = None
        read_attempted = False
        for index, ann in enumerate(image_info["anns"]):
            if not isinstance(ann, dict) or type(ann.get("leg")) is not int:
                continue
            if ann["leg"] not in LEG:
                continue
            cohort = LEG[ann["leg"]]
            counts[(cohort, "all_source_annotations")] += 1
            label = ann.get("ocr")
            if not isinstance(label, str) or not any(p.fullmatch(label) for p in FORMATS):
                counts[(cohort, "invalid_plate_label")] += 1
                continue
            tight = bbox(ann.get("xy"))
            if tight is None:
                counts[(cohort, "invalid_box")] += 1
                continue
            if not read_attempted:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
                read_attempted = True
            if image is None:
                counts[(cohort, "missing_or_unreadable_image")] += 1
                continue
            ih, iw = image.shape[:2]
            x1, y1, x2, y2 = tight
            if not (0 <= x1 < x2 <= iw and 0 <= y1 < y2 <= ih):
                counts[(cohort, "out_of_bounds_tight_bbox")] += 1
                continue
            width, height = x2-x1, y2-y1
            ratio = width / height
            if not RATIO_MIN <= ratio <= RATIO_MAX:
                counts[(cohort, "outside_native_aspect")] += 1
                continue
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                counts[(cohort, "under_native_size")] += 1
                continue
            # Identical floor/ceil convention to camera.crop_xywh, interpreting
            # the axis-aligned annotation envelope as the detector-box proxy.
            px1, py1 = math.floor(x1 - PAD * width), math.floor(y1 - PAD * height)
            px2, py2 = math.ceil(x2 + PAD * width), math.ceil(y2 + PAD * height)
            if min(px1, py1, iw-px2, ih-py2) < FRAME_MARGIN:
                counts[(cohort, "padded_box_under_2px_margin")] += 1
                continue
            rows.append({
                "track_id": f"track_{len(rows):06d}",
                "source_image": filename,
                "annotation_index": index,
                "ocr": label,
                "source_legibility": cohort,
                "native_tight_box_xyxy": [x1, y1, x2, y2],
                "native_padded_box_xyxy": [px1, py1, px2, py2],
                "source_width": iw, "source_height": ih,
                "tight_width": width, "tight_height": height,
                "padded_width": px2-px1, "padded_height": py2-py1,
            })
            counts[(cohort, "selected")] += 1
    if len(rows) != TOTAL:
        raise RuntimeError(f"Frozen count mismatch: expected {TOTAL}, selected {len(rows)}")
    observed = Counter(r["source_legibility"] for r in rows)
    if observed != EXPECTED_COUNTS:
        raise RuntimeError(f"Frozen cohort mismatch: {dict(observed)} != {EXPECTED_COUNTS}")
    identities = [(r["source_image"], r["annotation_index"]) for r in rows]
    if len(set(identities)) != len(rows):
        raise RuntimeError("Repeated annotation identity in selection")
    manifest = {
        "protocol": "lplcv2_full_acq20pct_margin2_native2p1to3p5_tight24x8",
        "original_annotations_sha256": actual_sha,
        "selection": {
            "source": "original LPLCv2 Poor/Illegible annotations, no subsampling",
            "plate_format": "old Brazilian LLLNNNN or Mercosur LLLNLNN",
            "tight_bbox": "floor/ceil extrema of the original quadrilateral",
            "tight_minimum_wh": [MIN_WIDTH, MIN_HEIGHT],
            "tight_native_aspect_inclusive": [RATIO_MIN, RATIO_MAX],
            "padding_each_side_fraction_of_tight_box": PAD,
            "expanded_bbox_minimum_frame_margin_px": FRAME_MARGIN,
            "source_crop_color": "BGR via cv2.imread",
            "crop_storage": "PNG lossless; evaluator resizes each crop to 96x32 bicubic",
            "vehicle_types": "all, including missing categories and motorcycles",
            "occlusion": "retained",
            "same_sample_ids_for_tight_and_padded": True,
            "duplicate_plate_strings_and_source_photos": "retained, recorded",
            "checkpoint_or_prediction_based_selection": False,
        },
        "counts": dict(observed),
        "samples": rows,
    }
    return manifest, counts


def show_summary(manifest, counts):
    rows = manifest["samples"]
    print("Frozen original annotations SHA256:", EXPECTED_SOURCE_SHA)
    print("Selected:", len(rows), "Poor:", manifest["counts"]["Poor"],
          "Illegible:", manifest["counts"]["Illegible"])
    print("Distinct source images:", len({r["source_image"] for r in rows}))
    print("Distinct OCR strings:", len({r["ocr"] for r in rows}))
    for cohort in ("Poor", "Illegible"):
        print(cohort, "selection rejections:", {
            k[1]: value for k, value in sorted(counts.items())
            if k[0] == cohort and k[1] != "selected"
        })


def encode_png(bgr):
    ok, output = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode original crop to lossless PNG")
    return output.tobytes()


def build_one(root, destination, variant, manifest):
    stage = root / variant
    stage.mkdir()
    (stage / "crops").mkdir()
    rows = manifest["samples"]
    metadata = []
    env = lmdb.open(str(stage), subdir=True, map_size=8 * 1024**3, lock=True)
    try:
        txn = env.begin(write=True)
        try:
            image_root = (destination["dataset_root"] / "images").resolve(strict=True)
            cached_file = None
            cached_image = None
            for i, r in enumerate(rows):
                if r["source_image"] != cached_file:
                    cached_image = cv2.imread(str(image_root / r["source_image"]), cv2.IMREAD_COLOR)
                    cached_file = r["source_image"]
                if cached_image is None or cached_image.shape[:2] != (
                    r["source_height"], r["source_width"]
                ):
                    raise RuntimeError("Source image missing or source geometry drift")
                box = r["native_padded_box_xyxy"] if variant == "padded" else r["native_tight_box_xyxy"]
                x1, y1, x2, y2 = box
                crop = cached_image[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[:2] != (y2-y1, x2-x1):
                    raise RuntimeError("Crop geometry mismatch")
                payload = encode_png(crop)
                key = f"image-{i:08d}".encode("ascii")
                if not txn.put(key, payload, overwrite=False):
                    raise RuntimeError("Duplicate LMDB key")
                relpath = f"crops/{r['track_id']}/lr-001.png"
                outpath = stage / relpath
                outpath.parent.mkdir(parents=True, exist_ok=False)
                outpath.write_bytes(payload)
                metadata.append({
                    "lmdb_key": key,
                    "gt": r["ocr"], "split": "test",
                    "original_path": str(destination[variant] / relpath),
                    "crop_sha256": hashlib.sha256(payload).hexdigest(),
                })
                if (i+1) % BATCH == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                if (i+1) % 1000 == 0:
                    print(variant, "created", i+1, "/", len(rows), flush=True)
            txn.commit()
            txn = None
        finally:
            if txn is not None:
                txn.abort()
    finally:
        env.sync()
        env.close()
    with (stage / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f, protocol=4)
    (stage / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (stage / "dataset_info.json").write_text(json.dumps({
        "variant": variant, "samples": len(rows),
        "counts": manifest["counts"],
        "source_annotations_sha256": EXPECTED_SOURCE_SHA,
        "manifest_sha256": fingerprint(stage / "selection_manifest.json"),
        "lossless_pixels": True,
    }, indent=2) + "\n", encoding="utf-8")


def verify_one(folder, variant, expected_manifest_sha=None, intended_final_dir=None):
    folder = folder.expanduser().resolve(strict=True)
    final_root = (Path(intended_final_dir).expanduser().absolute()
                  if intended_final_dir is not None else folder)
    info = json.loads((folder / "dataset_info.json").read_text())
    manifest_path = folder / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = fingerprint(manifest_path)
    if expected_manifest_sha and digest != expected_manifest_sha:
        raise RuntimeError("Pair manifests are not identical")
    if info["manifest_sha256"] != digest:
        raise RuntimeError("Frozen manifest changed")
    if info["variant"] != variant or len(manifest["samples"]) != TOTAL:
        raise RuntimeError("Wrong variant or sample count")
    if manifest["original_annotations_sha256"] != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Original annotation SHA mismatch")
    if manifest["counts"] != EXPECTED_COUNTS:
        raise RuntimeError("Unexpected Poor/Illegible counts")
    with (folder / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)  # local files generated only by this script
    if len(metadata) != TOTAL:
        raise RuntimeError("LMDB metadata count mismatch")
    env = lmdb.open(str(folder), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            for i, (r, meta) in enumerate(zip(manifest["samples"], metadata)):
                key = f"image-{i:08d}".encode("ascii")
                relpath = f"crops/{r['track_id']}/lr-001.png"
                target = folder / relpath
                virtual_path = final_root / relpath
                if (meta["lmdb_key"] != key or meta["gt"] != r["ocr"] or
                    meta["split"] != "test" or Path(meta["original_path"]) != virtual_path):
                    raise RuntimeError(f"Metadata mismatch at index {i}")
                data = txn.get(key)
                if data is None or not target.is_file() or target.read_bytes() != data:
                    raise RuntimeError(f"Crop/LMDB byte mismatch at index {i}")
                if hashlib.sha256(data).hexdigest() != meta["crop_sha256"]:
                    raise RuntimeError(f"Crop hash mismatch at index {i}")
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                dims = (r["padded_height"], r["padded_width"]) if variant == "padded" else (
                    r["tight_height"], r["tight_width"])
                if img is None or img.shape[:2] != dims:
                    raise RuntimeError(f"Crop decode/geometry mismatch at index {i}")
            if txn.stat()["entries"] != TOTAL:
                raise RuntimeError("Unexpected number of LMDB keys")
    finally:
        env.close()
    print("VERIFIED", variant, TOTAL, "crops; manifest SHA256", digest)
    return digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path,
                   default=Path.home() / "doc2025/LMDB-Datasets/LPLC_dataset")
    p.add_argument("--tight-output", type=Path,
                   default=Path.home() / "doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Tight_7023")
    p.add_argument("--padded-output", type=Path,
                   default=Path.home() / "doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Padded20pct_7023")
    ops = p.add_mutually_exclusive_group(required=True)
    ops.add_argument("--preflight", action="store_true")
    ops.add_argument("--build", action="store_true")
    ops.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    paths = {
        "dataset_root": args.dataset_root.expanduser().resolve(strict=True),
        "tight": args.tight_output.expanduser().absolute(),
        "padded": args.padded_output.expanduser().absolute(),
    }
    if args.verify_only:
        first = verify_one(paths["tight"], "tight")
        verify_one(paths["padded"], "padded", expected_manifest_sha=first)
        print("PAIRED COHORT VERIFIED: identical manifest and sample order")
        return
    manifest, counts = select(paths["dataset_root"])
    show_summary(manifest, counts)
    if args.preflight:
        print("PREFLIGHT ONLY; no directories created")
        return
    if paths["tight"] == paths["padded"]:
        raise ValueError("Tight and padded outputs must be different")
    if paths["tight"].parent != paths["padded"].parent:
        raise ValueError("Outputs must share a parent so staging stays on one filesystem")
    for variant in ("tight", "padded"):
        if paths[variant].exists():
            raise FileExistsError("Refusing to overwrite " + str(paths[variant]))
    parent = paths["tight"].parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lplcv2-acq-pair-", dir=parent) as td:
        stage = Path(td)
        build_one(stage, paths, "tight", manifest)
        build_one(stage, paths, "padded", manifest)
        first = verify_one(stage / "tight", "tight", intended_final_dir=paths["tight"])
        verify_one(stage / "padded", "padded", expected_manifest_sha=first,
                   intended_final_dir=paths["padded"])
        # Both complete and independently verified before publishing either.
        os.replace(stage / "tight", paths["tight"])
        os.replace(stage / "padded", paths["padded"])
    print("SUCCESS: verified matched tight/padded cohort LMDBs")
    print("Tight:", paths["tight"])
    print("Padded:", paths["padded"])
    print("Manifest SHA256:", first)


if __name__ == "__main__":
    main()
