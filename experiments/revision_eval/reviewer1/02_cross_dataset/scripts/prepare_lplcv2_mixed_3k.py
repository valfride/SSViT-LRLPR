#!/usr/bin/env python3
"""Build a frozen 3000-crop LPLCv2 mixed Illegible/Poor F=1 test LMDB.

Source populations: ALL 905 validated-format Illegible annotations, plus 2095
samples selected from the PREVIOUSLY FROZEN Poor 3000 test LMDB.
Selection is prediction-blind and deterministic (seed 2026). We avoid
cross-subset source-photograph and OCR-string overlap when drawing Poor,
but do not remove internal duplicate OCR/source identities in Illegible.

Reads existing crops directly as LMDB PNG bytes. Never changes either source
dataset, annotation file, selection manifest or model predictions.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import lmdb
import numpy as np


EXPECTED_ILLEGIBLE_SHA = "7e74f46d991db9835739e7b99b5c01f1be6da7b21ba1b50b2c9bef848b7dfb0e"
EXPECTED_POOR_SHA = "8a08180604d88b652c6d915853830b4b1fd0a9ff6bcceace597d8a57f199d408"
PLATE = re.compile(r"^(?:[A-Z]{3}[0-9]{4}|[A-Z]{3}[0-9][A-Z][0-9]{2})$")
TRACK = re.compile(r"^track_[0-9]+$")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_source(folder, expected_hash, expected_total, legibility):
    folder = folder.expanduser().resolve()
    manifest_path = folder / "selection_manifest.json"
    observed_hash = sha256(manifest_path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"{legibility} manifest hash mismatch: {observed_hash} "
            f"(expected {expected_hash}). Do not silently change frozen source.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if len(samples) != expected_total:
        raise RuntimeError(f"{legibility} expected {expected_total}, got {len(samples)}")
    identities = set()
    track_ids = set()
    for entry in samples:
        if entry["legibility"] != legibility:
            raise RuntimeError(f"{legibility} manifest contains another legibility class")
        if PLATE.fullmatch(entry["ocr"]) is None:
            raise RuntimeError(f"Invalid or placeholder OCR field: {entry['ocr']!r}")
        if entry["track_id"] in track_ids:
            raise RuntimeError("Duplicate track identifier in source subset")
        track_ids.add(entry["track_id"])
        identity = (entry["source_image"], entry["annotation_index"])
        if identity in identities:
            raise RuntimeError("Duplicate source image/annotation identity")
        identities.add(identity)
    if not (folder / "data.mdb").is_file() or not (folder / "metadata.pkl").is_file():
        raise FileNotFoundError(f"Missing original LMDB data/metadata: {folder}")
    with (folder / "metadata.pkl").open("rb") as file:
        metadata = pickle.load(file)  # locally generated dataset only
    if len(metadata) != len(samples):
        raise RuntimeError(f"Metadata/manifest count mismatch in {folder}")
    rows = {}
    # The original generators write metadata and samples in the same order,
    # but we additionally match the track identifier embedded in original_path.
    for meta in metadata:
        path = Path(meta["original_path"])
        old_track = path.parent.name
        if not TRACK.fullmatch(old_track):
            raise RuntimeError(f"Cannot parse source LMDB track from {path}")
        if old_track in rows:
            raise RuntimeError(f"Duplicate source LMDB track {old_track}")
        if meta["split"] != "test" or not isinstance(meta["lmdb_key"], bytes):
            raise RuntimeError("Expected test-split byte-key LMDB record")
        rows[old_track] = meta
    for sample in samples:
        if sample["track_id"] not in rows:
            raise RuntimeError("Missing original LMDB metadata for " + sample["track_id"])
        if rows[sample["track_id"]]["gt"] != sample["ocr"]:
            raise RuntimeError("Source ground-truth/manifest mismatch")
    return folder, samples, rows, observed_hash


def sample_mix(illegible, poor, poor_needed, seed):
    illegible_sources = {x["source_image"] for x in illegible}
    illegible_labels = {x["ocr"] for x in illegible}
    excluded = Counter()
    candidate_poor = []
    for item in poor:
        if item["source_image"] in illegible_sources:
            excluded["shared_source_image_with_illegible"] += 1
            continue
        if item["ocr"] in illegible_labels:
            excluded["shared_ocr_label_with_illegible"] += 1
            continue
        candidate_poor.append(item)
    if len(candidate_poor) < poor_needed:
        raise RuntimeError(
            f"Only {len(candidate_poor)} Poor samples remain after cross-subset "
            f"source/label exclusions, need {poor_needed}. Do not reduce the "
            "905 Illegible entries or silently relax constraints.")
    candidate_poor.sort(key=lambda x: (x["source_image"], x["annotation_index"]))
    chosen_poor = random.Random(seed).sample(candidate_poor, poor_needed)
    # Shuffle final combined order, independent of the models' predictions.
    tagged = [("Illegible", x) for x in illegible]
    tagged.extend(("Poor", x) for x in chosen_poor)
    random.Random(seed + 1).shuffle(tagged)
    return tagged, excluded, len(candidate_poor)


def layout(gt):
    if re.fullmatch(r"[A-Z]{3}[0-9]{4}", gt):
        return "old"
    if re.fullmatch(r"[A-Z]{3}[0-9][A-Z][0-9]{2}", gt):
        return "mercosur"
    raise ValueError(f"Unexpected OCR syntax: {gt}")


def report_mix(tagged, excluded, candidate_count, seed, parents):
    if len(tagged) != 3000:
        raise RuntimeError("Mixed test must contain exactly 3000 crops")
    classes = Counter(kind for kind, _ in tagged)
    if classes != {"Illegible": 905, "Poor": 2095}:
        raise RuntimeError(f"Unexpected mix: {classes}")
    seen_ids = set()
    by_class_sources, by_class_labels = {}, {}
    formats = Counter()
    for kind in ("Illegible", "Poor"):
        subset = [r for cls, r in tagged if cls == kind]
        by_class_sources[kind] = {r["source_image"] for r in subset}
        by_class_labels[kind] = {r["ocr"] for r in subset}
    if by_class_sources["Poor"] & by_class_sources["Illegible"]:
        raise RuntimeError("Cross-class source-photo overlap")
    if by_class_labels["Poor"] & by_class_labels["Illegible"]:
        raise RuntimeError("Cross-class OCR-label overlap")
    for kind, item in tagged:
        identity = (item["source_image"], item["annotation_index"])
        if identity in seen_ids:
            raise RuntimeError("Repeated annotated plate identity in mix")
        seen_ids.add(identity)
        formats[f"{kind}_{layout(item['ocr'])}"] += 1
    return {
        "selection": "All original 905 Illegible + 2095 uniformly sampled from original frozen 3000 Poor",
        "seed": seed,
        "selection_used_model_predictions": False,
        "source_manifest_sha256": parents,
        "counts": dict(classes),
        "total_plate_crops": len(tagged),
        "unique_source_photographs": len({r["source_image"] for _, r in tagged}),
        "unique_plate_labels": len({r["ocr"] for _, r in tagged}),
        "unique_annotation_identities": len(seen_ids),
        "original_poor_eligible_after_overlap_filter": candidate_count,
        "poor_overlap_exclusions": dict(excluded),
        "plate_formats": dict(formats),
        "source_overlap_between_groups": 0,
        "label_overlap_between_groups": 0,
        "important": ("Multiple separately annotated Illegible plates can occur in "
                      "one source photo; do not claim 3000 unique source photos. "
                      "This post-hoc-defined mixed-difficulty subset is exploratory "
                      "and not directly comparable to the original 3000 Poor-only score."),
    }


def verify(folder, expected_hash=None):
    folder = folder.expanduser().resolve()
    manifest_path = folder / "selection_manifest.json"
    if expected_hash and sha256(manifest_path) != expected_hash:
        raise RuntimeError("Mixed manifest hash differs from expected")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["samples"]
    if len(rows) != 3000:
        raise RuntimeError("Expected mixed 3000 crops")
    kinds = Counter(row["source_legibility"] for row in rows)
    if kinds != {"Illegible": 905, "Poor": 2095}:
        raise RuntimeError(f"Wrong class counts: {kinds}")
    with (folder / "metadata.pkl").open("rb") as inp:
        metadata = pickle.load(inp)
    if len(metadata) != len(rows):
        raise RuntimeError("Mixed LMDB metadata count mismatch")
    env = lmdb.open(str(folder), readonly=True, lock=False, readahead=False, max_readers=4)
    try:
        with env.begin() as txn:
            for row, meta in zip(rows, metadata):
                if row["ocr"] != meta["gt"] or meta["split"] != "test":
                    raise RuntimeError("Ground-truth/phase mismatch")
                key = meta["lmdb_key"]
                if not isinstance(key, bytes) or key is None:
                    raise RuntimeError("Bad LMDB key")
                dest = folder / row["crop_path"]
                if Path(meta["original_path"]) != dest:
                    raise RuntimeError("Unexpected virtual track path")
                raw = txn.get(key)
                if raw is None:
                    raise RuntimeError("Missing source crop bytes in mixed LMDB")
                decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                if decoded is None or list(decoded.shape[:2]) != [
                        row["crop_height"], row["crop_width"]]:
                    raise RuntimeError("Source-crop dimension mismatch")
                if not dest.is_file() or dest.read_bytes() != raw:
                    raise RuntimeError("Mixed saved crop differs from LMDB bytes")
    finally:
        env.close()
    summary = {
        "verified": len(rows),
        "manifest_sha256": sha256(manifest_path),
        "class_counts": dict(kinds),
        "unique_source_photographs": len({r["source_image"] for r in rows}),
        "unique_plate_labels": len({r["ocr"] for r in rows}),
        "all_test_split": True,
        "all_crops_lmdb_match": True,
    }
    return summary


def build(args):
    ill_dir, ill, ill_meta, ill_sha = read_source(
        args.illegible, EXPECTED_ILLEGIBLE_SHA, 905, 0)
    poor_dir, poor, poor_meta, poor_sha = read_source(
        args.poor, EXPECTED_POOR_SHA, 3000, 1)
    tagged, excluded, available = sample_mix(ill, poor, 2095, args.seed)
    preview = report_mix(tagged, excluded, available, args.seed, {
        "Illegible": ill_sha, "Poor": poor_sha})
    print(json.dumps(preview, indent=2), flush=True)
    if args.dry_run:
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite mixed benchmark: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=output.parent, prefix=output.name + ".staging-") as temp:
        stage = Path(temp)
        (stage / "crops").mkdir()
        original_envs = {
            "Illegible": lmdb.open(str(ill_dir), readonly=True, lock=False,
                                   readahead=False, max_readers=2),
            "Poor": lmdb.open(str(poor_dir), readonly=True, lock=False,
                              readahead=False, max_readers=2),
        }
        written_rows = []
        metadata = []
        new_env = lmdb.open(str(stage), map_size=4 * (1024 ** 3),
                            subdir=True, max_dbs=1)
        try:
            with original_envs["Illegible"].begin() as itxn, \
                 original_envs["Poor"].begin() as ptxn, \
                 new_env.begin(write=True) as dest_txn:
                txn_by_class = {"Illegible": itxn, "Poor": ptxn}
                meta_by_class = {"Illegible": ill_meta, "Poor": poor_meta}
                for i, (cls, src) in enumerate(tagged):
                    parent_meta = meta_by_class[cls][src["track_id"]]
                    raw = txn_by_class[cls].get(parent_meta["lmdb_key"])
                    if raw is None:
                        raise RuntimeError(f"Missing source record {cls}/{src['track_id']}")
                    parsed = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                                          cv2.IMREAD_COLOR)
                    if parsed is None or list(parsed.shape[:2]) != [
                            src["crop_height"], src["crop_width"]]:
                        raise RuntimeError("Original crop not decodable or geometry changed")
                    new_track = f"track_{i:06d}"
                    relpath = f"crops/{new_track}/lr-001.png"
                    dst = stage / relpath
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(raw)
                    key = f"image-{i:08d}".encode("ascii")
                    if not dest_txn.put(key, raw, overwrite=False):
                        raise RuntimeError("Duplicate output LMDB key")
                    metadata.append({
                        "lmdb_key": key, "gt": src["ocr"], "split": "test",
                        "original_path": str(output / relpath),
                    })
                    record = dict(src)
                    record.update({
                        "track_id": new_track, "crop_path": relpath,
                        "source_legibility": cls,
                        "source_track_id": src["track_id"],
                        "source_dataset_directory": str(
                            ill_dir if cls == "Illegible" else poor_dir),
                    })
                    written_rows.append(record)
                    if (i + 1) % 500 == 0:
                        print(f"Copied {i + 1}/{len(tagged)} unchanged PNG crops", flush=True)
        finally:
            new_env.sync()
            new_env.close()
            for env in original_envs.values():
                env.close()
        with (stage / "metadata.pkl").open("wb") as f:
            pickle.dump(metadata, f, protocol=4)
        final_manifest = {
            "dataset": "LPLCv2",
            "subset": "Illegible 905 + Poor 2095 mixed fixed-seed F1 test",
            "count": 3000, "frames": 1,
            "selection_policy": preview,
            "samples": written_rows,
        }
        (stage / "selection_manifest.json").write_text(
            json.dumps(final_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        info = {
            **preview, "manifest_sha256": sha256(stage / "selection_manifest.json"),
            "outputs": ["data.mdb", "metadata.pkl", "selection_manifest.json",
                        "dataset_info.json", "validation.json", "crops/"],
        }
        (stage / "dataset_info.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8")
        validation = verify(stage)
        (stage / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print("SUCCESS: new mixed F1 LMDB prepared and verified")
    print(json.dumps(info, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--illegible", type=Path, required=True)
    p.add_argument("--poor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dry-run", action="store_true",
                   help="Report planned selection/collisions without writing")
    p.add_argument("--verify-only", action="store_true",
                   help="Verify already-built output against metadata and LMDB bytes")
    args = p.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.output), indent=2))
    else:
        build(args)


if __name__ == "__main__":
    main()
