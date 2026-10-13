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
import math
import pickle
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import lmdb
import numpy as np
import yaml


INPUT_WIDTH = 96
INPUT_HEIGHT = 32
INPUT_ASPECT = INPUT_WIDTH / INPUT_HEIGHT

DEFAULT_CNNVIT_CONFIG = "experiments/revision_eval/submitted_model/config_snapshot.yaml"
DEFAULT_SVTRV2_CONFIG = "baselines_configs/SVTRV2_AR_BASELINE.yaml"

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


def verify_model_geometry(configs):
    """Read real frozen validation wrappers and enforce identical F=1 resize.

    Image aspect-ratio metadata is checked, but the actual wrapper uses
    cv2.resize(raw, (imgW,imgH), interpolation=cv2.INTER_CUBIC), i.e.,
    the raw source aspect ratio is *not* preserved by padding.
    """
    details = {}
    for name, path in configs.items():
        config_file = Path(path)
        if not config_file.is_file():
            raise FileNotFoundError(f"{name} config not found: {config_file}")
        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        val = cfg["val_dataset"]["wrapper"]
        if val.get("name") != "VSR_Sequence_collate_fn":
            raise RuntimeError(f"{name}: wrong eval wrapper {val.get('name')}")
        opts = val["args"]
        width, height = int(opts["imgW"]), int(opts["imgH"])
        declared = float(opts["image_aspect_ratio"])
        if (width, height) != (INPUT_WIDTH, INPUT_HEIGHT):
            raise RuntimeError(f"{name}: expected 96x32 input; got {width}x{height}")
        if abs(declared - INPUT_ASPECT) > 1e-6:
            raise RuntimeError(f"{name}: declared input aspect ratio {declared}, expected 3")
        if opts.get("test") is not True or opts.get("aug") is not False:
            raise RuntimeError(f"{name}: unexpected validation augmentation/test settings")
        details[name] = {
            "config": str(config_file),
            "config_sha256": sha256(config_file),
            "validation_wrapper": val["name"],
            "declared_image_aspect_ratio": declared,
            "imgW": width,
            "imgH": height,
            "resize": "cv2.resize(raw_rgb, (96, 32), interpolation=cv2.INTER_CUBIC)",
            "padding": False,
            "warning": "Raw crops are anisotropically resized if their source aspect differs from 3:1",
        }
    return details


def geometry_stats(tagged):
    """Describe original *unaltered* crop dimensions before the model resize."""
    report = {}
    for cls in ("All", "Illegible", "Poor"):
        subset = [r for kind, r in tagged if cls == "All" or kind == cls]
        heights = np.array([int(r["crop_height"]) for r in subset], dtype=np.float64)
        widths = np.array([int(r["crop_width"]) for r in subset], dtype=np.float64)
        if (widths <= 0).any() or (heights <= 0).any():
            raise ValueError(f"{cls} has non-positive native crop dimensions")
        ratios = widths / heights
        rel_deviation = np.abs(ratios / INPUT_ASPECT - 1)
        report[cls] = {
            "samples": len(subset),
            "native_width_px_p10_median_p90": [round(float(np.percentile(widths, q)), 2)
                                               for q in (10, 50, 90)],
            "native_height_px_p10_median_p90": [round(float(np.percentile(heights, q)), 2)
                                                for q in (10, 50, 90)],
            "native_aspect_ratio_min": round(float(ratios.min()), 4),
            "native_aspect_ratio_p10_median_p90": [
                round(float(np.percentile(ratios, q)), 4) for q in (10, 50, 90)],
            "native_aspect_ratio_max": round(float(ratios.max()), 4),
            "native_aspect_ratio_exactly_3_to_1": int((widths == 3 * heights).sum()),
            "native_aspect_ratio_deviation_over_10pct": int((rel_deviation > 0.1).sum()),
            "native_aspect_ratio_deviation_over_20pct": int((rel_deviation > 0.2).sum()),
        }
    return report


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


def original_aspect(row):
    height = int(row["crop_height"])
    width = int(row["crop_width"])
    if height <= 0 or width <= 0:
        raise ValueError("Invalid native crop dimensions")
    return width / height


def aspect_filter_and_refill(base_tagged, all_illegible, all_poor,
                             seed, threshold, max_threshold,
                             refill_min, refill_max):
    """Post-hoc geometry variant: reject extreme native ratios; refill from Poor.

    Begin with EXACTLY the default seeded 905+2095 selection. Retain every
    originally selected crop within [threshold, max_threshold] (if upper
    bound is supplied). Never use model predictions. Replacements must
    be unused, nonoverlapping frozen Poor crops with near-3:1 geometry.
    """
    if threshold is None:
        if max_threshold is not None:
            raise ValueError("Maximum native aspect requires a minimum bound")
        return base_tagged, None, set()
    if not (math.isfinite(threshold) and
            (max_threshold is None or math.isfinite(max_threshold)) and
            math.isfinite(refill_min) and math.isfinite(refill_max)):
        raise ValueError("Native aspect limits must be finite")
    if not (1.0 <= threshold < refill_min <= refill_max):
        raise ValueError("Require 1.0 <= minimum < refill_min <= refill_max")
    if max_threshold is not None and not refill_max <= max_threshold:
        raise ValueError("Refill upper bound cannot exceed final native aspect maximum")

    def outside(row):
        ratio = original_aspect(row)
        return ratio < threshold or (max_threshold is not None and ratio > max_threshold)
    initial_selected_poor = {
        (r["source_image"], r["annotation_index"])
        for cls, r in base_tagged if cls == "Poor"
    }
    illegible_source_names = {r["source_image"] for r in all_illegible}
    illegible_ocr = {r["ocr"] for r in all_illegible}
    removed = [(cls, row) for cls, row in base_tagged if outside(row)]
    retained = [(cls, row) for cls, row in base_tagged if not outside(row)]
    unused_near_ideal_poor = sorted(
        (row for row in all_poor
         if (row["source_image"], row["annotation_index"]) not in initial_selected_poor
         and row["source_image"] not in illegible_source_names
         and row["ocr"] not in illegible_ocr
         and refill_min <= original_aspect(row) <= refill_max),
        key=lambda row: (row["source_image"], row["annotation_index"]),
    )
    required = len(removed)
    if len(unused_near_ideal_poor) < required:
        raise RuntimeError(
            f"Cannot produce the promised aspect-controlled 3000: "
            f"must replace {required} crops, but only "
            f"{len(unused_near_ideal_poor)} unused Poor crops have native "
            f"aspect ratio in [{refill_min}, {refill_max}]. "
            "No automatic relaxation or changed proportions is permitted.")
    replacement_poor = random.Random(seed + 17).sample(
        unused_near_ideal_poor, required)
    final_tagged = retained + [("Poor", row) for row in replacement_poor]
    random.Random(seed + 41).shuffle(final_tagged)
    if len(final_tagged) != 3000:
        raise RuntimeError("Aspect-controlled population does not have 3000 records")

    def identity(cls, row):
        return {
            "source_legibility": cls,
            "source_track_id": row["track_id"],
            "source_image": row["source_image"],
            "annotation_index": row["annotation_index"],
            "original_aspect_ratio": original_aspect(row),
            "crop_width": int(row["crop_width"]),
            "crop_height": int(row["crop_height"]),
        }

    report = {
        "enabled": True,
        "min_native_crop_aspect_ratio_inclusive": threshold,
        "max_native_crop_aspect_ratio_inclusive": max_threshold,
        "refill_native_aspect_ratio_inclusive": [refill_min, refill_max],
        "removed_below_min": dict(Counter(
            cls for cls, row in removed if original_aspect(row) < threshold)),
        "removed_above_max": dict(Counter(
            cls for cls, row in removed
            if max_threshold is not None and original_aspect(row) > max_threshold)),
        "base_pool": f"905 Illegible + 2095 seeded Poor; fixed seed {seed}",
        "sampling_policy": (
            "Remove base-pool records outside native-aspect acceptance range "
            "(inclusive bounds), then uniformly sample exactly that many "
            "unused, collision-free, near-3:1 Poor "
            "records with Random(seed+17); reshuffle using Random(seed+41)."
        ),
        "original_pool_counts": {"Illegible": 905, "Poor": 2095},
        "removed_counts": dict(Counter(cls for cls, _ in removed)),
        "removed_total": required,
        "unused_near_ideal_poor_candidates": len(unused_near_ideal_poor),
        "added_near_ideal_poor": required,
        "retained_original_samples": len(retained),
        "final_counts": dict(Counter(cls for cls, _ in final_tagged)),
        "removed_records": [identity(cls, row) for cls, row in removed],
        "replacement_records": [identity("Poor", row) for row in replacement_poor],
        "note": ("This is an intentionally post-hoc geometry-controlled variant, "
                 "not the original all-Illegible mixed dataset. Samples with "
                 "extreme narrow or wide native ratios can still be correctly "
                 "annotated. Illegible cases removed cannot be replenished from "
                 "another known-labelled Illegible pool; final class proportions "
                 "therefore change.")
    }
    replacement_ids = {(row["source_image"], row["annotation_index"])
                       for row in replacement_poor}
    return final_tagged, report, replacement_ids


def layout(gt):
    if re.fullmatch(r"[A-Z]{3}[0-9]{4}", gt):
        return "old"
    if re.fullmatch(r"[A-Z]{3}[0-9][A-Z][0-9]{2}", gt):
        return "mercosur"
    raise ValueError(f"Unexpected OCR syntax: {gt}")


def report_mix(tagged, excluded, candidate_count, seed, parents, model_geometry,
               aspect_selection=None):
    if len(tagged) != 3000:
        raise RuntimeError("Mixed test must contain exactly 3000 crops")
    classes = Counter(kind for kind, _ in tagged)
    if aspect_selection is None:
        expected = {"Illegible": 905, "Poor": 2095}
    else:
        expected = aspect_selection["final_counts"]
    if dict(classes) != expected:
        raise RuntimeError(f"Unexpected mix: {classes} vs {expected}")
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
        "selection": (
            "Post-hoc native aspect-ratio controlled replacement from frozen Poor"
            if aspect_selection is not None else
            "All original 905 Illegible + 2095 uniformly sampled from original frozen 3000 Poor"
        ),
        "native_aspect_selection": aspect_selection,
        "seed": seed,
        "selection_used_model_predictions": False,
        "source_manifest_sha256": parents,
        "model_input_geometry": model_geometry,
        "model_input_shape_c_h_w": [3, INPUT_HEIGHT, INPUT_WIDTH],
        "model_input_aspect_ratio": INPUT_ASPECT,
        "native_crop_aspect_ratio_audit": geometry_stats(tagged),
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
    selection_policy = manifest["selection_policy"]
    native_aspect_selection = selection_policy.get("native_aspect_selection")
    expected = (
        native_aspect_selection["final_counts"] if native_aspect_selection else
        {"Illegible": 905, "Poor": 2095}
    )
    if dict(kinds) != expected:
        raise RuntimeError(f"Wrong class counts: {kinds}, expected {expected}")
    if native_aspect_selection:
        threshold = float(native_aspect_selection["min_native_crop_aspect_ratio_inclusive"])
        maximum = native_aspect_selection.get("max_native_crop_aspect_ratio_inclusive")
        lower, upper = native_aspect_selection["refill_native_aspect_ratio_inclusive"]
        if any(original_aspect(row) < threshold or
               (maximum is not None and original_aspect(row) > maximum) for row in rows):
            raise RuntimeError("Native aspect-ratio acceptance interval was violated")
        replacements = [r for r in rows if r.get("selection_origin") == "near_ideal_poor_refill"]
        required = int(native_aspect_selection["removed_total"])
        if len(replacements) != required:
            raise RuntimeError("Replacement-record count mismatch")
        if any(row["source_legibility"] != "Poor" or
               not lower <= original_aspect(row) <= upper for row in replacements):
            raise RuntimeError("Replacement was not a near-ideal original Poor crop")
        if native_aspect_selection["added_near_ideal_poor"] != required:
            raise RuntimeError("Replenishment accounting differs from removed population")
        removal_rows = native_aspect_selection["removed_records"]
        reported_removals = Counter(r["source_legibility"] for r in removal_rows)
        if dict(reported_removals) != native_aspect_selection["removed_counts"]:
            raise RuntimeError("Removal record provenance/count mismatch")
        if sum(reported_removals.values()) != required:
            raise RuntimeError("Removed count inconsistent with per-record log")
        below = Counter(
            r["source_legibility"] for r in removal_rows
            if r["original_aspect_ratio"] < threshold)
        above = Counter(
            r["source_legibility"] for r in removal_rows
            if maximum is not None and r["original_aspect_ratio"] > maximum)
        if dict(below) != native_aspect_selection.get("removed_below_min", dict(below)):
            raise RuntimeError("Recorded below-minimum removals differ from summary")
        if dict(above) != native_aspect_selection.get("removed_above_max", dict(above)):
            raise RuntimeError("Recorded above-maximum removals differ from summary")
        if sum(below.values()) + sum(above.values()) != required:
            raise RuntimeError("Removed records do not violate the native aspect interval")
        expected_from_base = {
            "Illegible": 905 - reported_removals["Illegible"],
            "Poor": 2095 + reported_removals["Illegible"],
        }
        if dict(kinds) != expected_from_base:
            raise RuntimeError("Final category counts inconsistent with replacements")
        replacement_log_ids = {
            (r["source_image"], r["annotation_index"])
            for r in native_aspect_selection["replacement_records"]
        }
        actual_refill_ids = {
            (r["source_image"], r["annotation_index"]) for r in replacements
        }
        if replacement_log_ids != actual_refill_ids or len(replacement_log_ids) != required:
            raise RuntimeError("Replacement records differ from recorded decision log")
        removed_log_ids = {
            (r["source_image"], r["annotation_index"]) for r in removal_rows
        }
        present_ids = {
            (r["source_image"], r["annotation_index"]) for r in rows
        }
        if removed_log_ids & present_ids:
            raise RuntimeError("A supposedly removed out-of-range crop remains in test")
    elif any(r.get("selection_origin") == "near_ideal_poor_refill" for r in rows):
        raise RuntimeError("Unexpected replacement in unfiltered mixed LMDB")
    if len({(row["source_image"], row["annotation_index"]) for row in rows}) != 3000:
        raise RuntimeError("Duplicate annotated plate identity in assembled test")
    if len({row["track_id"] for row in rows}) != 3000:
        raise RuntimeError("Duplicate virtual track identifiers")
    sources = {
        cls: {r["source_image"] for r in rows if r["source_legibility"] == cls}
        for cls in ("Illegible", "Poor")
    }
    labels = {
        cls: {r["ocr"] for r in rows if r["source_legibility"] == cls}
        for cls in ("Illegible", "Poor")
    }
    if sources["Illegible"] & sources["Poor"] or labels["Illegible"] & labels["Poor"]:
        raise RuntimeError("Mixed set has cross-class source or OCR label overlap")
    if any(PLATE.fullmatch(r["ocr"]) is None for r in rows):
        raise RuntimeError("Mixed set contains invalid/placeholder OCR transcription")
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
                # During atomic staging metadata already points to the
                # FINAL output directory, not its temporary staging name.
                if Path(meta["original_path"]).parts[-3:] != Path(row["crop_path"]).parts[-3:]:
                    raise RuntimeError("Unexpected virtual track suffix")
                raw = txn.get(key)
                if raw is None:
                    raise RuntimeError("Missing source crop bytes in mixed LMDB")
                decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                if decoded is None or list(decoded.shape[:2]) != [
                        row["crop_height"], row["crop_width"]]:
                    raise RuntimeError("Source-crop dimension mismatch")
                # Exercise exactly the image geometry used by both frozen
                # validation wrappers; preserve raw bytes in LMDB.
                model_input = cv2.resize(
                    decoded, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
                if model_input.shape != (INPUT_HEIGHT, INPUT_WIDTH, 3):
                    raise RuntimeError("Model preprocessing failed to yield 3:1 RGB input")
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
        "all_model_inputs_96x32_bicubic": True,
        "model_input_aspect_ratio": INPUT_ASPECT,
        "native_crop_aspect_ratio_audit": geometry_stats([
            (r["source_legibility"], r) for r in rows]),
    }
    return summary


def build(args):
    configs = {"CNN-ViT": args.cnnvit_config, "SVTRv2-AR": args.svtrv2_config}
    model_geometry = verify_model_geometry(configs)
    ill_dir, ill, ill_meta, ill_sha = read_source(
        args.illegible, EXPECTED_ILLEGIBLE_SHA, 905, 0)
    poor_dir, poor, poor_meta, poor_sha = read_source(
        args.poor, EXPECTED_POOR_SHA, 3000, 1)
    base_tagged, excluded, available = sample_mix(ill, poor, 2095, args.seed)
    tagged, aspect_selection, replacement_ids = aspect_filter_and_refill(
        base_tagged, ill, poor, args.seed, args.min_native_aspect,
        args.max_native_aspect, args.refill_aspect_min, args.refill_aspect_max)
    preview = report_mix(tagged, excluded, available, args.seed, {
        "Illegible": ill_sha, "Poor": poor_sha}, model_geometry, aspect_selection)
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
                        "selection_origin": (
                            "near_ideal_poor_refill" if
                            (src["source_image"], src["annotation_index"]) in replacement_ids
                            else "original_seeded_mix"
                        ),
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
            "subset": (
                "Native-aspect-controlled Illegible/Poor mixed 3k F1 test"
                if aspect_selection else
                "Illegible 905 + Poor 2095 mixed fixed-seed F1 test"
            ),
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
    p.add_argument("--cnnvit-config", type=Path, default=Path(DEFAULT_CNNVIT_CONFIG),
                   help="Frozen CNN–ViT config to validate 96x32 evaluation geometry")
    p.add_argument("--svtrv2-config", type=Path, default=Path(DEFAULT_SVTRV2_CONFIG),
                   help="Frozen SVTRv2-AR config to validate 96x32 evaluation geometry")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min-native-aspect", type=float, default=None,
                   help="Distinct geometry-controlled variant: reject native width/height "
                        "below this inclusive limit (e.g., 1.7)")
    p.add_argument("--max-native-aspect", type=float, default=None,
                   help="Optional inclusive native aspect maximum (e.g., 4.0); "
                        "requires --min-native-aspect. Omitted for older lower-only variant.")
    p.add_argument("--refill-aspect-min", type=float, default=2.5,
                   help="Minimum native aspect ratio for replacement Poor crops (default 2.5)")
    p.add_argument("--refill-aspect-max", type=float, default=3.5,
                   help="Maximum native aspect ratio for replacement Poor crops (default 3.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report planned selection/collisions without writing")
    p.add_argument("--verify-only", action="store_true",
                   help="Verify already-built output against metadata and LMDB bytes")
    args = p.parse_args()
    if args.max_native_aspect is not None and args.min_native_aspect is None:
        p.error("--max-native-aspect requires --min-native-aspect")
    if args.min_native_aspect is not None:
        limits = (args.min_native_aspect, args.refill_aspect_min,
                  args.refill_aspect_max)
        if args.max_native_aspect is not None:
            limits += (args.max_native_aspect,)
        if not all(math.isfinite(v) for v in limits):
            p.error("Aspect bounds must be finite")
        if not 1.0 <= args.min_native_aspect < args.refill_aspect_min:
            p.error("--min-native-aspect must be >=1.0 and below --refill-aspect-min")
        if args.refill_aspect_max < args.refill_aspect_min:
            p.error("--refill-aspect-max must be >= --refill-aspect-min")
        if args.max_native_aspect is not None and args.max_native_aspect < args.refill_aspect_max:
            p.error("--max-native-aspect must be >= --refill-aspect-max")
    if args.verify_only:
        verify_model_geometry({"CNN-ViT": args.cnnvit_config,
                               "SVTRv2-AR": args.svtrv2_config})
        print(json.dumps(verify(args.output), indent=2))
    else:
        build(args)


if __name__ == "__main__":
    main()
