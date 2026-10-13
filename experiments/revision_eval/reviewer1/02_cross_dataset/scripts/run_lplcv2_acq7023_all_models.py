#!/usr/bin/env python3
"""Frozen 7,023-crop LPLCv2 acquisition-style tight vs 20%-padded F=1 evaluation.

Reuse the validation-selected eight checkpoint identities from the previous
locked LPLCv2 experiment, without retuning. Never publish predictions or
GT/sample identities. Results are stored under the Git-ignored results/ path.
No paired significance testing is performed.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from run_lplcv2_all_models import (
    BASE, ROOT, MODELS, config_for, load_lock,
    preflight_config, sha256
)
from summarize_mixed_legibility_predictions import (
    edit_distance, normalize_plate, positional_matches, stats
)

MANIFEST_SHA = "107d53e3c2f48ecfe118bb222a828080f6750632b7c9b51a523ef186347ffb48"
SOURCE_SHA = "2da194f7393970b7ddf11085a9bb5adb3d6780eca49d9d7389eed0ca06d49a49"
EXPECTED = {"Poor": 6240, "Illegible": 783}
N = 7023
VARIANTS = ("tight", "padded")


def dataset_paths(args):
    return {
        "tight": args.tight_dataset.expanduser().resolve(strict=True),
        "padded": args.padded_dataset.expanduser().resolve(strict=True),
    }


def check_datasets(datasets):
    if datasets["tight"] == datasets["padded"]:
        raise ValueError("Tight and padded directories cannot be the same")
    row_lists = {}
    for variant, dataset in datasets.items():
        manifest_path = dataset / "selection_manifest.json"
        if sha256(manifest_path) != MANIFEST_SHA:
            raise ValueError(f"{variant}: frozen manifest SHA256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = json.loads((dataset / "dataset_info.json").read_text(encoding="utf-8"))
        rows = manifest["samples"]
        if (manifest["original_annotations_sha256"] != SOURCE_SHA or
            info["source_annotations_sha256"] != SOURCE_SHA or
            info["manifest_sha256"] != MANIFEST_SHA or
            info["variant"] != variant):
            raise ValueError(f"{variant}: original source/pair protocol mismatch")
        if len(rows) != N or Counter(r["source_legibility"] for r in rows) != EXPECTED:
            raise ValueError(f"{variant}: unexpected frozen cohort composition")
        if info["counts"] != EXPECTED or info["samples"] != N:
            raise ValueError(f"{variant}: unexpected dataset_info summary")
        ids = [r["track_id"] for r in rows]
        identities = [(r["source_image"], r["annotation_index"]) for r in rows]
        if len(set(ids)) != N or len(set(identities)) != N:
            raise ValueError(f"{variant}: duplicate sample identity")
        for required in ("metadata.pkl", "data.mdb"):
            if not (dataset / required).is_file():
                raise FileNotFoundError(dataset / required)
        row_lists[variant] = ids
        print(f"{variant}: verified manifest {MANIFEST_SHA[:16]}... "
              f"({EXPECTED['Poor']} Poor + {EXPECTED['Illegible']} Illegible)")
    if row_lists["tight"] != row_lists["padded"]:
        raise ValueError("Paired datasets have different sample IDs/order")
    print("Identical, frozen 7,023-annotation population across both conditions.")


def output_dir(variant, slug):
    return BASE / "results" / f"lplcv2_acq7023_{variant}_{slug}"


def complete(entry, dataset, variant, checkpoint=None):
    model, slug, _, loss = entry
    output = output_dir(variant, slug)
    metrics_file = output / "F1_bayes.json"
    csv_file = output / "F1_bayes_predictions.csv"
    if not metrics_file.exists() and not csv_file.exists():
        return None
    if not metrics_file.is_file() or not csv_file.is_file():
        raise RuntimeError(f"{model} {variant}: incomplete result. Inspect {output}")
    doc = json.loads(metrics_file.read_text(encoding="utf-8"))
    if Path(doc["dataset"]).expanduser().resolve() != dataset:
        raise ValueError(f"{model} {variant}: metrics belong to another dataset")
    if (doc["mode"] != "val" or int(doc["frames"]) != 1 or
        doc["tta"] or doc["swa"] or doc["cls_loss"] != loss):
        raise ValueError(f"{model} {variant}: different inference protocol")
    m = doc["metrics"]
    if int(m["sequence_total"]) != N or int(m["character_total"]) != 7 * N or int(m["ground_truth_characters"]) != 7 * N:
        raise ValueError(f"{model} {variant}: unexpected global denominator")
    if checkpoint is not None:
        if [Path(x).name for x in doc["checkpoint_files"]] != [checkpoint.name]:
            raise ValueError(f"{model} {variant}: wrong validation-selected checkpoint")
    return doc


def checkpoint_preflight(lock, datasets):
    # The previous lock deliberately contains the old frozen manifest SHA;
    # importing load_lock verifies that untouched checkpoint selection.
    ckpts = load_lock(lock)
    print("Using the previously validation-selected eight checkpoints.")
    for entry in MODELS:
        model, slug, config, loss = entry
        preflight_config(config_for(config), loss)
        ckpt = ckpts[model]
        print(f"\n{model}: {ckpt.name}, checkpoint SHA256 {sha256(ckpt)}")
        for variant in VARIANTS:
            found = complete(entry, datasets[variant], variant, ckpt)
            output = output_dir(variant, slug)
            if found is None and output.exists() and any(output.iterdir()):
                raise RuntimeError(f"{model} {variant}: nonempty incomplete output {output}")
            print(f"  {variant}: {'complete (skip)' if found else 'pending'}")
    return ckpts


def run_all(ckpts, datasets, variants):
    for variant in variants:
        dataset = datasets[variant]
        for entry in MODELS:
            model, slug, config, _ = entry
            checkpoint = ckpts[model]
            if complete(entry, dataset, variant, checkpoint) is not None:
                print(f"SKIP {variant} / {model} (complete, checkpoint validated)", flush=True)
                continue
            output = output_dir(variant, slug)
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"Refusing nonempty incomplete output: {output}")
            links = output / "frozen_checkpoint_only"
            links.mkdir(parents=True, exist_ok=False)
            (links / checkpoint.name).symlink_to(checkpoint)
            cmd = [
                sys.executable,
                "experiments/revision_eval/shared/scripts/run_eval_matrix.py",
                "--config", config_for(config),
                "--checkpoints", str(links),
                "--split", str(dataset),
                "--output-dir", str(output),
                "--frames", "1",
                "--fusions", "bayes",
            ]
            print(f"RUN {variant} / {model}: " + " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            if complete(entry, dataset, variant, checkpoint) is None:
                raise RuntimeError(f"{variant}/{model}: missing results after inference")
    print("Requested variant evaluations are complete; generate verified subgroup reports.")


def verify_predictions(entry, dataset, variant):
    """Verify 7,023 per-annotation predictions and original integer totals."""
    model, slug, _, _ = entry
    result = complete(entry, dataset, variant)
    if result is None:
        return None
    rows = json.loads((dataset / "selection_manifest.json").read_text(encoding="utf-8"))["samples"]
    by_track = {r["track_id"]: r for r in rows}
    if len(by_track) != N:
        raise ValueError("Manifest duplicate track IDs")
    found = set()
    groups = {"All": [], "Poor": [], "Illegible": []}
    csv_path = output_dir(variant, slug) / "F1_bayes_predictions.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"track", "prediction", "ground_truth", "correct",
                    "positional_matches", "edit_distance"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{model} {variant}: missing prediction columns")
        for i, item in enumerate(reader, start=2):
            tid = item["track"]
            if tid not in by_track or tid in found:
                raise ValueError(f"{model} {variant}: unknown/duplicate row {i}")
            found.add(tid)
            row = by_track[tid]
            gt = normalize_plate(item["ground_truth"])
            if gt != row["ocr"] or len(gt) != 7:
                raise ValueError(f"{model} {variant}: GT mismatch at row {i}")
            pred = normalize_plate(item["prediction"])
            correct = pred == gt
            matches = positional_matches(pred, gt)
            edits = edit_distance(pred, gt)
            if item["correct"].lower() not in ("true", "false"):
                raise ValueError(f"{model} {variant}: bad boolean at row {i}")
            if ((item["correct"].lower() == "true") != correct or
                int(item["positional_matches"]) != matches or
                int(item["edit_distance"]) != edits):
                raise ValueError(f"{model} {variant}: prediction metric mismatch row {i}")
            record = {"correct": int(correct), "matches": matches, "edits": edits, "gt": gt}
            groups["All"].append(record)
            groups[row["source_legibility"]].append(record)
    if len(found) != N or found != set(by_track):
        raise ValueError(f"{model} {variant}: missing predictions")
    combined = {cohort: stats(records) for cohort, records in groups.items()}
    if combined["Poor"]["samples"] != EXPECTED["Poor"] or combined["Illegible"]["samples"] != EXPECTED["Illegible"]:
        raise ValueError(f"{model} {variant}: incorrect subgroup denominators")
    totals = combined["All"]
    m = result["metrics"]
    for metric in ("sequence_correct", "sequence_total", "character_correct",
                   "character_total", "edit_errors", "ground_truth_characters",
                   "partial_5_correct", "partial_6_correct"):
        local = "samples" if metric == "sequence_total" else metric
        if totals[local] != int(m[metric]):
            raise ValueError(f"{model} {variant}: recomputed {metric} mismatches original report")
    for metric in ("sequence_accuracy_percent", "character_accuracy_percent", "cer_percent"):
        if abs(totals[metric] - float(m[metric])) > 1e-6:
            raise ValueError(f"{model} {variant}: recomputed {metric} rate mismatch")
    for metric in ("samples", "sequence_correct", "character_correct", "character_total",
                   "edit_errors", "ground_truth_characters", "partial_6_correct", "partial_5_correct"):
        if combined["Poor"][metric] + combined["Illegible"][metric] != totals[metric]:
            raise ValueError(f"{model} {variant}: subgroup counts do not add up")
    return combined


def render_table(variant, cohort, rows):
    n = N if cohort == "All" else EXPECTED[cohort]
    print(f"\n### {variant.capitalize()} / {cohort} (n={n:,})")
    print("| Model | 7/7 (%) | >=6 (%) | >=5 (%) | Char. acc. (%) | CER (%) | Exact correct |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model, group in rows:
        if group is None:
            print(f"| {model} | pending | pending | pending | pending | pending | - |")
            continue
        m = group[cohort]
        print(f"| {model} | {m['sequence_accuracy_percent']:.2f} | "
              f"{100*m['partial_6_correct']/n:.2f} | "
              f"{100*m['partial_5_correct']/n:.2f} | "
              f"{m['character_accuracy_percent']:.2f} | "
              f"{m['cer_percent']:.2f} | "
              f"{m['sequence_correct']}/{n} |")


def report(datasets, variants):
    all_groups = {}
    for variant in variants:
        rows = []
        for entry in MODELS:
            model = entry[0]
            groups = verify_predictions(entry, datasets[variant], variant)
            rows.append((model, groups))
        all_groups[variant] = dict(rows)
        for cohort in ("All", "Poor", "Illegible"):
            render_table(variant, cohort, rows)
        print(f"{variant} fully verified: {sum(groups is not None for _, groups in rows)}/8 models")
    if set(variants) == set(VARIANTS):
        print("\n### Padded minus tight (percentage points, identical source identities)")
        print("| Model | All 7/7 | Poor 7/7 | Illegible 7/7 | All char. acc. | All CER |")
        print("|---|---:|---:|---:|---:|---:|")
        for model, *_ in MODELS:
            tight = all_groups["tight"][model]
            padded = all_groups["padded"][model]
            if tight is None or padded is None:
                print(f"| {model} | pending | pending | pending | pending | pending |")
                continue
            def delta(cohort, key):
                return padded[cohort][key] - tight[cohort][key]
            print(f"| {model} | {delta('All','sequence_accuracy_percent'):+.2f} | "
                  f"{delta('Poor','sequence_accuracy_percent'):+.2f} | "
                  f"{delta('Illegible','sequence_accuracy_percent'):+.2f} | "
                  f"{delta('All','character_accuracy_percent'):+.2f} | "
                  f"{delta('All','cer_percent'):+.2f} |")
    print("\nAll figures are as-annotated, descriptive; no paired significance test.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    operations = p.add_mutually_exclusive_group(required=True)
    operations.add_argument("--preflight", action="store_true")
    operations.add_argument("--run", action="store_true")
    operations.add_argument("--report", action="store_true")
    p.add_argument("--variant", choices=("both", "tight", "padded"), default="both")
    p.add_argument("--tight-dataset", type=Path, default=(
        Path.home() / "doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Tight_7023"))
    p.add_argument("--padded-dataset", type=Path, default=(
        Path.home() / "doc2025/LMDB-Datasets/LPLCv2_AcqStrict_Padded20pct_7023"))
    p.add_argument("--lock", type=Path, default=(
        BASE / "FROZEN_LPLCV2_CHECKPOINTS.local.json"))
    args = p.parse_args()
    datasets = dataset_paths(args)
    check_datasets(datasets)
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    print("Using isolated 7,023-sample results paths; old 3,000-sample results unchanged.")
    if args.report:
        report(datasets, variants)
    else:
        checkpoints = checkpoint_preflight(args.lock, datasets)
        if args.run:
            run_all(checkpoints, datasets, variants)


if __name__ == "__main__":
    main()
