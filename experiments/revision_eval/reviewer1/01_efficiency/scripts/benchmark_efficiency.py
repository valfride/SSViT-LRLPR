#!/usr/bin/env python3
"""Run Reviewer-1 efficiency benchmarks for the proposed model and baselines.

The orchestrator prepares one fixed cache of real TEST_3k tracklets, launches
each architecture in a fresh subprocess, and aggregates per-model JSON outputs
into CSV/Markdown tables plus a run manifest.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # noqa: E402


EFFICIENCY_DIR = (
    ROOT
    / "experiments/revision_eval/reviewer1/01_efficiency"
)

MODEL_SPECS = {
    "ours": {
        "display_name": "Ours",
        "config": ROOT
        / "experiments/revision_eval/submitted_model/config_snapshot.yaml",
        "checkpoints": ROOT / "experiments/revision_eval/submitted_model",
    },
    "svtrv2": {
        "display_name": "SVTRv2",
        "config": ROOT / "baselines_configs/SVTRV2_BASELINE.yaml",
        "checkpoints": ROOT
        / "experiments/baselines/SVTRV2_BASELINE",
    },
    "ote": {
        "display_name": "OTE",
        "config": ROOT / "baselines_configs/OTE_BASELINE.yaml",
        "checkpoints": ROOT
        / "experiments/baselines/OTE_BASELINE",
    },
    "lister": {
        "display_name": "LISTER",
        "config": ROOT / "baselines_configs/LISTER_BASELINE.yaml",
        "checkpoints": ROOT
        / "experiments/baselines/LISTER_BASELINE",
    },
    "igtr": {
        "display_name": "IGTR",
        "config": ROOT / "baselines_configs/IGTR_BASELINE.yaml",
        "checkpoints": ROOT
        / "experiments/baselines/IGTR_BASELINE",
    },
    "cppd": {
        "display_name": "CPPD",
        "config": ROOT / "baselines_configs/CPPD_BASELINE.yaml",
        "checkpoints": ROOT
        / "experiments/baselines/CPPD_BASELINE",
    },
    "mdiff": {
        "display_name": "MDiff4STR",
        "config": ROOT / "baselines_configs/MDIFF_BASELINE.yaml",
        "checkpoints": ROOT
        / "experiments/baselines/MDIFF_BASELINE",
    },
}

DEFAULT_MODELS = tuple(MODEL_SPECS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_real_input_cache(
    dataset_path: Path,
    output_path: Path,
    sample_count: int,
    config_path: Path,
) -> dict:
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_spec = copy.deepcopy(config["val_dataset"]["dataset"])
    dataset_spec["args"]["path_split"] = str(dataset_path)
    dataset_spec["args"]["phase"] = "test"
    # The wrapper groups every track during construction. Avoid additionally
    # caching compressed LMDB bytes in RAM.
    dataset_spec["args"]["in_memory"] = False

    print(f"\nPreparing fixed real-input cache from {dataset_path}")
    base_dataset = datasets.make(dataset_spec)

    wrapper_spec = copy.deepcopy(config["val_dataset"]["wrapper"])
    wrapper_spec["args"]["dataset"] = base_dataset
    wrapper_spec["args"]["test"] = True
    wrapper_spec["args"]["in_images"] = 5
    val_dataset = datasets.make(wrapper_spec)

    count = min(sample_count, len(val_dataset))
    if count < 1:
        raise RuntimeError("External TEST_3k wrapper yielded zero tracklets.")

    sequences = []
    labels = []
    names = []
    for index in range(count):
        item = val_dataset[index]
        lr_sequence = item["lr_seq"]
        if lr_sequence.ndim != 4 or lr_sequence.shape[0] < 5:
            raise ValueError(
                f"Unexpected lr_seq shape at index {index}: "
                f"{tuple(lr_sequence.shape)}"
            )
        sequences.append(lr_sequence[:5].contiguous())
        labels.append(str(item.get("gt", "")))
        names.append(item.get("names", []))

    tensor = torch.stack(sequences).contiguous()
    payload = {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "source_config": str(config_path),
        "selection": "first tracklets in deterministic dataset order",
        "sample_count": count,
        "frames": 5,
        "tensor_shape": list(tensor.shape),
        "labels": labels,
        "names": names,
        "lr_sequences": tensor,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    metadata = {
        key: value
        for key, value in payload.items()
        if key != "lr_sequences"
    }
    metadata["cache_sha256"] = sha256_file(output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved input cache: {output_path}")
    print(f"Tensor shape: {tuple(tensor.shape)}")
    print(f"SHA-256: {metadata['cache_sha256']}")

    del tensor, sequences, val_dataset, base_dataset
    gc.collect()

    return metadata


def nested(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def fmt(value, digits=3):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def aggregate_results(
    ordered_models,
    result_files,
    output_dir: Path,
) -> list[dict]:
    rows = []
    for model_key in ordered_models:
        result_path = result_files.get(model_key)
        if result_path is None or not result_path.is_file():
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row = {
            "model_key": model_key,
            "model": payload["model"],
            "model_spec_name": payload.get("model_spec_name"),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_state_source": payload.get("checkpoint_state_source"),
            "precision": payload.get("precision"),
            "params_m": nested(payload, "parameters", "total_millions"),
            "parameter_storage_mib": nested(
                payload,
                "parameters",
                "parameter_storage_mib",
            ),
            "f1_gflops_mean": nested(
                payload,
                "flops",
                "f1",
                "gflops",
                "mean",
            ),
            "f1_gflops_std": nested(
                payload,
                "flops",
                "f1",
                "gflops",
                "std",
            ),
            "f5_batched_gflops_mean": nested(
                payload,
                "flops",
                "f5_batched_tracklet_with_product_fusion",
                "gflops",
                "mean",
            ),
            "f5_batched_gflops_std": nested(
                payload,
                "flops",
                "f5_batched_tracklet_with_product_fusion",
                "gflops",
                "std",
            ),
            "f5_sequential_gflops_mean": nested(
                payload,
                "flops",
                "f5_sequential_tracklet_with_product_fusion",
                "gflops",
                "mean",
            ),
            "f5_sequential_gflops_std": nested(
                payload,
                "flops",
                "f5_sequential_tracklet_with_product_fusion",
                "gflops",
                "std",
            ),
            "f1_latency_ms_mean": nested(
                payload,
                "latency_ms",
                "f1_forward",
                "mean",
            ),
            "f1_latency_ms_std": nested(
                payload,
                "latency_ms",
                "f1_forward",
                "std",
            ),
            "f1_latency_ms_median": nested(
                payload,
                "latency_ms",
                "f1_forward",
                "median",
            ),
            "f1_latency_ms_p95": nested(
                payload,
                "latency_ms",
                "f1_forward",
                "p95",
            ),
            "f5_batched_latency_ms_mean": nested(
                payload,
                "latency_ms",
                "f5_batched_tracklet_forward_plus_product_fusion",
                "mean",
            ),
            "f5_batched_latency_ms_std": nested(
                payload,
                "latency_ms",
                "f5_batched_tracklet_forward_plus_product_fusion",
                "std",
            ),
            "f5_sequential_latency_ms_mean": nested(
                payload,
                "latency_ms",
                "f5_sequential_tracklet_forward_plus_product_fusion",
                "mean",
            ),
            "f5_sequential_latency_ms_std": nested(
                payload,
                "latency_ms",
                "f5_sequential_tracklet_forward_plus_product_fusion",
                "std",
            ),
            "f1_peak_memory_mib": nested(
                payload,
                "peak_memory",
                "f1",
                "peak_allocated_mib",
            ),
            "f5_batched_peak_memory_mib": nested(
                payload,
                "peak_memory",
                "f5_batched_tracklet_forward_plus_product_fusion",
                "peak_allocated_mib",
            ),
            "f5_sequential_peak_memory_mib": nested(
                payload,
                "peak_memory",
                "f5_sequential_tracklet_forward_plus_product_fusion",
                "peak_allocated_mib",
            ),
            "device_name": nested(payload, "environment", "device_name"),
            "torch": nested(payload, "environment", "torch"),
            "cuda_runtime": nested(payload, "environment", "cuda_runtime"),
        }
        rows.append(row)

    csv_path = output_dir / "efficiency_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "model_key",
        "model",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "efficiency_summary.md"
    lines = [
        "# Reviewer 1.1 — computational-efficiency summary",
        "",
        (
            "F=1 latency is a single-frame model forward. The primary F=5 "
            "tracklet latency is deployment-oriented: five consecutive batch-size-one "
            "forwards plus product-rule / sum-log-probability fusion. A separate "
            "batched F=5 latency is also reported because test.py evaluates the five "
            "observations together as a batch of five. Disk I/O, preprocessing, "
            "host-to-device transfer, and final string/CTC decoding are excluded."
        ),
        "",
        (
            "| Model | Params (M) | F1 GFLOPs | F1 latency (ms) | "
            "F5 seq. GFLOPs | F5 seq. latency (ms) | "
            "F5 batched latency (ms) | F1 peak mem (MiB) | "
            "F5 seq. peak mem (MiB) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        f1_latency = (
            f"{fmt(row['f1_latency_ms_mean'])} +/- "
            f"{fmt(row['f1_latency_ms_std'])}"
            if row["f1_latency_ms_mean"] is not None
            else "N/A"
        )
        f5_sequential_latency = (
            f"{fmt(row['f5_sequential_latency_ms_mean'])} +/- "
            f"{fmt(row['f5_sequential_latency_ms_std'])}"
            if row["f5_sequential_latency_ms_mean"] is not None
            else "N/A"
        )
        f5_batched_latency = (
            f"{fmt(row['f5_batched_latency_ms_mean'])} +/- "
            f"{fmt(row['f5_batched_latency_ms_std'])}"
            if row["f5_batched_latency_ms_mean"] is not None
            else "N/A"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    fmt(row["params_m"]),
                    fmt(row["f1_gflops_mean"]),
                    f1_latency,
                    fmt(row["f5_sequential_gflops_mean"]),
                    f5_sequential_latency,
                    f5_batched_latency,
                    fmt(row["f1_peak_memory_mib"], digits=1),
                    fmt(row["f5_sequential_peak_memory_mib"], digits=1),
                ]
            )
            + " |"
        )

    if rows:
        first_result = json.loads(
            result_files[rows[0]["model_key"]].read_text(encoding="utf-8")
        )
        environment = first_result.get("environment", {})
        protocol = first_result.get("protocol", {})
        lines.extend(
            [
                "",
                "## Shared benchmark conditions",
                "",
                f"- Device: {environment.get('device_name')}",
                f"- PyTorch: {environment.get('torch')}",
                f"- CUDA runtime: {environment.get('cuda_runtime')}",
                f"- Precision: {first_result.get('precision')}",
                f"- Warm-up iterations: {protocol.get('warmup_iterations')}",
                f"- Timed iterations: {protocol.get('timed_iterations')}",
                f"- cuDNN benchmark: {protocol.get('cudnn_benchmark')}",
                f"- cuDNN deterministic: {protocol.get('cudnn_deterministic')}",
                "- Input: real LRLPR-26 TEST_3k tracklets resized/normalized by the repository validation wrapper.",
                "- Latency numbers are synchronous end-to-end device execution for the defined tensor operation, not throughput measurements.",
                "",
            ]
        )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the proposed model and principal baselines under one "
            "Reviewer-1 efficiency protocol."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "/home/vwnascimento/doc2025/LMDB-Datasets/"
            "CompetitionDataset_LMDB_TEST_3k"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(EFFICIENCY_DIR / "results"),
    )
    parser.add_argument(
        "--input-cache",
        default=None,
        help=(
            "Optional prepared .pt cache. By default a cache is created under "
            "<output-dir>/benchmark_inputs.pt."
        ),
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--flop-samples", type=int, default=10)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="Physical CUDA device exposed to each fresh worker subprocess.",
    )
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "Override one model checkpoint path without editing the script. "
            "Repeat as needed, e.g. --checkpoint-override svtrv2=/path/to/student_weights. "
            f"Valid model keys: {', '.join(MODEL_SPECS)}"
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument("--rebuild-input-cache", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sample_count < 1:
        parser.error("--sample-count must be >= 1")
    if args.warmup < 1:
        parser.error("--warmup must be >= 1")
    if args.iterations < 2:
        parser.error("--iterations must be >= 2")
    if args.flop_samples < 1:
        parser.error("--flop-samples must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_overrides = {}
    for item in args.checkpoint_override:
        if "=" not in item:
            parser.error(
                "--checkpoint-override must use MODEL=PATH, "
                f"got: {item!r}"
            )
        model_key, raw_path = item.split("=", 1)
        model_key = model_key.strip()
        raw_path = raw_path.strip()
        if model_key not in MODEL_SPECS:
            parser.error(
                "Unknown checkpoint override model "
                f"{model_key!r}; choose from {', '.join(MODEL_SPECS)}"
            )
        if not raw_path:
            parser.error(
                f"Empty checkpoint path for override {model_key!r}"
            )
        checkpoint_overrides[model_key] = Path(raw_path).expanduser()

    input_cache = (
        Path(args.input_cache)
        if args.input_cache
        else output_dir / "benchmark_inputs.pt"
    )

    dataset_path = Path(args.dataset)
    prep_config = Path(MODEL_SPECS["ours"]["config"])

    if not input_cache.is_file() or args.rebuild_input_cache:
        if args.dry_run:
            print(
                f"DRY RUN: would prepare {args.sample_count} real tracklets "
                f"from {dataset_path} -> {input_cache}"
            )
        else:
            prepare_real_input_cache(
                dataset_path=dataset_path,
                output_path=input_cache,
                sample_count=args.sample_count,
                config_path=prep_config,
            )

    worker = EFFICIENCY_DIR / "scripts/benchmark_one.py"
    if not worker.is_file():
        raise SystemExit(f"Worker script not found: {worker}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    manifest = {
        "schema_version": 1,
        "purpose": "Reviewer 1.1 computational-efficiency comparison",
        "models": args.models,
        "dataset": str(dataset_path),
        "input_cache": str(input_cache),
        "sample_count": args.sample_count,
        "gpu": args.gpu,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "flop_samples": args.flop_samples,
        "precision": args.precision,
        "deterministic": bool(args.deterministic),
        "checkpoint_overrides": {
            key: str(value)
            for key, value in checkpoint_overrides.items()
        },
        "runs": [],
    }

    failures = []
    result_files = {}

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        config_path = Path(spec["config"])
        checkpoint_path = checkpoint_overrides.get(
            model_key,
            Path(spec["checkpoints"]),
        )
        result_path = output_dir / f"{model_key}.json"
        result_files[model_key] = result_path

        record = {
            "model_key": model_key,
            "display_name": spec["display_name"],
            "config": str(config_path),
            "checkpoints": str(checkpoint_path),
            "result": str(result_path),
        }

        missing = []
        if not config_path.is_file():
            missing.append(f"config: {config_path}")
        if not checkpoint_path.exists():
            missing.append(f"checkpoint path: {checkpoint_path}")
        if not args.dry_run and not input_cache.is_file():
            missing.append(f"input cache: {input_cache}")

        if missing:
            record["status"] = "missing"
            record["missing"] = missing
            manifest["runs"].append(record)
            failures.append(model_key)
            print(
                f"SKIP {model_key}: " + "; ".join(missing),
                file=sys.stderr,
            )
            if not args.keep_going:
                break
            continue

        if args.skip_existing and result_path.is_file():
            record["status"] = "skipped_existing"
            manifest["runs"].append(record)
            print(f"SKIP {model_key}: existing result {result_path}")
            continue

        command = [
            sys.executable,
            str(worker),
            "--model-name",
            spec["display_name"],
            "--config",
            str(config_path),
            "--checkpoints",
            str(checkpoint_path),
            "--input-cache",
            str(input_cache),
            "--output",
            str(result_path),
            "--precision",
            args.precision,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--flop-samples",
            str(args.flop_samples),
        ]
        if args.deterministic:
            command.append("--deterministic")
        if args.hash_checkpoints:
            command.append("--hash-checkpoint")

        record["command"] = command

        print(
            f"\n=== Efficiency benchmark: {spec['display_name']} ==="
        )
        print("$ " + " ".join(command))

        if args.dry_run:
            record["status"] = "dry_run"
            manifest["runs"].append(record)
            continue

        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
        )
        record["return_code"] = completed.returncode
        record["status"] = "ok" if completed.returncode == 0 else "failed"
        manifest["runs"].append(record)

        if completed.returncode != 0:
            failures.append(model_key)
            if not args.keep_going:
                break

    if not args.dry_run:
        aggregate_results(
            ordered_models=args.models,
            result_files=result_files,
            output_dir=output_dir,
        )

    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nManifest: {manifest_path}")
    if not args.dry_run:
        print(f"CSV:      {output_dir / 'efficiency_summary.csv'}")
        print(f"Markdown: {output_dir / 'efficiency_summary.md'}")

    if failures:
        print(
            "Failed/missing models: " + ", ".join(failures),
            file=sys.stderr,
        )
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
