#!/usr/bin/env python3
"""Run the same evaluation twice and verify byte-identical predictions.

This is a focused regression check for deterministic inference. It invokes the
shared evaluation runner twice with the same config/checkpoint/dataset settings,
then compares the generated prediction CSVs and the reported metric payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    root = repo_root()
    runner = root / "experiments/revision_eval/shared/scripts/run_eval_matrix.py"

    parser = argparse.ArgumentParser(
        description="Verify that repeated inference produces identical outputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--fusion", default="bayes")
    parser.add_argument("--gpu", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(
            root
            / "experiments/revision_eval/shared/results/determinism_check"
        ),
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep both run directories after the comparison.",
    )
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be >= 1")

    output_root = Path(args.output_dir)
    run_a = output_root / "run_a"
    run_b = output_root / "run_b"

    if output_root.exists():
        shutil.rmtree(output_root)
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    env = None
    if args.gpu is not None:
        import os

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base = [
        sys.executable,
        str(runner),
        "--config",
        args.config,
        "--checkpoints",
        args.checkpoints,
        "--split",
        args.split,
        "--frames",
        str(args.frames),
        "--fusions",
        args.fusion,
    ]

    for label, destination in (("A", run_a), ("B", run_b)):
        command = [*base, "--output-dir", str(destination)]
        print(f"\n=== Determinism run {label} ===")
        print("$ " + " ".join(command))
        completed = subprocess.run(command, cwd=root, env=env, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"Determinism run {label} failed with return code "
                f"{completed.returncode}."
            )

    stem = f"F{args.frames}_{args.fusion}"
    pred_a = run_a / f"{stem}_predictions.csv"
    pred_b = run_b / f"{stem}_predictions.csv"
    metrics_a = run_a / f"{stem}.json"
    metrics_b = run_b / f"{stem}.json"

    for path in (pred_a, pred_b, metrics_a, metrics_b):
        if not path.is_file():
            raise SystemExit(f"Expected output missing: {path}")

    pred_hash_a = sha256(pred_a)
    pred_hash_b = sha256(pred_b)
    predictions_identical = pred_hash_a == pred_hash_b

    payload_a = read_json(metrics_a)
    payload_b = read_json(metrics_b)

    metrics_identical = payload_a.get("metrics") == payload_b.get("metrics")
    reproducibility_identical = (
        payload_a.get("reproducibility") == payload_b.get("reproducibility")
    )

    manifest = {
        "schema_version": 1,
        "config": str(Path(args.config)),
        "checkpoints": str(Path(args.checkpoints)),
        "dataset": str(Path(args.split)),
        "frames": args.frames,
        "fusion": args.fusion,
        "prediction_sha256_run_a": pred_hash_a,
        "prediction_sha256_run_b": pred_hash_b,
        "predictions_byte_identical": predictions_identical,
        "metrics_identical": metrics_identical,
        "reproducibility_metadata_identical": reproducibility_identical,
        "run_a_reproducibility": payload_a.get("reproducibility"),
        "run_b_reproducibility": payload_b.get("reproducibility"),
    }

    manifest_path = output_root / "determinism_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\n=== Determinism result ===")
    print(f"Predictions byte-identical: {predictions_identical}")
    print(f"Metrics identical:          {metrics_identical}")
    print(f"Run A SHA256: {pred_hash_a}")
    print(f"Run B SHA256: {pred_hash_b}")
    print(f"Manifest: {manifest_path}")

    success = predictions_identical and metrics_identical and reproducibility_identical

    if success and not args.keep_output:
        for directory in (run_a, run_b):
            shutil.rmtree(directory)

    if not success:
        raise SystemExit(
            "Determinism check FAILED. Keep the two run directories and inspect "
            "their prediction/metric differences."
        )

    print("✅ Determinism check PASSED.")


if __name__ == "__main__":
    main()
