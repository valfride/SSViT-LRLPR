#!/usr/bin/env python3
"""Evaluate all controlled-ablation checkpoints at F=1/F=3/F=5.

The seed-42 full model is the frozen submitted checkpoint by default. Other
variant/seed combinations are read from the controlled-ablation training tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_VARIANTS = ("full", "no_restormer", "no_pixelshuffle", "no_sfb", "linear_head")
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_FRAMES = (1, 3, 5)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def resolve_run(
    root: Path,
    revision_dir: Path,
    variant: str,
    seed: int,
    weight_source: str,
) -> tuple[Path, Path, str]:
    if variant == "full" and seed == 42 and weight_source == "ghost":
        return (
            root / "experiments/revision_eval/submitted_model/config_snapshot.yaml",
            root / "experiments/revision_eval/submitted_model",
            "frozen_submitted_seed42",
        )

    config = revision_dir / "configs" / f"{variant}_seed{seed}.yaml"
    run_dir = revision_dir / "checkpoints" / f"{variant}_seed{seed}"
    checkpoints = run_dir / f"{weight_source}_weights"
    return config, checkpoints, "controlled_retraining"


def has_checkpoint(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return (directory / "last.pth").exists() or any(directory.glob("*acc_*.pth"))


def main() -> None:
    root = repo_root()
    revision_dir = root / "experiments/revision_eval/reviewer2/01_controlled_ablation"
    shared_runner = root / "experiments/revision_eval/shared/scripts/run_eval_matrix.py"

    parser = argparse.ArgumentParser(
        description="Evaluate controlled ablations at F=1/F=3/F=5."
    )
    parser.add_argument(
        "--dataset",
        default="/home/vwnascimento/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--frames", nargs="+", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--fusion", default="bayes")
    parser.add_argument(
        "--weight-source",
        choices=("ghost", "student"),
        default="ghost",
        help="Evaluate EMA ghost weights (default) or student weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(revision_dir / "results" / "evaluations"),
    )
    parser.add_argument("--gpu", default=None, help="Physical GPU index exposed to test.py.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on a missing requested config/checkpoint instead of skipping it.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset}")
    if not shared_runner.exists():
        raise SystemExit(f"Shared evaluation runner not found: {shared_runner}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "variants": args.variants,
        "seeds": args.seeds,
        "frames": args.frames,
        "fusion": args.fusion,
        "weight_source": args.weight_source,
        "gpu": args.gpu,
        "runs": [],
    }
    failures = []
    skipped = []

    for seed in args.seeds:
        for variant in args.variants:
            config, checkpoints, provenance = resolve_run(
                root,
                revision_dir,
                variant,
                seed,
                args.weight_source,
            )
            run_output = output_root / variant / f"seed{seed}" / args.weight_source
            record = {
                "variant": variant,
                "seed": seed,
                "weight_source": args.weight_source,
                "provenance": provenance,
                "config": str(config),
                "checkpoints": str(checkpoints),
                "output_dir": str(run_output),
            }

            missing = []
            if not config.is_file():
                missing.append(f"config: {config}")
            if not has_checkpoint(checkpoints):
                missing.append(f"checkpoint directory: {checkpoints}")

            if missing:
                record["status"] = "missing"
                record["missing"] = missing
                manifest["runs"].append(record)
                skipped.append(f"{variant}/seed{seed}")
                message = "Missing " + "; ".join(missing)
                if args.strict:
                    raise SystemExit(message)
                print(f"SKIP {variant}/seed{seed}: {message}")
                continue

            command = [
                sys.executable,
                str(shared_runner),
                "--config",
                str(config),
                "--checkpoints",
                str(checkpoints),
                "--split",
                str(dataset),
                "--output-dir",
                str(run_output),
                "--frames",
                *[str(value) for value in args.frames],
                "--fusions",
                args.fusion,
            ]
            if args.skip_existing:
                command.append("--skip-existing")
            if args.keep_going:
                command.append("--keep-going")

            print(
                f"\n=== {variant} | seed {seed} | {args.weight_source} | "
                f"{provenance} ==="
            )
            print("$ " + " ".join(command))

            if args.dry_run:
                record["status"] = "dry_run"
                record["command"] = command
                manifest["runs"].append(record)
                continue

            completed = subprocess.run(command, cwd=root, env=env, check=False)
            record["return_code"] = completed.returncode
            record["status"] = "ok" if completed.returncode == 0 else "failed"
            manifest["runs"].append(record)

            if completed.returncode != 0:
                failures.append(f"{variant}/seed{seed}")
                if not args.keep_going:
                    break

        if failures and not args.keep_going:
            break

    manifest_path = output_root / f"evaluation_manifest_{args.weight_source}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nManifest: {manifest_path}")
    if skipped:
        print("Skipped missing runs: " + ", ".join(skipped))
    if failures:
        print("Failed evaluations: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
