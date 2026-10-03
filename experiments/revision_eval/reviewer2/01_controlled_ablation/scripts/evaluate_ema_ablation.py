#!/usr/bin/env python3
"""Evaluate a clean matched-epoch EMA-versus-student ablation.

For each full-model seed, this script compares student_weights/last.pth and
ghost_weights/last.pth from the SAME completed training run. Both checkpoints
must report the same epoch. This isolates EMA without confounding the result
with best-checkpoint selection.

Outputs:
  results/ema_ablation/evaluations/seed<seed>/{student,ghost}/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_FRAMES = (1, 3, 5)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") is None:
        raise RuntimeError(f"Checkpoint has no epoch metadata: {path}")
    return int(checkpoint["epoch"])


def resolve_full_run(root: Path, revision_dir: Path, seed: int, seed42_run_dir: Path):
    if seed == 42:
        run_dir = seed42_run_dir
        config = run_dir / "config_snapshot.yaml"
        if not config.is_file():
            config = root / "experiments/revision_eval/submitted_model/config_snapshot.yaml"
        provenance = "historical_submitted_seed42_trajectory"
    else:
        run_dir = revision_dir / "checkpoints" / f"full_seed{seed}"
        config = revision_dir / "configs" / f"full_seed{seed}.yaml"
        provenance = "controlled_full_training"
    return run_dir, config, provenance


def stage_one(source: Path, destination: Path, label: str, epoch: int) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)

    staged = destination / f"{label}_acc_0.0000_ep_{epoch}.pth"
    staged.symlink_to(source.resolve())
    return staged


def main() -> None:
    root = repo_root()
    revision_dir = root / "experiments/revision_eval/reviewer2/01_controlled_ablation"
    shared_runner = root / "experiments/revision_eval/shared/scripts/run_eval_matrix.py"

    parser = argparse.ArgumentParser(
        description="Evaluate same-final-epoch student and EMA checkpoints."
    )
    parser.add_argument(
        "--dataset",
        default="/home/vwnascimento/doc2025/LMDB-Datasets/CompetitionDataset_LMDB_TEST_3k",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--frames", nargs="+", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--fusion", default="bayes")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--seed42-run-dir",
        default=str(root / "experiments/ablations/ce_sfb/ce_sfb_13-05-2026-final"),
        help="Historical full seed-42 training run containing student_weights/last.pth and ghost_weights/last.pth.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset}")
    if not shared_runner.is_file():
        raise SystemExit(f"Shared evaluation runner not found: {shared_runner}")

    seed42_run_dir = Path(args.seed42_run_dir)
    if not seed42_run_dir.is_absolute():
        seed42_run_dir = root / seed42_run_dir

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_root = revision_dir / "results/ema_ablation/evaluations"
    staging_root = revision_dir / "results/ema_ablation/staged_checkpoints"

    manifest = {
        "schema_version": 2,
        "purpose": "same-final-epoch EMA versus student ablation",
        "dataset": str(dataset),
        "fusion": args.fusion,
        "frames": args.frames,
        "seeds": args.seeds,
        "runs": [],
    }

    failures = []

    for seed in args.seeds:
        record = {"seed": seed}
        try:
            run_dir, config, provenance = resolve_full_run(
                root, revision_dir, seed, seed42_run_dir
            )

            student = run_dir / "student_weights/last.pth"
            ghost = run_dir / "ghost_weights/last.pth"

            missing = [str(p) for p in (student, ghost, config) if not p.is_file()]
            if missing:
                raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

            student_epoch = checkpoint_epoch(student)
            ghost_epoch = checkpoint_epoch(ghost)
            if student_epoch != ghost_epoch:
                raise RuntimeError(
                    f"Epoch mismatch for seed {seed}: student={student_epoch}, ghost={ghost_epoch}"
                )

            epoch = student_epoch
            record.update(
                {
                    "epoch": epoch,
                    "run_dir": str(run_dir),
                    "config": str(config),
                    "student_checkpoint": str(student),
                    "ghost_checkpoint": str(ghost),
                    "provenance": provenance,
                }
            )

            print(f"\n=== EMA ablation | seed {seed} | matched final epoch {epoch} ===")
            print(f"Student: {student}")
            print(f"EMA:     {ghost}")

            for source, checkpoint in (("student", student), ("ghost", ghost)):
                stage_dir = staging_root / f"seed{seed}" / source
                staged = stage_one(checkpoint, stage_dir, source, epoch)
                out_dir = output_root / f"seed{seed}" / source

                command = [
                    sys.executable,
                    str(shared_runner),
                    "--config",
                    str(config),
                    "--checkpoints",
                    str(stage_dir),
                    "--split",
                    str(dataset),
                    "--output-dir",
                    str(out_dir),
                    "--frames",
                    *[str(frame) for frame in args.frames],
                    "--fusions",
                    args.fusion,
                ]
                if args.skip_existing:
                    command.append("--skip-existing")
                if args.keep_going:
                    command.append("--keep-going")

                record[f"{source}_staged_checkpoint"] = str(staged)
                record[f"{source}_output_dir"] = str(out_dir)
                record[f"{source}_command"] = command

                print("$ " + " ".join(command))

                if args.dry_run:
                    continue

                completed = subprocess.run(command, cwd=root, env=env, check=False)
                record[f"{source}_return_code"] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{source} evaluation failed for seed {seed} "
                        f"with return code {completed.returncode}"
                    )

            record["status"] = "dry_run" if args.dry_run else "ok"

        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            failures.append(seed)
            print(f"ERROR seed {seed}: {exc}", file=sys.stderr)

        manifest["runs"].append(record)

        if failures and (args.strict or not args.keep_going):
            break

    manifest_path = revision_dir / "results/ema_ablation/ema_evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")

    if failures:
        print(
            "EMA ablation failed for seed(s): " + ", ".join(str(seed) for seed in failures),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
