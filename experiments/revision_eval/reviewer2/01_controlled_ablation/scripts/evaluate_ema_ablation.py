#!/usr/bin/env python3
"""Evaluate the student model at the same epoch selected for EMA/ghost inference.

This isolates the effect of EMA without retraining. For each full-model seed:
  1. determine the epoch used by the evaluated EMA/ghost checkpoint;
  2. locate the student checkpoint from that same training epoch;
  3. evaluate the student at F=1/F=3/F=5 with the same fusion protocol.

The existing ghost evaluations remain under:
  results/evaluations/full/seed<seed>/ghost/

This script writes the matched student evaluations under:
  results/evaluations/full/seed<seed>/student/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_FRAMES = (1, 3, 5)
EPOCH_RE = re.compile(r"_ep_(\d+)(?:\D|$)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def epoch_from_filename(path: Path):
    match = EPOCH_RE.search(path.name)
    return int(match.group(1)) if match else None


def epoch_from_checkpoint(path: Path):
    """Read an epoch from checkpoint metadata, returning None when unavailable."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"WARNING: could not inspect checkpoint metadata for {path}: {exc}")
        return None

    if not isinstance(checkpoint, dict):
        return None

    for key in ("epoch", "trained_epoch", "current_epoch"):
        value = checkpoint.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return None


def selected_ghost_checkpoint(
    evaluation_root: Path,
    seed: int,
    fusion: str,
):
    metrics_path = evaluation_root / "full" / f"seed{seed}" / "ghost" / f"F1_{fusion}.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing existing ghost evaluation: {metrics_path}\n"
            "Run evaluate_all.py for the ghost models first."
        )

    payload = read_json(metrics_path)
    checkpoint_files = payload.get("checkpoint_files") or []
    if len(checkpoint_files) != 1:
        raise ValueError(
            f"Expected exactly one selected ghost checkpoint in {metrics_path}, "
            f"found {checkpoint_files!r}"
        )

    return Path(checkpoint_files[0]), metrics_path


def find_student_checkpoint(student_dir: Path, epoch: int) -> Path:
    if not student_dir.is_dir():
        raise FileNotFoundError(f"Student checkpoint directory not found: {student_dir}")

    matches = sorted(student_dir.glob(f"*_ep_{epoch}.pth"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple student checkpoints found for epoch {epoch} in {student_dir}: "
            + ", ".join(str(path) for path in matches)
        )

    # Some runs may retain only last.pth. It is acceptable only if its metadata
    # confirms that it belongs to the requested epoch.
    last_path = student_dir / "last.pth"
    if last_path.is_file() and epoch_from_checkpoint(last_path) == epoch:
        return last_path

    raise FileNotFoundError(
        f"No student checkpoint for epoch {epoch} found in {student_dir}. "
        "Use --seed42-student-checkpoint for the historical submitted run if needed."
    )


def prepare_single_checkpoint_dir(source: Path, destination: Path, epoch: int) -> Path:
    """Create a one-checkpoint directory so test.py cannot select another epoch."""
    destination.mkdir(parents=True, exist_ok=True)

    for existing in destination.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)

    # Keep an accuracy-like filename because test.py discovers *acc_*.pth first.
    if "acc_" in source.name:
        staged_name = source.name
    else:
        staged_name = f"student_acc_0.0000_ep_{epoch}.pth"

    staged = destination / staged_name
    staged.symlink_to(source.resolve())
    return staged


def main() -> None:
    root = repo_root()
    revision_dir = root / "experiments/revision_eval/reviewer2/01_controlled_ablation"
    evaluation_root = revision_dir / "results/evaluations"
    shared_runner = root / "experiments/revision_eval/shared/scripts/run_eval_matrix.py"

    parser = argparse.ArgumentParser(
        description="Evaluate matched-epoch student checkpoints for the EMA ablation."
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
        help="Historical full-model run that produced the submitted seed-42 trajectory.",
    )
    parser.add_argument(
        "--seed42-student-checkpoint",
        default=None,
        help="Optional explicit student checkpoint for seed 42.",
    )
    parser.add_argument(
        "--seed42-epoch",
        type=int,
        default=None,
        help="Optional epoch override if the submitted seed-42 checkpoint lacks epoch metadata.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset}")
    if not shared_runner.is_file():
        raise SystemExit(f"Shared evaluation runner not found: {shared_runner}")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    staging_root = revision_dir / "results/ema_ablation/selected_student_checkpoints"
    manifest = {
        "schema_version": 1,
        "purpose": "matched-epoch EMA versus student ablation",
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
            ghost_checkpoint, ghost_metrics_path = selected_ghost_checkpoint(
                evaluation_root,
                seed,
                args.fusion,
            )
            record["ghost_checkpoint"] = str(ghost_checkpoint)
            record["ghost_metrics_json"] = str(ghost_metrics_path)

            if seed == 42:
                submitted = root / "experiments/revision_eval/submitted_model/last.pth"
                selected_epoch = args.seed42_epoch
                if selected_epoch is None:
                    selected_epoch = epoch_from_checkpoint(submitted)

                if selected_epoch is None:
                    raise RuntimeError(
                        "Could not determine the submitted seed-42 EMA epoch from "
                        f"{submitted}. Re-run with --seed42-epoch EPOCH."
                    )

                config = root / "experiments/revision_eval/submitted_model/config_snapshot.yaml"

                if args.seed42_student_checkpoint:
                    student_checkpoint = Path(args.seed42_student_checkpoint)
                    if not student_checkpoint.is_absolute():
                        student_checkpoint = root / student_checkpoint
                    if not student_checkpoint.is_file():
                        raise FileNotFoundError(
                            f"Explicit seed-42 student checkpoint not found: {student_checkpoint}"
                        )
                    checkpoint_epoch = epoch_from_filename(student_checkpoint)
                    if checkpoint_epoch is None:
                        checkpoint_epoch = epoch_from_checkpoint(student_checkpoint)
                    if checkpoint_epoch is not None and checkpoint_epoch != selected_epoch:
                        raise RuntimeError(
                            f"Seed-42 student checkpoint epoch {checkpoint_epoch} does not match "
                            f"submitted EMA epoch {selected_epoch}: {student_checkpoint}"
                        )
                else:
                    student_dir = Path(args.seed42_run_dir) / "student_weights"
                    student_checkpoint = find_student_checkpoint(student_dir, selected_epoch)

                provenance = "historical_submitted_seed42_trajectory"

            else:
                ghost_epoch = epoch_from_filename(ghost_checkpoint)
                if ghost_epoch is None:
                    # The JSON path may point at a symlink or unusual checkpoint name.
                    candidate = Path(ghost_checkpoint)
                    if candidate.is_file():
                        ghost_epoch = epoch_from_checkpoint(candidate)
                if ghost_epoch is None:
                    raise RuntimeError(
                        f"Could not determine selected ghost epoch for seed {seed}: "
                        f"{ghost_checkpoint}"
                    )

                selected_epoch = ghost_epoch
                config = revision_dir / "configs" / f"full_seed{seed}.yaml"
                student_dir = (
                    revision_dir
                    / "checkpoints"
                    / f"full_seed{seed}"
                    / "student_weights"
                )
                student_checkpoint = find_student_checkpoint(student_dir, selected_epoch)
                provenance = "controlled_full_training"

            if not config.is_file():
                raise FileNotFoundError(f"Config not found: {config}")

            record.update(
                {
                    "selected_epoch": selected_epoch,
                    "student_checkpoint": str(student_checkpoint),
                    "config": str(config),
                    "provenance": provenance,
                }
            )

            stage_dir = staging_root / f"seed{seed}"
            staged_checkpoint = prepare_single_checkpoint_dir(
                student_checkpoint,
                stage_dir,
                selected_epoch,
            )
            record["staged_checkpoint"] = str(staged_checkpoint)

            output_dir = evaluation_root / "full" / f"seed{seed}" / "student"
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
                str(output_dir),
                "--frames",
                *[str(frame) for frame in args.frames],
                "--fusions",
                args.fusion,
            ]
            if args.skip_existing:
                command.append("--skip-existing")
            if args.keep_going:
                command.append("--keep-going")

            record["output_dir"] = str(output_dir)
            record["command"] = command

            print(
                f"\n=== EMA ablation | seed {seed} | epoch {selected_epoch} ==="
            )
            print(f"EMA:     {ghost_checkpoint}")
            print(f"Student: {student_checkpoint}")
            print("$ " + " ".join(command))

            if args.dry_run:
                record["status"] = "dry_run"
            else:
                completed = subprocess.run(command, cwd=root, env=env, check=False)
                record["return_code"] = completed.returncode
                record["status"] = "ok" if completed.returncode == 0 else "failed"
                if completed.returncode != 0:
                    failures.append(seed)

        except Exception as exc:
            record["status"] = "failed_resolution"
            record["error"] = str(exc)
            failures.append(seed)
            print(f"ERROR seed {seed}: {exc}", file=sys.stderr)
            if args.strict:
                manifest["runs"].append(record)
                break

        manifest["runs"].append(record)

        if failures and not args.keep_going:
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
