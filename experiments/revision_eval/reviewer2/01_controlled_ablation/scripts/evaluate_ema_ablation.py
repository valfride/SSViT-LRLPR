#!/usr/bin/env python3
"""Evaluate EMA-versus-student ablations under two checkpoint-selection protocols.

Selection modes
---------------
last:
    Compare student_weights/last.pth and ghost_weights/last.pth from the same
    completed training run. The checkpoints must report the same epoch. This is
    the strict matched-final-epoch comparison and isolates the averaging effect.

best:
    Independently select the highest-validation-accuracy retained checkpoint
    from student_weights/ and ghost_weights/. This mirrors practical
    validation-based model selection and allows the best student and best EMA
    checkpoints to come from different epochs.

Outputs
-------
last:
    results/ema_ablation/evaluations/seed<seed>/{student,ghost}/

best:
    results/ema_ablation/best/evaluations/seed<seed>/{student,ghost}/
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
BEST_RE = re.compile(
    r"^(?P<label>student|ghost)_acc_(?P<acc>[0-9]+(?:\.[0-9]+)?)_ep_(?P<epoch>\d+)\.pth$"
)


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


def select_best_checkpoint(directory: Path, label: str) -> tuple[Path, float, int]:
    """Select the retained checkpoint with highest validation accuracy.

    The training code names retained top checkpoints as:
      student_acc_<accuracy>_ep_<epoch>.pth
      ghost_acc_<accuracy>_ep_<epoch>.pth

    Ties in validation accuracy are broken deterministically by later epoch.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    candidates = []
    for path in directory.glob(f"{label}_acc_*_ep_*.pth"):
        match = BEST_RE.match(path.name)
        if match is None or match.group("label") != label:
            continue
        candidates.append(
            (
                float(match.group("acc")),
                int(match.group("epoch")),
                path,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"No retained best-{label} checkpoints found in {directory}. "
            f"Expected files like {label}_acc_0.7500_ep_42.pth."
        )

    accuracy, epoch, path = max(candidates, key=lambda item: (item[0], item[1]))
    metadata_epoch = checkpoint_epoch(path)
    if metadata_epoch != epoch:
        raise RuntimeError(
            f"Epoch mismatch between filename and checkpoint metadata for {path}: "
            f"filename={epoch}, metadata={metadata_epoch}"
        )
    return path, accuracy, epoch


def stage_one(source: Path, destination: Path, label: str, epoch: int) -> Path:
    """Stage exactly one checkpoint so the shared evaluator cannot pick another."""
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)

    if "_acc_" in source.name and "_ep_" in source.name:
        staged_name = source.name
    else:
        staged_name = f"{label}_acc_0.0000_ep_{epoch}.pth"

    staged = destination / staged_name
    staged.symlink_to(source.resolve())
    return staged


def selection_paths(revision_dir: Path, selection: str):
    base = revision_dir / "results/ema_ablation"
    if selection == "last":
        return (
            base / "evaluations",
            base / "staged_checkpoints",
            base / "ema_evaluation_manifest.json",
        )
    return (
        base / "best/evaluations",
        base / "best/staged_checkpoints",
        base / "best/ema_evaluation_manifest.json",
    )


def main() -> None:
    root = repo_root()
    revision_dir = root / "experiments/revision_eval/reviewer2/01_controlled_ablation"
    shared_runner = root / "experiments/revision_eval/shared/scripts/run_eval_matrix.py"

    parser = argparse.ArgumentParser(
        description="Evaluate EMA versus student with matched-last or best-vs-best selection."
    )
    parser.add_argument(
        "--selection",
        choices=("last", "best"),
        default="last",
        help=(
            "'last' compares same-final-epoch checkpoints; 'best' independently "
            "selects the highest-validation retained student and EMA checkpoints."
        ),
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
        help=(
            "Historical full seed-42 training run containing student_weights/ "
            "and ghost_weights/."
        ),
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

    output_root, staging_root, manifest_path = selection_paths(
        revision_dir, args.selection
    )

    purpose = (
        "same-final-epoch EMA versus student ablation"
        if args.selection == "last"
        else "independently validation-selected best EMA versus best student ablation"
    )
    manifest = {
        "schema_version": 3,
        "purpose": purpose,
        "selection": args.selection,
        "dataset": str(dataset),
        "fusion": args.fusion,
        "frames": args.frames,
        "seeds": args.seeds,
        "runs": [],
    }

    failures = []

    for seed in args.seeds:
        record = {"seed": seed, "selection": args.selection}
        try:
            run_dir, config, provenance = resolve_full_run(
                root, revision_dir, seed, seed42_run_dir
            )
            if not config.is_file():
                raise FileNotFoundError(f"Config not found: {config}")

            student_dir = run_dir / "student_weights"
            ghost_dir = run_dir / "ghost_weights"

            if args.selection == "last":
                student = student_dir / "last.pth"
                ghost = ghost_dir / "last.pth"
                missing = [str(p) for p in (student, ghost) if not p.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "Missing required final checkpoint(s): " + ", ".join(missing)
                    )

                student_epoch = checkpoint_epoch(student)
                ghost_epoch = checkpoint_epoch(ghost)
                if student_epoch != ghost_epoch:
                    raise RuntimeError(
                        f"Epoch mismatch for seed {seed}: "
                        f"student={student_epoch}, ghost={ghost_epoch}"
                    )
                student_acc = None
                ghost_acc = None
                selection_description = f"matched final epoch {student_epoch}"
            else:
                student, student_acc, student_epoch = select_best_checkpoint(
                    student_dir, "student"
                )
                ghost, ghost_acc, ghost_epoch = select_best_checkpoint(
                    ghost_dir, "ghost"
                )
                selection_description = (
                    f"best-vs-best: student val={student_acc:.4f} ep={student_epoch}; "
                    f"EMA val={ghost_acc:.4f} ep={ghost_epoch}"
                )

            record.update(
                {
                    "run_dir": str(run_dir),
                    "config": str(config),
                    "provenance": provenance,
                    "student_checkpoint": str(student),
                    "student_epoch": student_epoch,
                    "student_validation_accuracy": student_acc,
                    "ghost_checkpoint": str(ghost),
                    "ghost_epoch": ghost_epoch,
                    "ghost_validation_accuracy": ghost_acc,
                }
            )

            print(
                f"\n=== EMA ablation | seed {seed} | {args.selection} | "
                f"{selection_description} ==="
            )
            print(f"Student: {student}")
            print(f"EMA:     {ghost}")

            for source, checkpoint, epoch in (
                ("student", student, student_epoch),
                ("ghost", ghost, ghost_epoch),
            ):
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

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")

    if failures:
        print(
            "EMA ablation failed for seed(s): "
            + ", ".join(str(seed) for seed in failures),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
