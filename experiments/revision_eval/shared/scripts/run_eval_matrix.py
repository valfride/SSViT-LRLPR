#!/usr/bin/env python3
"""Run a reproducible evaluation matrix through the repository test.py entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VALID_FUSIONS = ("bayes", "average", "logit_average", "majority")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_and_tee(command, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(str(part) for part in command), flush=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return process.wait()


def main() -> None:
    root = repo_root()

    parser = argparse.ArgumentParser(
        description="Run test.py across requested temporal-frame and fusion combinations."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument(
        "--fusions",
        nargs="+",
        choices=VALID_FUSIONS,
        default=list(VALID_FUSIONS),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--test-script", default=str(root / "test.py"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--num-swa", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_manifest = {
        "schema_version": 1,
        "config": str(Path(args.config)),
        "checkpoints": str(Path(args.checkpoints)),
        "dataset": str(Path(args.split)),
        "frames": args.frames,
        "fusions": args.fusions,
        "tta": bool(args.tta),
        "swa": bool(args.swa),
        "num_swa": int(args.num_swa) if args.swa else 1,
        "runs": [],
    }

    failures = []

    for frames in args.frames:
        if frames < 1:
            parser.error("--frames values must be >= 1")

        for fusion in args.fusions:
            stem = f"F{frames}_{fusion}"
            metrics_path = output_dir / f"{stem}.json"
            predictions_path = output_dir / f"{stem}_predictions.csv"
            log_path = output_dir / f"{stem}.txt"

            run_record = {
                "frames": frames,
                "fusion": fusion,
                "metrics_json": str(metrics_path),
                "predictions_csv": str(predictions_path),
                "log": str(log_path),
            }

            if args.skip_existing and metrics_path.exists() and predictions_path.exists():
                print(f"SKIP {stem}: structured outputs already exist.")
                run_record["status"] = "skipped_existing"
                matrix_manifest["runs"].append(run_record)
                continue

            command = [
                args.python,
                args.test_script,
                "--config",
                args.config,
                "--checkpoints",
                args.checkpoints,
                "--split",
                args.split,
                "--mode",
                "val",
                "--in_images",
                str(frames),
                "--fusion",
                fusion,
                "--metrics-json",
                str(metrics_path),
                "--predictions-csv",
                str(predictions_path),
            ]

            if args.tta:
                command.append("--tta")
            if args.swa:
                command.extend(["--swa", "--num_swa", str(args.num_swa)])

            return_code = run_and_tee(command, log_path)
            run_record["return_code"] = return_code
            run_record["status"] = "ok" if return_code == 0 else "failed"
            matrix_manifest["runs"].append(run_record)

            if return_code != 0:
                failures.append(stem)
                if not args.keep_going:
                    break

        if failures and not args.keep_going:
            break

    manifest_path = output_dir / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(matrix_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nMatrix manifest saved to {manifest_path}")

    if failures:
        print("Failed runs: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
