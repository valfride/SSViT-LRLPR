#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VALID_VARIANTS = ("full", "no_restormer", "no_pixelshuffle", "no_sfb", "linear_head")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VALID_VARIANTS)
    parser.add_argument("seed", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", default="True")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[5]
    revision = root / "experiments/revision_eval/reviewer2/01_controlled_ablation"
    config = revision / "configs" / f"{args.variant}_seed{args.seed}.yaml"
    save = revision / "checkpoints"
    logs = revision / "logs"

    if not config.exists():
        raise SystemExit(
            f"Missing config: {config}\n"
            f"Run {revision / 'scripts/generate_configs.py'} first."
        )

    save.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{args.variant}_seed{args.seed}.log"

    command = [
        sys.executable,
        str(root / "train_gan.py"),
        "--config",
        str(config),
        "--save",
        str(save),
        "--seed",
        str(args.seed),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["DEBUG"] = str(args.debug)

    print("Running:", " ".join(command))
    print("Log:", log_path)

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        code = process.wait()

    raise SystemExit(code)


if __name__ == "__main__":
    main()
