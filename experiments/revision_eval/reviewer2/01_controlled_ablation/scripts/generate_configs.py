#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


VARIANTS = {
    "full": {},
    "no_restormer": {"use_restormer": False},
    "no_pixelshuffle": {"use_pixelshuffle": False},
    "no_sfb": {"use_sfb": False},
    "linear_head": {"use_cosine_classifier": False},
}


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def make_base(reference):
    config = copy.deepcopy(reference)
    config["resume"] = None
    config["finetune"] = False

    model_args = config.setdefault("model_g", {}).setdefault("args", {})
    model_args.setdefault("use_hcg", False)
    model_args.setdefault("use_sfb", True)
    model_args.setdefault("use_rope", False)
    model_args.setdefault("use_cosine_classifier", True)
    model_args.setdefault("use_token_masking", True)
    model_args.setdefault("use_restormer", True)
    model_args.setdefault("use_pixelshuffle", True)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default="experiments/revision_eval/submitted_model/config_snapshot.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/revision_eval/reviewer2/01_controlled_ablation/configs",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    args = parser.parse_args()

    reference = load_yaml(Path(args.reference))
    base = make_base(reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for seed in args.seeds:
        for variant, changes in VARIANTS.items():
            config = copy.deepcopy(base)
            config["seed"] = int(seed)
            config["name"] = f"VSR_REVISION_{variant.upper()}_SEED{seed}"

            model_args = config["model_g"]["args"]
            for key, value in changes.items():
                model_args[key] = value

            path = output_dir / f"{variant}_seed{seed}.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            written.append(path)

    print(f"Generated {len(written)} controlled-ablation config(s):")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
