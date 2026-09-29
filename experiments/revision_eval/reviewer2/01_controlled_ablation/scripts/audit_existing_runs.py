#!/usr/bin/env python3
"""Inventory historical ablation runs and compare them with the submitted protocol.

This script is intentionally read-only. It scans local experiment directories for
config_snapshot.yaml files, extracts the settings relevant to the controlled
reviewer ablation, and records available student/EMA checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


ARCH_DEFAULTS = {
    "use_hcg": True,
    "use_sfb": True,
    "use_rope": True,
    "use_cosine_classifier": True,
    "use_token_masking": True,
    "use_restormer": True,
    "use_pixelshuffle": True,
}

FIELDS = [
    "run_dir",
    "config_path",
    "name",
    "cls_loss",
    "fresh_start",
    "resume",
    "finetune",
    "force_lr",
    "epoch_max",
    "early_stop_patience",
    "ema_warmup_epochs",
    "epoch_max_ema",
    "optimizer",
    "lr",
    "weight_decay",
    "scheduler",
    "train_batch",
    "val_batch",
    "train_aug",
    "use_fda_lr",
    "use_ema_ghost",
    "use_fp16",
    "model_name",
    "feature_dim",
    "cnn_heads",
    "d_model",
    "vit_heads",
    "use_hcg",
    "use_sfb",
    "use_rope",
    "use_cosine_classifier",
    "use_token_masking",
    "use_restormer",
    "use_pixelshuffle",
    "protocol_match",
    "protocol_differences",
    "student_last",
    "student_top_count",
    "ghost_last",
    "ghost_top_count",
]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def nested(data, *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def effective_arch(config):
    args = nested(config, "model_g", "args", default={}) or {}
    result = {}
    for key, default in ARCH_DEFAULTS.items():
        result[key] = args.get(key, default)
    return result


def protocol(config):
    """Settings that should be identical across controlled retraining variants."""
    return {
        "cls_loss": str(config.get("cls_loss", "")).upper(),
        "epoch_max": config.get("epoch_max"),
        "early_stop_patience": config.get("early_stop_patience"),
        "ema_warmup_epochs": config.get("ema_warmup_epochs"),
        "epoch_max_ema": config.get("epoch_max_ema"),
        "optimizer": str(nested(config, "optimizer_sr", "name", default="")).lower(),
        "lr": nested(config, "optimizer_sr", "args", "lr"),
        "weight_decay": nested(config, "optimizer_sr", "args", "weight_decay"),
        "scheduler": str(nested(config, "LRScheduler", "name", default="ReduceLROnPlateau")),
        "train_batch": nested(config, "train_dataset", "batch"),
        "val_batch": nested(config, "val_dataset", "batch"),
        "train_aug": nested(config, "train_dataset", "wrapper", "args", "aug"),
        "use_fda_lr": nested(config, "train_dataset", "wrapper", "args", "use_fda_lr"),
        "use_ema_ghost": config.get("use_ema_ghost"),
        "use_fp16": config.get("use_fp16"),
        "model_name": nested(config, "model_g", "name"),
        "feature_dim": nested(config, "model_g", "args", "feature_dim"),
        "cnn_heads": nested(config, "model_g", "args", "cnn_heads"),
        "d_model": nested(config, "model_g", "args", "d_model"),
        "vit_heads": nested(config, "model_g", "args", "vit_heads"),
    }


def checkpoint_info(run_dir: Path, subdir: str, prefix: str):
    directory = run_dir / subdir
    last = directory / "last.pth"
    top = list(directory.glob("*_acc_*.pth")) if directory.is_dir() else []
    return {
        f"{prefix}_last": str(last) if last.exists() else "",
        f"{prefix}_top_count": len(top),
    }


def make_row(config_path: Path, reference_protocol):
    config = load_yaml(config_path)
    run_dir = config_path.parent
    arch = effective_arch(config)
    current_protocol = protocol(config)

    differences = []
    for key, ref_value in reference_protocol.items():
        value = current_protocol.get(key)
        if value != ref_value:
            differences.append(f"{key}:{value!r}!={ref_value!r}")

    resume = config.get("resume")
    row = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "name": config.get("name"),
        "cls_loss": str(config.get("cls_loss", "")).upper(),
        "fresh_start": resume in (None, "", False),
        "resume": "" if resume is None else str(resume),
        "finetune": config.get("finetune"),
        "force_lr": config.get("force_lr"),
        "epoch_max": config.get("epoch_max"),
        "early_stop_patience": config.get("early_stop_patience"),
        "ema_warmup_epochs": config.get("ema_warmup_epochs"),
        "epoch_max_ema": config.get("epoch_max_ema"),
        "optimizer": current_protocol["optimizer"],
        "lr": current_protocol["lr"],
        "weight_decay": current_protocol["weight_decay"],
        "scheduler": current_protocol["scheduler"],
        "train_batch": current_protocol["train_batch"],
        "val_batch": current_protocol["val_batch"],
        "train_aug": current_protocol["train_aug"],
        "use_fda_lr": current_protocol["use_fda_lr"],
        "use_ema_ghost": config.get("use_ema_ghost"),
        "use_fp16": config.get("use_fp16"),
        "model_name": current_protocol["model_name"],
        "feature_dim": current_protocol["feature_dim"],
        "cnn_heads": current_protocol["cnn_heads"],
        "d_model": current_protocol["d_model"],
        "vit_heads": current_protocol["vit_heads"],
        **arch,
        "protocol_match": not differences,
        "protocol_differences": "; ".join(differences),
    }
    row.update(checkpoint_info(run_dir, "student_weights", "student"))
    row.update(checkpoint_info(run_dir, "ghost_weights", "ghost"))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        default="experiments/ablations",
        help="Directory recursively searched for config_snapshot.yaml files.",
    )
    parser.add_argument(
        "--reference",
        default="experiments/revision_eval/submitted_model/config_snapshot.yaml",
        help="Canonical submitted-model configuration.",
    )
    parser.add_argument(
        "--output",
        default="experiments/revision_eval/reviewer2/01_controlled_ablation/results/existing_run_inventory.csv",
    )
    args = parser.parse_args()

    search_root = Path(args.search_root)
    reference_path = Path(args.reference)
    output_path = Path(args.output)

    if not search_root.exists():
        raise SystemExit(f"Search root does not exist: {search_root}")
    if not reference_path.exists():
        raise SystemExit(f"Reference config does not exist: {reference_path}")

    reference = load_yaml(reference_path)
    reference_protocol = protocol(reference)

    config_paths = sorted(search_root.rglob("config_snapshot.yaml"))
    rows = [make_row(path, reference_protocol) for path in config_paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    exact_protocol = [row for row in rows if row["protocol_match"]]
    fresh_exact = [row for row in exact_protocol if row["fresh_start"]]

    print(f"Scanned {len(config_paths)} historical run(s).")
    print(f"Protocol matches (ignoring architecture toggles/resume): {len(exact_protocol)}")
    print(f"Fresh-start protocol matches: {len(fresh_exact)}")
    print(f"Inventory: {output_path}")

    if exact_protocol:
        print("\nPotentially relevant runs:")
        for row in exact_protocol:
            flags = (
                f"SFB={row['use_sfb']} cosine={row['use_cosine_classifier']} "
                f"restormer={row['use_restormer']} pixelshuffle={row['use_pixelshuffle']} "
                f"EMA={row['use_ema_ghost']}"
            )
            print(
                f"- {row['run_dir']} | fresh={row['fresh_start']} | {flags} | "
                f"student_top={row['student_top_count']} ghost_top={row['ghost_top_count']}"
            )


if __name__ == "__main__":
    main()
