#!/usr/bin/env python3
"""Smoke test for the OJ-ITS SVTRv2-AR revision baseline."""

import argparse
from pathlib import Path
import sys

# Allow this script to be executed directly from its nested revision folder.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

import models
from models.svtrv2.svtrv2_ar_bridge import SVTRv2ARLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="baselines_configs/SVTRV2_AR_BASELINE.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Expected controlled-training protocol for the additional revision
    # baseline. These assertions prevent accidental drift from the baseline
    # setup documented in the manuscript/revision notes.
    assert int(config["epoch_max"]) == 100, config["epoch_max"]
    assert int(config["early_stop_patience"]) == 100, config["early_stop_patience"]
    assert float(config["optimizer_sr"]["args"]["lr"]) == 1.0e-4
    assert float(config["optimizer_sr"]["args"]["weight_decay"]) == 0.05
    assert config["LRScheduler"]["name"] == "OneCycleLR"
    assert float(config["LRScheduler"]["warmup_epoch"]) == 10.0
    assert int(config["train_dataset"]["batch"]) == 64
    assert int(config["train_dataset"]["wrapper"]["args"]["imgH"]) == 32
    assert int(config["train_dataset"]["wrapper"]["args"]["imgW"]) == 96
    assert int(config["train_dataset"]["wrapper"]["args"]["in_images"]) == 1
    assert int(config["val_dataset"]["wrapper"]["args"]["in_images"]) == 5
    assert config.get("resume") is None
    assert config.get("use_ema_ghost", False) is False

    device = torch.device(args.device)
    model = models.make(config["model_g"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Config: {config_path}")
    print(f"Model: {config['model_g']['name']}")
    print(f"Parameters: {total_params / 1e6:.3f} M")
    print(
        "Training protocol: "
        f"AdamW lr={config['optimizer_sr']['args']['lr']} "
        f"wd={config['optimizer_sr']['args']['weight_decay']} | "
        f"OneCycleLR warmup={config['LRScheduler']['warmup_epoch']} ep | "
        f"max={config['epoch_max']} ep | "
        f"early-stop={config['early_stop_patience']} | "
        f"batch={config['train_dataset']['batch']}"
    )

    # Teacher-forced training path: BOS + 7 characters + EOS.
    train_images = torch.randn(2, 3, 32, 96, device=device)
    targets = torch.tensor(
        [
            [37, 11, 12, 13, 1, 2, 3, 4, 0],
            [37, 14, 15, 16, 5, 17, 6, 7, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    model.train()
    train_output = model(train_images, tgt=targets)
    train_logits = train_output["logits"]
    loss = SVTRv2ARLoss()(train_output, targets)
    print(f"Train logits: {tuple(train_logits.shape)}")
    print(f"Train loss finite: {bool(torch.isfinite(loss).item())}")

    # BJP inference path for the paper's F=1/F=3/F=5 settings.
    model.eval()
    with torch.no_grad():
        for frames in (1, 3, 5):
            images = torch.randn(frames, 3, 32, 96, device=device)
            fused = model.bjp_decode(
                images,
                batch_size=1,
                frames=frames,
            )
            print(
                f"F={frames} BJP output: {tuple(fused.shape)} "
                f"(finite={bool(torch.isfinite(fused).all().item())})"
            )

    print("SVTRv2-AR smoke test: PASS")


if __name__ == "__main__":
    main()
