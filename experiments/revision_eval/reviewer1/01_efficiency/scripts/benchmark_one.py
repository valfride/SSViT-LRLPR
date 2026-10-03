#!/usr/bin/env python3
"""Benchmark one OCR model under the Reviewer-1 efficiency protocol.

This worker is intentionally executed in a fresh subprocess for each model so
CUDA allocator state from one architecture does not contaminate another.

The benchmark uses preprocessed real TEST_3k tracklets prepared by
benchmark_efficiency.py. Latency excludes disk I/O and image preprocessing.
F=5 tracklet latency includes the model forward pass for five frames (batched
exactly as test.py does) plus product-rule / sum-log-probability fusion. String
conversion / CTC collapse is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models  # noqa: E402


def checkpoint_accuracy(path: Path) -> float:
    match = re.search(r"(?:^|_)acc_([0-9]+(?:\.[0-9]+)?)", path.stem)
    return float(match.group(1)) if match else float("-inf")


def select_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = sorted(
        path.glob("*acc_*.pth"),
        key=lambda item: (checkpoint_accuracy(item), item.name),
        reverse=True,
    )
    if candidates:
        return candidates[0]

    fallback = path / "last.pth"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"No *acc_*.pth or last.pth checkpoint found under {path}"
    )


def clean_state_dict(state_dict):
    return {
        str(key).replace("module.", ""): value
        for key, value in state_dict.items()
    }


def state_dict_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    for key in ("model_ghost_sd", "model_g_sd", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return clean_state_dict(value), key

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return clean_state_dict(checkpoint), "raw_state_dict"

    raise KeyError(
        "Checkpoint contains no model_ghost_sd, model_g_sd, state_dict, "
        "or raw tensor state dict."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_logits(output):
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, dict):
        raise TypeError(
            f"Expected model output dict, got {type(output).__name__}"
        )
    if "logits" not in output:
        raise KeyError("Model output is missing 'logits'.")
    logits = output["logits"]
    if not torch.is_tensor(logits):
        raise TypeError("Model 'logits' output is not a tensor.")
    return logits


def looks_like_probabilities(tensor: torch.Tensor) -> bool:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return False
    if not torch.isfinite(tensor).all():
        return False
    minimum = float(tensor.detach().min().item())
    maximum = float(tensor.detach().max().item())
    if minimum < -1.0e-6 or maximum > 1.0 + 1.0e-6:
        return False
    sums = tensor.sum(dim=-1)
    return bool(
        torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=1.0e-3,
            rtol=1.0e-3,
        )
    )


def product_fuse(
    logits: torch.Tensor,
    frames: int,
    probability_output: bool,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits [frames, positions, classes], got {tuple(logits.shape)}"
        )
    if logits.shape[0] != frames:
        raise ValueError(
            f"Expected first dimension={frames}, got {logits.shape[0]}"
        )

    sequence = logits.view(
        1,
        frames,
        logits.shape[1],
        logits.shape[2],
    )
    if probability_output:
        log_probabilities = torch.log(sequence.clamp_min(1.0e-8))
    else:
        log_probabilities = F.log_softmax(sequence, dim=-1)
    return log_probabilities.sum(dim=1)


def summarize(values):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, use_fp16: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=use_fp16, dtype=torch.float16)
    return nullcontext()


def measure_latency(fn, pool, warmup: int, iterations: int, device):
    with torch.inference_mode():
        for index in range(warmup):
            fn(pool[index % len(pool)])
        synchronize(device)

        times_ms = []
        for index in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            output = fn(pool[index % len(pool)])
            synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000.0
            times_ms.append(elapsed)
            del output

    return summarize(times_ms)


def measure_peak_memory(fn, input_tensor, device):
    if device.type != "cuda":
        return {
            "peak_allocated_mib": None,
            "incremental_peak_mib": None,
            "baseline_allocated_mib": None,
        }

    synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    with torch.inference_mode():
        output = fn(input_tensor)
        synchronize(device)

    peak = torch.cuda.max_memory_allocated(device)
    del output

    return {
        "peak_allocated_mib": peak / (1024.0 ** 2),
        "incremental_peak_mib": max(0, peak - baseline) / (1024.0 ** 2),
        "baseline_allocated_mib": baseline / (1024.0 ** 2),
    }


def profile_flops(fn, pool, sample_count: int, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:
        return {
            "available": False,
            "error": f"FlopCounterMode unavailable: {exc}",
            "gflops": None,
        }

    values = []
    try:
        count = min(sample_count, len(pool))
        with torch.inference_mode():
            for index in range(count):
                synchronize(device)
                with FlopCounterMode(display=False) as counter:
                    output = fn(pool[index])
                synchronize(device)
                values.append(float(counter.get_total_flops()) / 1.0e9)
                del output
        return {
            "available": True,
            "error": None,
            "gflops": summarize(values),
            "convention": (
                "PyTorch torch.utils.flop_counter; operation coverage follows "
                "the installed PyTorch version"
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "gflops": None,
        }


def load_input_cache(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        sequences = payload
        metadata = {}
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("lr_sequences")):
        sequences = payload["lr_sequences"]
        metadata = {
            key: value
            for key, value in payload.items()
            if key != "lr_sequences"
        }
    else:
        raise TypeError(
            "Input cache must be a tensor or a dictionary containing lr_sequences."
        )

    if sequences.ndim != 5:
        raise ValueError(
            "Expected cached input shape [tracks, frames, channels, H, W], "
            f"got {tuple(sequences.shape)}"
        )
    if sequences.shape[1] < 5:
        raise ValueError(
            f"Input cache must contain at least five frames, got {sequences.shape[1]}"
        )

    return sequences.contiguous(), metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one OCR recognizer for Reviewer 1 efficiency analysis."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--flop-samples", type=int, default=10)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use deterministic cuDNN kernels. The default is performance mode "
            "(cudnn.benchmark=True) for deployment-oriented latency."
        ),
    )
    parser.add_argument("--hash-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.warmup < 1:
        parser.error("--warmup must be >= 1")
    if args.iterations < 2:
        parser.error("--iterations must be >= 2")
    if args.flop_samples < 1:
        parser.error("--flop-samples must be >= 1")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoints)
    input_cache_path = Path(args.input_cache)
    output_path = Path(args.output)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "model_g" not in config:
        raise ValueError(f"Invalid model config: {config_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is not available.")

    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = bool(args.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    if args.precision == "auto":
        use_fp16 = bool(config.get("use_fp16", True)) and device.type == "cuda"
    else:
        use_fp16 = args.precision == "fp16" and device.type == "cuda"

    print(f"\n=== {args.model_name} ===")
    print(f"Config:      {config_path}")
    print(f"Checkpoints: {checkpoint_path}")
    print(f"Input cache: {input_cache_path}")
    print(f"Device:      {device}")
    print(f"FP16:        {use_fp16}")

    model = models.make(config["model_g"])

    selected_checkpoint = select_checkpoint(checkpoint_path)
    checkpoint = torch.load(
        selected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, state_source = state_dict_from_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del checkpoint, state_dict

    if len(missing) > 25 or len(unexpected) > 25:
        raise RuntimeError(
            "Large checkpoint/model mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model = model.to(device)
    model.eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    sequences, cache_metadata = load_input_cache(input_cache_path)
    val_args = config.get("val_dataset", {}).get("wrapper", {}).get("args", {})
    expected_h = int(val_args.get("imgH", sequences.shape[-2]))
    expected_w = int(val_args.get("imgW", sequences.shape[-1]))
    expected_c = int(config["model_g"].get("args", {}).get("in_channels", 3))

    if tuple(sequences.shape[-3:]) != (expected_c, expected_h, expected_w):
        raise ValueError(
            "Cached tensor shape does not match config: "
            f"cache={tuple(sequences.shape[-3:])}, "
            f"config={(expected_c, expected_h, expected_w)}"
        )

    sample_count = int(sequences.shape[0])

    f1_pool = []
    f5_pool = []
    for index in range(sample_count):
        f1 = sequences[index, 0].unsqueeze(0).to(device)
        f5 = sequences[index, :5].to(device)
        if device.type == "cuda":
            f1 = f1.contiguous(memory_format=torch.channels_last)
            f5 = f5.contiguous(memory_format=torch.channels_last)
        else:
            f1 = f1.contiguous()
            f5 = f5.contiguous()
        f1_pool.append(f1)
        f5_pool.append(f5)

    def forward_logits(x):
        with autocast_context(device, use_fp16):
            output = model(x, epoch=100)
            return extract_logits(output)

    with torch.inference_mode():
        probe_logits = forward_logits(f5_pool[0])
        synchronize(device)
        probability_output = looks_like_probabilities(probe_logits)
        probe_shape = list(probe_logits.shape)
        del probe_logits

    def f1_call(x):
        return forward_logits(x)

    def f5_call(x):
        logits = forward_logits(x)
        return product_fuse(
            logits,
            frames=5,
            probability_output=probability_output,
        )

    print(
        "Output interpretation: "
        + ("probabilities" if probability_output else "logits")
    )
    print(f"Probe output shape: {probe_shape}")

    f1_latency = measure_latency(
        f1_call,
        f1_pool,
        args.warmup,
        args.iterations,
        device,
    )
    f5_latency = measure_latency(
        f5_call,
        f5_pool,
        args.warmup,
        args.iterations,
        device,
    )

    f1_flops = profile_flops(
        f1_call,
        f1_pool,
        args.flop_samples,
        device,
    )
    f5_flops = profile_flops(
        f5_call,
        f5_pool,
        args.flop_samples,
        device,
    )

    # Free the latency pools before peak-memory measurement. Peak-memory runs use
    # one representative real tracklet, with model parameters and the active
    # input included in the reported total allocated memory.
    del f1_pool, f5_pool
    if device.type == "cuda":
        torch.cuda.empty_cache()

    first_sequence = sequences[0]
    f1_memory_input = first_sequence[0].unsqueeze(0).to(device)
    if device.type == "cuda":
        f1_memory_input = f1_memory_input.contiguous(
            memory_format=torch.channels_last
        )
    f1_memory = measure_peak_memory(f1_call, f1_memory_input, device)
    del f1_memory_input

    if device.type == "cuda":
        torch.cuda.empty_cache()

    f5_memory_input = first_sequence[:5].to(device)
    if device.type == "cuda":
        f5_memory_input = f5_memory_input.contiguous(
            memory_format=torch.channels_last
        )
    f5_memory = measure_peak_memory(f5_call, f5_memory_input, device)
    del f5_memory_input

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        device_properties = torch.cuda.get_device_properties(device)
        total_gpu_memory_gib = device_properties.total_memory / (1024.0 ** 3)
        cudnn_version = torch.backends.cudnn.version()
    else:
        device_name = platform.processor() or "CPU"
        total_gpu_memory_gib = None
        cudnn_version = None

    result = {
        "schema_version": 1,
        "model": args.model_name,
        "config": str(config_path),
        "model_spec_name": config["model_g"]["name"],
        "checkpoint": str(selected_checkpoint),
        "checkpoint_state_source": state_source,
        "checkpoint_sha256": (
            sha256_file(selected_checkpoint) if args.hash_checkpoint else None
        ),
        "checkpoint_load": {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        },
        "input_cache": str(input_cache_path),
        "input_cache_metadata": cache_metadata,
        "input_shape_per_frame": [expected_c, expected_h, expected_w],
        "number_of_real_tracklets_in_pool": sample_count,
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "total_millions": total_params / 1.0e6,
            "parameter_storage_mib": parameter_bytes / (1024.0 ** 2),
        },
        "precision": "fp16" if use_fp16 else "fp32",
        "output_interpretation": (
            "probabilities" if probability_output else "logits"
        ),
        "probe_output_shape_f5_flat_batch": probe_shape,
        "flops": {
            "f1": f1_flops,
            "f5_tracklet_with_product_fusion": f5_flops,
        },
        "latency_ms": {
            "f1_forward": f1_latency,
            "f5_tracklet_forward_plus_product_fusion": f5_latency,
        },
        "peak_memory": {
            "f1": f1_memory,
            "f5_tracklet_forward_plus_product_fusion": f5_memory,
        },
        "protocol": {
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "flop_samples": args.flop_samples,
            "batching": (
                "F=1 uses batch 1; F=5 flattens the five observations into "
                "a single batch of 5, matching test.py"
            ),
            "latency_timer": (
                "time.perf_counter with device synchronization immediately "
                "before and after each inference"
            ),
            "latency_excludes": [
                "disk I/O",
                "LMDB access",
                "image resize/normalization",
                "host-to-device transfer",
                "final Python string conversion",
                "CTC collapse/string decoding",
            ],
            "f5_includes": (
                "network forward for five frames plus product-rule / "
                "sum-log-probability tensor fusion"
            ),
            "channels_last": device.type == "cuda",
            "deterministic": bool(args.deterministic),
            "cudnn_benchmark": (
                bool(torch.backends.cudnn.benchmark)
                if device.type == "cuda"
                else None
            ),
            "cudnn_deterministic": (
                bool(torch.backends.cudnn.deterministic)
                if device.type == "cuda"
                else None
            ),
            "matmul_allow_tf32": (
                bool(torch.backends.cuda.matmul.allow_tf32)
                if device.type == "cuda"
                else None
            ),
            "cudnn_allow_tf32": (
                bool(torch.backends.cudnn.allow_tf32)
                if device.type == "cuda"
                else None
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": cudnn_version,
            "device": str(device),
            "device_name": device_name,
            "gpu_total_memory_gib": total_gpu_memory_gib,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nResult")
    print(f"  Params: {total_params / 1e6:.3f} M")
    if f1_flops.get("gflops"):
        print(
            "  F1 FLOPs: "
            f"{f1_flops['gflops']['mean']:.3f} GFLOPs"
        )
    print(
        "  F1 latency: "
        f"{f1_latency['mean']:.3f} +/- {f1_latency['std']:.3f} ms"
    )
    print(
        "  F5 latency: "
        f"{f5_latency['mean']:.3f} +/- {f5_latency['std']:.3f} ms"
    )
    if f5_memory["peak_allocated_mib"] is not None:
        print(
            "  F5 peak memory: "
            f"{f5_memory['peak_allocated_mib']:.1f} MiB"
        )
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
