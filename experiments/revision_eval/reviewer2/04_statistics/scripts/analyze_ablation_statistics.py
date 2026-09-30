#!/usr/bin/env python3
"""Reviewer-2 statistics for controlled ablation evaluations.

Produces:
  * mean +/- sample standard deviation across seeds;
  * paired per-seed exact McNemar tests;
  * paired per-seed bootstrap 95% confidence intervals;
  * hierarchical paired-bootstrap 95% confidence intervals across seeds.

For ablation comparisons, a positive delta means the full model is more accurate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


DEFAULT_VARIANTS = ("full", "no_restormer", "no_pixelshuffle", "no_sfb", "linear_head")
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_FRAMES = (1, 3, 5)

METRICS = (
    "sequence_accuracy_percent",
    "character_accuracy_percent",
    "cer_percent",
    "partial_6_percent",
    "partial_5_percent",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_predictions(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            track = row["track"]
            if track in rows:
                raise ValueError(f"Duplicate track {track!r} in {path}")
            rows[track] = row
    return rows


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def exact_mcnemar_pvalue(full_only: int, ablation_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value, evaluated stably."""
    n = int(full_only + ablation_only)
    if n == 0:
        return 1.0

    tail = min(int(full_only), int(ablation_only))
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2.0)
        for k in range(tail + 1)
    ]
    maximum = max(logs)
    lower_tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)
    return min(1.0, 2.0 * lower_tail)


def bootstrap_delta_ci(
    full_only: int,
    ablation_only: int,
    same: int,
    samples: int,
    rng: np.random.Generator,
):
    """Paired non-parametric bootstrap CI for exact-match delta in percentage points."""
    total = full_only + ablation_only + same
    if total <= 0:
        return float("nan"), float("nan")

    probabilities = np.array([full_only, ablation_only, same], dtype=float) / total
    counts = rng.multinomial(total, probabilities, size=samples)
    deltas = 100.0 * (counts[:, 0] - counts[:, 1]) / total
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def hierarchical_bootstrap_ci(
    discordance_by_seed,
    samples: int,
    rng: np.random.Generator,
):
    """Two-stage bootstrap: resample tracks within seed, then resample seeds."""
    seed_keys = sorted(discordance_by_seed)
    if not seed_keys:
        return float("nan"), float("nan")

    seed_bootstrap = []
    for seed in seed_keys:
        full_only, ablation_only, same = discordance_by_seed[seed]
        total = full_only + ablation_only + same
        probabilities = np.array([full_only, ablation_only, same], dtype=float) / total
        counts = rng.multinomial(total, probabilities, size=samples)
        seed_bootstrap.append(100.0 * (counts[:, 0] - counts[:, 1]) / total)

    boot = np.stack(seed_bootstrap, axis=0)
    n_seeds = boot.shape[0]
    choices = rng.integers(0, n_seeds, size=(samples, n_seeds))
    sample_index = np.arange(samples)

    hierarchical = np.zeros(samples, dtype=float)
    for slot in range(n_seeds):
        hierarchical += boot[choices[:, slot], sample_index]
    hierarchical /= n_seeds

    low, high = np.percentile(hierarchical, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(rows):
    """Holm-Bonferroni correction within each seed/frame family."""
    families = {}
    for index, row in enumerate(rows):
        key = (row["seed"], row["frames"])
        families.setdefault(key, []).append((index, float(row["mcnemar_exact_p"])))

    for family in families.values():
        ordered = sorted(family, key=lambda item: item[1])
        m = len(ordered)
        running = 0.0
        adjusted = {}
        for rank, (index, pvalue) in enumerate(ordered):
            candidate = min(1.0, (m - rank) * pvalue)
            running = max(running, candidate)
            adjusted[index] = running
        for index, value in adjusted.items():
            rows[index]["mcnemar_holm_p"] = value


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = repo_root()
    default_eval = (
        root
        / "experiments/revision_eval/reviewer2/01_controlled_ablation/results/evaluations"
    )
    default_output = root / "experiments/revision_eval/reviewer2/04_statistics/results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", default=str(default_eval))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--frames", nargs="+", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--fusion", default="bayes")
    parser.add_argument("--weight-source", default="ghost")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260930)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing evaluations instead of analyzing available results.",
    )
    args = parser.parse_args()

    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")

    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.random_seed)

    run_metrics = {}
    missing = []

    for variant in args.variants:
        for seed in args.seeds:
            for frames in args.frames:
                stem = f"F{frames}_{args.fusion}"
                metrics_path = (
                    evaluation_root
                    / variant
                    / f"seed{seed}"
                    / args.weight_source
                    / f"{stem}.json"
                )
                if not metrics_path.is_file():
                    missing.append(str(metrics_path))
                    continue

                payload = read_json(metrics_path)
                metrics = payload.get("metrics", {})
                run_metrics[(variant, seed, frames)] = metrics

    if missing and args.strict:
        raise SystemExit("Missing evaluation files:\n" + "\n".join(missing))

    seed_summary = []
    for variant in args.variants:
        for frames in args.frames:
            available = [
                (seed, run_metrics[(variant, seed, frames)])
                for seed in args.seeds
                if (variant, seed, frames) in run_metrics
            ]
            if not available:
                continue

            row = {
                "variant": variant,
                "frames": frames,
                "n_seeds": len(available),
                "seeds": ",".join(str(seed) for seed, _ in available),
            }
            for metric in METRICS:
                values = [float(metrics[metric]) for _, metrics in available]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            seed_summary.append(row)

    summary_fields = ["variant", "frames", "n_seeds", "seeds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "ablation_seed_summary.csv", seed_summary, summary_fields)

    per_seed_tests = []
    hierarchical_rows = []
    ablations = [variant for variant in args.variants if variant != "full"]

    for ablation in ablations:
        for frames in args.frames:
            discordance_by_seed = {}
            seed_deltas = []

            for seed in args.seeds:
                stem = f"F{frames}_{args.fusion}_predictions.csv"
                full_path = (
                    evaluation_root
                    / "full"
                    / f"seed{seed}"
                    / args.weight_source
                    / stem
                )
                ablation_path = (
                    evaluation_root
                    / ablation
                    / f"seed{seed}"
                    / args.weight_source
                    / stem
                )

                if not full_path.is_file() or not ablation_path.is_file():
                    if args.strict:
                        raise SystemExit(
                            f"Missing paired predictions: {full_path} or {ablation_path}"
                        )
                    continue

                full = read_predictions(full_path)
                changed = read_predictions(ablation_path)

                if set(full) != set(changed):
                    only_ablation = sorted(set(changed) - set(full))[:5]
                    only_full = sorted(set(full) - set(changed))[:5]
                    raise ValueError(
                        f"Track mismatch for {ablation}, seed {seed}, F={frames}. "
                        f"Only in ablation: {only_ablation}; only in full: {only_full}"
                    )

                full_only = 0
                ablation_only = 0
                same = 0

                for track in sorted(full):
                    if full[track]["ground_truth"] != changed[track]["ground_truth"]:
                        raise ValueError(
                            f"Ground-truth mismatch for track {track}: "
                            f"{full_path} vs {ablation_path}"
                        )

                    full_correct = as_bool(full[track]["correct"])
                    ablation_correct = as_bool(changed[track]["correct"])
                    if full_correct and not ablation_correct:
                        full_only += 1
                    elif ablation_correct and not full_correct:
                        ablation_only += 1
                    else:
                        same += 1

                total = full_only + ablation_only + same
                delta_pp = 100.0 * (full_only - ablation_only) / total
                ci_low, ci_high = bootstrap_delta_ci(
                    full_only,
                    ablation_only,
                    same,
                    args.bootstrap_samples,
                    rng,
                )
                pvalue = exact_mcnemar_pvalue(full_only, ablation_only)

                discordance_by_seed[seed] = (full_only, ablation_only, same)
                seed_deltas.append(delta_pp)
                per_seed_tests.append(
                    {
                        "ablation": ablation,
                        "seed": seed,
                        "frames": frames,
                        "n_tracks": total,
                        "full_only_correct": full_only,
                        "ablation_only_correct": ablation_only,
                        "same_outcome": same,
                        "delta_full_minus_ablation_pp": delta_pp,
                        "bootstrap95_low_pp": ci_low,
                        "bootstrap95_high_pp": ci_high,
                        "mcnemar_exact_p": pvalue,
                        "mcnemar_holm_p": None,
                    }
                )

            if discordance_by_seed:
                hierarchical_low, hierarchical_high = hierarchical_bootstrap_ci(
                    discordance_by_seed,
                    args.bootstrap_samples,
                    rng,
                )
                hierarchical_rows.append(
                    {
                        "ablation": ablation,
                        "frames": frames,
                        "n_seeds": len(discordance_by_seed),
                        "seeds": ",".join(
                            str(seed) for seed in sorted(discordance_by_seed)
                        ),
                        "mean_delta_full_minus_ablation_pp": statistics.mean(
                            seed_deltas
                        ),
                        "std_delta_pp": (
                            statistics.stdev(seed_deltas)
                            if len(seed_deltas) > 1
                            else 0.0
                        ),
                        "hierarchical_bootstrap95_low_pp": hierarchical_low,
                        "hierarchical_bootstrap95_high_pp": hierarchical_high,
                    }
                )

    holm_adjust(per_seed_tests)

    paired_fields = [
        "ablation",
        "seed",
        "frames",
        "n_tracks",
        "full_only_correct",
        "ablation_only_correct",
        "same_outcome",
        "delta_full_minus_ablation_pp",
        "bootstrap95_low_pp",
        "bootstrap95_high_pp",
        "mcnemar_exact_p",
        "mcnemar_holm_p",
    ]
    write_csv(output_dir / "paired_seed_tests.csv", per_seed_tests, paired_fields)

    hierarchical_fields = [
        "ablation",
        "frames",
        "n_seeds",
        "seeds",
        "mean_delta_full_minus_ablation_pp",
        "std_delta_pp",
        "hierarchical_bootstrap95_low_pp",
        "hierarchical_bootstrap95_high_pp",
    ]
    write_csv(
        output_dir / "paired_hierarchical_bootstrap.csv",
        hierarchical_rows,
        hierarchical_fields,
    )

    lines = [
        "# Controlled-ablation statistical summary",
        "",
        "Positive paired deltas mean the full model is more accurate than the ablation.",
        "",
        "## Across-seed exact-match accuracy",
        "",
        "| Variant | F | Seeds | Exact match, mean +/- std (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in seed_summary:
        lines.append(
            f"| {row['variant']} | {row['frames']} | {row['n_seeds']} | "
            f"{fmt(row['sequence_accuracy_percent_mean'])} +/- "
            f"{fmt(row['sequence_accuracy_percent_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired full-minus-ablation effects",
            "",
            "| Ablation | F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in hierarchical_rows:
        lines.append(
            f"| {row['ablation']} | {row['frames']} | {row['n_seeds']} | "
            f"{fmt(row['mean_delta_full_minus_ablation_pp'])} +/- "
            f"{fmt(row['std_delta_pp'])} | "
            f"[{fmt(row['hierarchical_bootstrap95_low_pp'])}, "
            f"{fmt(row['hierarchical_bootstrap95_high_pp'])}] |"
        )

    lines.extend(
        [
            "",
            "Per-seed exact McNemar tests, with Holm correction within each seed/F "
            "family, are stored in paired_seed_tests.csv.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "evaluation_root": str(evaluation_root),
        "weight_source": args.weight_source,
        "fusion": args.fusion,
        "requested_seeds": args.seeds,
        "requested_frames": args.frames,
        "requested_variants": args.variants,
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.random_seed,
        "missing_evaluation_files": missing,
        "outputs": [
            "ablation_seed_summary.csv",
            "paired_seed_tests.csv",
            "paired_hierarchical_bootstrap.csv",
            "summary.md",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote statistical analysis to: {output_dir}")
    print(f"Across-seed rows: {len(seed_summary)}")
    print(f"Per-seed paired tests: {len(per_seed_tests)}")
    print(f"Hierarchical comparisons: {len(hierarchical_rows)}")
    if missing:
        print(f"Missing evaluation files skipped: {len(missing)}")


if __name__ == "__main__":
    main()
