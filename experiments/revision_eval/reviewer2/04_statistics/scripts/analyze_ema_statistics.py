#!/usr/bin/env python3
"""Statistical analysis for the matched-epoch EMA-versus-student ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


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


def exact_mcnemar_pvalue(ema_only: int, student_only: int) -> float:
    n = int(ema_only + student_only)
    if n == 0:
        return 1.0

    tail = min(int(ema_only), int(student_only))
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
    ema_only: int,
    student_only: int,
    same: int,
    samples: int,
    rng: np.random.Generator,
):
    total = ema_only + student_only + same
    if total <= 0:
        return float("nan"), float("nan")

    probabilities = np.array([ema_only, student_only, same], dtype=float) / total
    counts = rng.multinomial(total, probabilities, size=samples)
    deltas = 100.0 * (counts[:, 0] - counts[:, 1]) / total
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def hierarchical_bootstrap_ci(discordance_by_seed, samples, rng):
    seed_keys = sorted(discordance_by_seed)
    if not seed_keys:
        return float("nan"), float("nan")

    seed_bootstrap = []
    for seed in seed_keys:
        ema_only, student_only, same = discordance_by_seed[seed]
        total = ema_only + student_only + same
        probabilities = np.array([ema_only, student_only, same], dtype=float) / total
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


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.{digits}f}"


def main() -> None:
    root = repo_root()
    evaluation_root_default = (
        root
        / "experiments/revision_eval/reviewer2/01_controlled_ablation/results/evaluations"
    )
    output_default = root / "experiments/revision_eval/reviewer2/04_statistics/results/ema"

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", default=str(evaluation_root_default))
    parser.add_argument("--output-dir", default=str(output_default))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--frames", nargs="+", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--fusion", default="bayes")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20261002)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")

    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.random_seed)

    metrics_by_source = {"ghost": {}, "student": {}}
    missing = []

    for source in ("ghost", "student"):
        for seed in args.seeds:
            for frames in args.frames:
                path = (
                    evaluation_root
                    / "full"
                    / f"seed{seed}"
                    / source
                    / f"F{frames}_{args.fusion}.json"
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = read_json(path)
                metrics_by_source[source][(seed, frames)] = payload.get("metrics", {})

    if missing and args.strict:
        raise SystemExit("Missing EMA/student evaluation files:\n" + "\n".join(missing))

    seed_summary = []
    for source in ("ghost", "student"):
        for frames in args.frames:
            available = [
                (seed, metrics_by_source[source][(seed, frames)])
                for seed in args.seeds
                if (seed, frames) in metrics_by_source[source]
            ]
            if not available:
                continue

            row = {
                "weight_source": source,
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

    summary_fields = ["weight_source", "frames", "n_seeds", "seeds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "ema_seed_summary.csv", seed_summary, summary_fields)

    per_seed = []
    hierarchical = []

    for frames in args.frames:
        discordance_by_seed = {}
        seed_deltas = []

        for seed in args.seeds:
            stem = f"F{frames}_{args.fusion}_predictions.csv"
            ema_path = (
                evaluation_root / "full" / f"seed{seed}" / "ghost" / stem
            )
            student_path = (
                evaluation_root / "full" / f"seed{seed}" / "student" / stem
            )

            if not ema_path.is_file() or not student_path.is_file():
                if args.strict:
                    raise SystemExit(
                        f"Missing paired EMA/student predictions: {ema_path} or {student_path}"
                    )
                continue

            ema = read_predictions(ema_path)
            student = read_predictions(student_path)

            if set(ema) != set(student):
                only_student = sorted(set(student) - set(ema))[:5]
                only_ema = sorted(set(ema) - set(student))[:5]
                raise ValueError(
                    f"Track mismatch for seed {seed}, F={frames}. "
                    f"Only in student: {only_student}; only in EMA: {only_ema}"
                )

            ema_only = 0
            student_only = 0
            same = 0

            for track in sorted(ema):
                if ema[track]["ground_truth"] != student[track]["ground_truth"]:
                    raise ValueError(
                        f"Ground-truth mismatch for track {track}: "
                        f"{ema_path} vs {student_path}"
                    )

                ema_correct = as_bool(ema[track]["correct"])
                student_correct = as_bool(student[track]["correct"])

                if ema_correct and not student_correct:
                    ema_only += 1
                elif student_correct and not ema_correct:
                    student_only += 1
                else:
                    same += 1

            total = ema_only + student_only + same
            delta_pp = 100.0 * (ema_only - student_only) / total
            ci_low, ci_high = bootstrap_delta_ci(
                ema_only,
                student_only,
                same,
                args.bootstrap_samples,
                rng,
            )
            pvalue = exact_mcnemar_pvalue(ema_only, student_only)

            discordance_by_seed[seed] = (ema_only, student_only, same)
            seed_deltas.append(delta_pp)
            per_seed.append(
                {
                    "seed": seed,
                    "frames": frames,
                    "n_tracks": total,
                    "ema_only_correct": ema_only,
                    "student_only_correct": student_only,
                    "same_outcome": same,
                    "delta_ema_minus_student_pp": delta_pp,
                    "bootstrap95_low_pp": ci_low,
                    "bootstrap95_high_pp": ci_high,
                    "mcnemar_exact_p": pvalue,
                }
            )

        if discordance_by_seed:
            low, high = hierarchical_bootstrap_ci(
                discordance_by_seed,
                args.bootstrap_samples,
                rng,
            )
            hierarchical.append(
                {
                    "frames": frames,
                    "n_seeds": len(discordance_by_seed),
                    "seeds": ",".join(str(seed) for seed in sorted(discordance_by_seed)),
                    "mean_delta_ema_minus_student_pp": statistics.mean(seed_deltas),
                    "std_delta_pp": (
                        statistics.stdev(seed_deltas)
                        if len(seed_deltas) > 1
                        else 0.0
                    ),
                    "hierarchical_bootstrap95_low_pp": low,
                    "hierarchical_bootstrap95_high_pp": high,
                }
            )

    per_seed_fields = [
        "seed",
        "frames",
        "n_tracks",
        "ema_only_correct",
        "student_only_correct",
        "same_outcome",
        "delta_ema_minus_student_pp",
        "bootstrap95_low_pp",
        "bootstrap95_high_pp",
        "mcnemar_exact_p",
    ]
    write_csv(output_dir / "ema_paired_seed_tests.csv", per_seed, per_seed_fields)

    hierarchical_fields = [
        "frames",
        "n_seeds",
        "seeds",
        "mean_delta_ema_minus_student_pp",
        "std_delta_pp",
        "hierarchical_bootstrap95_low_pp",
        "hierarchical_bootstrap95_high_pp",
    ]
    write_csv(
        output_dir / "ema_hierarchical_bootstrap.csv",
        hierarchical,
        hierarchical_fields,
    )

    lines = [
        "# EMA-versus-student statistical summary",
        "",
        "Positive deltas mean the matched-epoch EMA model is more accurate than the student.",
        "",
        "## Across-seed exact-match accuracy",
        "",
        "| Source | F | Seeds | Exact match, mean +/- std (%) |",
        "|---|---:|---:|---:|",
    ]

    for row in seed_summary:
        lines.append(
            f"| {row['weight_source']} | {row['frames']} | {row['n_seeds']} | "
            f"{fmt(row['sequence_accuracy_percent_mean'])} +/- "
            f"{fmt(row['sequence_accuracy_percent_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired EMA-minus-student effects",
            "",
            "| F | Seeds | Delta exact match, mean +/- std (pp) | Hierarchical 95% CI (pp) |",
            "|---:|---:|---:|---:|",
        ]
    )

    for row in hierarchical:
        lines.append(
            f"| {row['frames']} | {row['n_seeds']} | "
            f"{fmt(row['mean_delta_ema_minus_student_pp'])} +/- "
            f"{fmt(row['std_delta_pp'])} | "
            f"[{fmt(row['hierarchical_bootstrap95_low_pp'])}, "
            f"{fmt(row['hierarchical_bootstrap95_high_pp'])}] |"
        )

    lines.extend(
        [
            "",
            "Per-seed exact McNemar tests are stored in ema_paired_seed_tests.csv.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ema_summary.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "purpose": "matched-epoch EMA versus student ablation",
        "evaluation_root": str(evaluation_root),
        "fusion": args.fusion,
        "requested_seeds": args.seeds,
        "requested_frames": args.frames,
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.random_seed,
        "missing_evaluation_files": missing,
        "outputs": [
            "ema_seed_summary.csv",
            "ema_paired_seed_tests.csv",
            "ema_hierarchical_bootstrap.csv",
            "ema_summary.md",
        ],
    }
    (output_dir / "ema_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote EMA statistical analysis to: {output_dir}")
    print(f"Across-seed rows: {len(seed_summary)}")
    print(f"Per-seed paired tests: {len(per_seed)}")
    print(f"Hierarchical comparisons: {len(hierarchical)}")
    if missing:
        print(f"Missing evaluation files skipped: {len(missing)}")


if __name__ == "__main__":
    main()
