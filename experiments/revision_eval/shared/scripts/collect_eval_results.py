#!/usr/bin/env python3
"""Collect structured test.py metrics JSON files into reviewer-ready summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "source",
    "frames",
    "fusion",
    "sequence_accuracy_percent",
    "sequence_correct",
    "sequence_total",
    "character_accuracy_percent",
    "cer_percent",
    "partial_6_percent",
    "partial_5_percent",
    "old_brazilian_accuracy_percent",
    "mercosur_accuracy_percent",
    "confidence_gap",
]


def load_rows(input_dir: Path):
    rows = []

    for path in sorted(input_dir.rglob("*.json")):
        if path.name == "matrix_manifest.json":
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue

        row = {
            "source": str(path),
            "frames": payload.get("frames"),
            "fusion": payload.get("fusion"),
        }
        for field in FIELDS[3:]:
            row[field] = metrics.get(field)
        rows.append(row)

    rows.sort(key=lambda row: (int(row["frames"] or 0), str(row["fusion"] or ""), row["source"]))
    return rows


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("F", "frames"),
        ("Fusion", "fusion"),
        ("Seq. Acc. (%)", "sequence_accuracy_percent"),
        ("Char. Acc. (%)", "character_accuracy_percent"),
        ("CER (%)", "cer_percent"),
        ("≥6/7 (%)", "partial_6_percent"),
        ("≥5/7 (%)", "partial_5_percent"),
    ]

    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(key)) for _, key in columns)
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate structured evaluation JSON files into CSV/Markdown."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_dir))
    if not rows:
        raise SystemExit(f"No structured metrics JSON files found under {args.input_dir}")

    write_csv(rows, Path(args.output_csv))
    if args.output_md:
        write_markdown(rows, Path(args.output_md))

    print(f"Collected {len(rows)} evaluation result(s).")
    print(f"CSV: {args.output_csv}")
    if args.output_md:
        print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
