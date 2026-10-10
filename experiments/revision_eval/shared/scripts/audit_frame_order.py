#!/usr/bin/env python3
"""Audit LRLPR frame selection without reading image bytes or changing evaluation.

The current VSR_Sequence_collate_fn wrapper preserves LMDB metadata insertion
order and takes the first F entries for F=1/3/5. This tool checks whether that
order agrees with numbered lr-*.jpg filenames. Numbered filenames are assumed
to indicate capture sequence; timestamps are not available in these inputs.
"""

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


FRAME_RE = re.compile(r"^lr[-_](\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
TRACK_RE = re.compile(r"track_\d+")


def load_names(args):
    if args.benchmark_inputs:
        manifest = json.loads(args.benchmark_inputs.read_text(encoding="utf-8"))
        sequences = manifest.get("names", [])
        if not isinstance(sequences, list):
            raise ValueError("Benchmark input JSON must contain a names list.")
        return {f"benchmark_{i:05d}": names for i, names in enumerate(sequences)}

    # Only load metadata.pkl from a dataset you trust; pickle is not a safe
    # format for untrusted input.
    with args.metadata.open("rb") as stream:
        metadata = pickle.load(stream)
    grouped = defaultdict(list)
    for item in metadata:
        if args.split and args.split not in str(item.get("split", "")):
            continue
        path = str(item.get("original_path", "")).replace("\\", "/")
        match = TRACK_RE.search(path)
        name = path.rsplit("/", 1)[-1]
        if match and name.lower().startswith("lr-"):
            grouped[match.group(0)].append(name)
    return dict(grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metadata", type=Path, help="LMDB metadata.pkl file")
    group.add_argument(
        "--benchmark-inputs", type=Path,
        help="Saved benchmark_inputs.json containing sequence names",
    )
    parser.add_argument("--split", default="test", help="Metadata split (default: test)")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON output")
    parser.add_argument(
        "--require-chronological", action="store_true",
        help="Exit unsuccessfully when metadata order is not ascending by frame index",
    )
    args = parser.parse_args()
    sequences = load_names(args)
    counts = Counter()
    examples = []
    order_counts = Counter()
    for track, names in sequences.items():
        indices = []
        for name in names:
            match = FRAME_RE.match(str(name))
            if not match:
                indices = []
                break
            indices.append(int(match.group(1)))
        if not indices:
            counts["unreadable_frame_names"] += 1
            if len(examples) < args.examples:
                examples.append({"track": track, "names": names, "issue": "unreadable"})
            continue
        counts["tracks_with_frame_indices"] += 1
        order_counts[",".join(map(str, indices))] += 1
        if len(set(indices)) != len(indices):
            counts["duplicate_frame_indices"] += 1
        if indices != sorted(indices):
            counts["nonchronological_order"] += 1
            if len(examples) < args.examples:
                examples.append({
                    "track": track, "names": names,
                    "selected_F1": names[:1], "selected_F3": names[:3],
                    "chronological_first_F3": sorted(names, key=lambda name: int(FRAME_RE.match(name).group(1)))[:3],
                })
        if len(indices) >= 3 and indices[:3] != list(range(indices[0], indices[0] + 3)):
            counts["noncontiguous_F3_prefix"] += 1
        if len(indices) < 5:
            counts["fewer_than_5_LR_frames"] += 1

    report = {
        "source": str(args.benchmark_inputs or args.metadata),
        "total_tracks": len(sequences),
        "assumption": "lr-NNN filename number reflects chronological capture order",
        "counts": dict(counts),
        "most_common_metadata_orders": [
            {"order": order, "tracks": count}
            for order, count in order_counts.most_common(10)
        ],
        "examples": examples,
        "action": "Do not silently reorder for reported tests without rerunning affected F1/F3 evaluations.",
    }
    output_text = json.dumps(report, indent=2) + "\n"
    print(output_text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    return int(args.require_chronological and counts["nonchronological_order"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
