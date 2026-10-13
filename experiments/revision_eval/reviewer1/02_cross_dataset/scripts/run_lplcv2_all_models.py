#!/usr/bin/env python3
"""Run all Table-I models on the SAME frozen LPLCv2 F=1 test set.

Requires an explicit local checkpoint lock. Does not select checkpoints,
modify frozen data, train models, or publish individual plate predictions.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "experiments/revision_eval/reviewer1/02_cross_dataset"
EXPECTED = "3d64cb68c98732fc5803a3de40950efa1da1cb9421c0c653ba18b915bed30d05"
MODELS = [
    ("MDiff4STR", "mdiff4str", "MDIFF_BASELINE", "MDIFF_INTERNAL"),
    ("CPPD", "cppd", "CPPD_BASELINE", "CPPD"),
    ("IGTR", "igtr", "IGTR_BASELINE", "IGTR_INTERNAL"),
    ("LISTER", "lister", "LISTER_BASELINE", "LISTER_INTERNAL"),
    ("SVTRv2", "svtrv2", "SVTRV2_BASELINE", "CTC"),
    ("OTE", "ote", "OTE_BASELINE", "OTE"),
    ("SVTRv2-AR", "svtrv2_ar", "SVTRV2_AR_BASELINE", "SVTRV2_AR"),
    ("Ours", "ours", None, "CE"),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_dir(slug):
    return BASE / "results" / ("lplcv2_mixed_aspect_" + slug)


def validate_dataset(path):
    path = path.expanduser().resolve(strict=True)
    if sha256(path / "selection_manifest.json") != EXPECTED:
        raise ValueError("Frozen selection-manifest SHA256 does not match")
    manifest = json.loads((path / "selection_manifest.json").read_text())
    counts = Counter(r["source_legibility"] for r in manifest["samples"])
    if len(manifest["samples"]) != 3000 or counts != {"Poor": 2158, "Illegible": 842}:
        raise ValueError("Unexpected frozen sample counts")
    for name in ("metadata.pkl", "data.mdb"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return path


# Candidate files listed by the user from experiments/baselines on daugman.
# For tied validation accuracies, test.py sorts paths alphabetically first,
# then stably sorts descending by the score after '*acc_*'.
# Therefore the lower-numbered epoch filename wins an exact score tie.
CANDIDATE_CHECKPOINTS = {
    "MDiff4STR": (
        "MDIFF_BASELINE/MDIFF_BASELINE_14-04-2026/"
        "model_acc_0.6040_ep_21.pth"
    ),
    "CPPD": (
        "CPPD_BASELINE/CPPD_BASELINE_14-04-2026/"
        "model_acc_0.6880_ep_77.pth"
    ),
    "IGTR": (
        "IGTR_BASELINE/IGTR_BASELINE_14-04-2026/"
        "model_acc_0.6600_ep_61.pth"
    ),
    "LISTER": (
        "LISTER_BASELINE/LISTER_BASELINE_14-04-2026/"
        "model_acc_0.6830_ep_103.pth"
    ),
    "SVTRv2": (
        "SVTRV2_BASELINE/SVTRV2_BASELINE_14-04-2026/"
        "model_acc_0.7030_ep_146.pth"
    ),
    "OTE": (
        "OTE_BASELINE/OTE_BASELINE_14-04-2026/"
        "model_acc_0.7430_ep_88.pth"
    ),
    "SVTRv2-AR": (
        "SVTRV2_AR_BASELINE_svtrv2_ar/student_weights/"
        "student_acc_0.7500_ep_52.pth"
    ),
}


def inventory():
    base = ROOT / "experiments/baselines"
    if not base.exists():
        raise FileNotFoundError(base)
    print("Available baseline checkpoints (from local daugman directory):")
    for model, _, _, _ in MODELS:
        if model == "Ours":
            continue
        relative = CANDIDATE_CHECKPOINTS[model]
        folder = (base / relative).parent
        print(f"\n{model}: {folder.relative_to(ROOT)}")
        if not folder.is_dir():
            print("    DIRECTORY MISSING")
            continue
        names = sorted(p.name for p in folder.glob("*acc_*.pth") if p.is_file())
        for name in names:
            flag = "  <-- candidate" if name == Path(relative).name else ""
            print(f"    {name}{flag}")
        if not (base / relative).is_file():
            print("    WARNING: expected candidate checkpoint MISSING")
    print("\nTied top accuracies: LISTER epochs 103/113, SVTRv2 epochs "
          "146/156, SVTRv2-AR epochs 52/85/94.")
    print("The candidate follows test.py's stable alphabetical tie order; "
          "confirm the original LRLPR-26 Table-I checkpoint-selection logs "
          "before running LPLCv2.")


def write_template(path):
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing local checkpoint lock")
    choices = {
        model: str(Path("experiments/baselines") / candidate)
        for model, candidate in CANDIDATE_CHECKPOINTS.items()
    }
    choices["Ours"] = "experiments/revision_eval/submitted_model/last.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_manifest_sha256": EXPECTED,
        "checkpoint_selection_note": (
            "Candidates use the existing test.py filename-score selection "
            "rule. Verify against original LRLPR-26 validation logs, "
            "particularly tied checkpoints for LISTER, SVTRv2 and SVTRv2-AR. "
            "Do not select using LPLCv2 performance."
        ),
        "checkpoints": choices,
    }, indent=2) + "\n")
    print(f"Created {path} with all eight local candidate checkpoints.")
    print("Review tied validation selections and confirm paths before --run.")


def load_lock(path):
    doc = json.loads(path.read_text())
    if doc.get("dataset_manifest_sha256") != EXPECTED:
        raise ValueError("Checkpoint lock is for the wrong dataset")
    choices = doc.get("checkpoints", {})
    if set(choices) != {model for model, *_ in MODELS}:
        raise ValueError("Checkpoint lock must specify all eight Table-I models")
    result = {}
    for model, *_ in MODELS:
        value = choices[model]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing validation-selected checkpoint for {model}")
        p = Path(value).expanduser()
        p = (p if p.is_absolute() else ROOT / p).resolve(strict=True)
        if not p.is_file() or p.suffix != ".pth":
            raise ValueError(f"Invalid checkpoint for {model}: {p}")
        result[model] = p
    return result


def config_for(name):
    return ("experiments/revision_eval/submitted_model/config_snapshot.yaml"
            if name is None else f"baselines_configs/{name}.yaml")


def preflight_config(config, expected_loss):
    import yaml
    cfg = yaml.safe_load((ROOT / config).read_text())
    wrapper = cfg["val_dataset"]["wrapper"]
    w = wrapper["args"]
    if cfg.get("cls_loss", "CE") != expected_loss:
        raise ValueError(f"Unexpected cls_loss in {config}")
    if wrapper["name"] != "VSR_Sequence_collate_fn":
        raise ValueError(f"Unexpected validation wrapper in {config}")
    if (int(w["imgW"]), int(w["imgH"]), float(w["image_aspect_ratio"])) != (96, 32, 3.0):
        raise ValueError(f"Input geometry differs from frozen 96x32 in {config}")
    if w.get("aug") or not w.get("test"):
        raise ValueError(f"Validation augmentation/test flags differ in {config}")


def completed(entry, dataset, checkpoint=None):
    model, slug, _, loss = entry
    result = output_dir(slug)
    metrics, preds = result / "F1_bayes.json", result / "F1_bayes_predictions.csv"
    if not metrics.exists() and not preds.exists():
        return None
    if not metrics.is_file() or not preds.is_file():
        raise RuntimeError(f"Incomplete output for {model}; inspect {result}")
    doc = json.loads(metrics.read_text())
    if Path(doc["dataset"]).resolve() != dataset:
        raise ValueError(f"{model}: existing result is for another dataset")
    if (doc["mode"] != "val" or doc["frames"] != 1 or
            doc["tta"] or doc["swa"] or doc["cls_loss"] != loss):
        raise ValueError(f"{model}: existing result uses another evaluation protocol")
    if doc["metrics"]["sequence_total"] != 3000:
        raise ValueError(f"{model}: sequence_total is not 3000")
    if checkpoint and [Path(f).name for f in doc["checkpoint_files"]] != [checkpoint.name]:
        raise ValueError(f"{model}: existing output selected another checkpoint")
    return doc["metrics"]


def preflight(dataset, lock):
    checkpoints = load_lock(lock)
    print("Checkpoint lock preflight (no inference):", flush=True)
    for entry in MODELS:
        model, slug, conf, loss = entry
        preflight_config(config_for(conf), loss)
        current = completed(entry, dataset, checkpoints[model])
        print(f"  {model}: {checkpoints[model]}", flush=True)
        print(f"    SHA256: {sha256(checkpoints[model])}", flush=True)
        print(f"    Status: {'complete, will skip' if current is not None else 'pending'}", flush=True)
        if current is None:
            folder = output_dir(slug)
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(
                    f"{model}: incomplete nonempty output directory; "
                    f"inspect before inference: {folder}")
    return checkpoints


def run(dataset, lock):
    checkpoints = preflight(dataset, lock)
    for entry in MODELS:
        model, slug, conf, loss = entry
        checkpoint = checkpoints[model]
        print(f"{model}: locked {checkpoint} (SHA256 {sha256(checkpoint)})", flush=True)
        if completed(entry, dataset, checkpoint) is not None:
            print(f"  SKIP verified existing F=1 result for {model}", flush=True)
            continue
        folder = output_dir(slug)
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing incomplete output: {folder}")
        links = folder / "frozen_checkpoint_only"
        links.mkdir(parents=True)
        (links / checkpoint.name).symlink_to(checkpoint)
        cmd = [
            sys.executable, "experiments/revision_eval/shared/scripts/run_eval_matrix.py",
            "--config", config_for(conf), "--checkpoints", str(links),
            "--split", str(dataset), "--output-dir", str(folder),
            "--frames", "1", "--fusions", "bayes"
        ]
        print("  Running: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        if completed(entry, dataset, checkpoint) is None:
            raise RuntimeError(f"No complete result returned for {model}")
    print("All eight Table-I model evaluations present and checked.")


def table(dataset):
    print("| Model | F | 7/7 (%) | >=6 (%) | >=5 (%) |")
    print("|---|---:|---:|---:|---:|")
    scores = []
    for entry in MODELS:
        model = entry[0]
        result = completed(entry, dataset)
        if result is None:
            print(f"| {model} | 1 | pending | pending | pending |")
        else:
            values = (result["sequence_accuracy_percent"],
                      result["partial_6_percent"], result["partial_5_percent"])
            scores.append((model, values))
            print(f"| {model} | 1 | {values[0]:.1f} | {values[1]:.1f} | {values[2]:.1f} |")
    print(f"\nCompleted: {len(scores)}/8")
    if len(scores) == 8:
        best = tuple(max(v[j] for _, v in scores) for j in range(3))
        print("\nLaTeX rows (best highlighted):")
        for model, values in scores:
            body = ["\\mathbf{%.1f}" % v if abs(v - best[j]) < 1e-9 else "%.1f" % v
                    for j, v in enumerate(values)]
            print(f"{model} & 1 & " + " & ".join(body) + " \\\\")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--inventory", action="store_true")
    operations.add_argument("--write-lock-template", action="store_true")
    operations.add_argument("--preflight", action="store_true")
    operations.add_argument("--run", action="store_true")
    operations.add_argument("--table", action="store_true")
    parser.add_argument("--dataset", type=Path, default=(
        Path.home() / "doc2025/LMDB-Datasets/LPLCv2_MixedAspect1p7to4p0_TEST_3k"))
    parser.add_argument("--lock", type=Path,
                        default=BASE / "FROZEN_LPLCV2_CHECKPOINTS.local.json")
    args = parser.parse_args()
    if args.inventory:
        inventory()
    elif args.write_lock_template:
        write_template(args.lock)
    else:
        dataset = validate_dataset(args.dataset)
        print("Verified frozen LPLCv2 manifest:", EXPECTED)
        if args.run:
            run(dataset, args.lock)
        elif args.preflight:
            preflight(dataset, args.lock)
        else:
            table(dataset)


if __name__ == "__main__":
    main()
