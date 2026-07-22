import argparse
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm

import datasets
import models
import utils
from train_funcs.train_utils import ctc_greedy_decoder, decode_batch_logits, strLabelConverter


PLATE_LENGTH = 7


def normalize_plate(text):
    """Normalize labels/predictions to uppercase alphanumeric plate text."""
    return "".join(ch for ch in str(text or "").upper() if ch.isalnum())


def classify_brazilian_layout(text):
    """Return 'old', 'mercosur', or None for a normalized seven-character plate."""
    plate = normalize_plate(text)
    is_letter = lambda ch: "A" <= ch <= "Z"
    is_digit = lambda ch: "0" <= ch <= "9"
    if len(plate) != PLATE_LENGTH or not all(is_letter(ch) for ch in plate[:3]):
        return None
    if all(is_digit(ch) for ch in plate[3:]):
        return "old"
    if is_digit(plate[3]) and is_letter(plate[4]) and all(is_digit(ch) for ch in plate[5:]):
        return "mercosur"
    return None


def positional_match_count(prediction, target, length=PLATE_LENGTH):
    """Count equal characters at fixed positions; missing characters count as errors."""
    prediction = normalize_plate(prediction)
    target = normalize_plate(target)
    return sum(
        int(i < len(prediction) and i < len(target) and prediction[i] == target[i])
        for i in range(int(length))
    )


def edit_distance(left, right):
    """Levenshtein distance for CER, including insertions and deletions."""
    left = normalize_plate(left)
    right = normalize_plate(right)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def checkpoint_accuracy(path):
    """Extract only the value following 'acc_' from a checkpoint filename."""
    match = re.search(r"(?:^|_)acc_([0-9]+(?:\.[0-9]+)?)", Path(path).stem)
    return float(match.group(1)) if match else float("-inf")


def clean_state_dict(state_dict):
    return {str(key).replace("module.", ""): value for key, value in state_dict.items()}


def state_dict_from_checkpoint(checkpoint, preferred_key=None):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary or contain a state dictionary.")

    candidate_keys = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys.extend(["model_ghost_sd", "model_g_sd", "state_dict"])

    for key in dict.fromkeys(candidate_keys):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return clean_state_dict(value), key

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return clean_state_dict(checkpoint), "raw_state_dict"

    raise KeyError("Checkpoint contains no model_ghost_sd, model_g_sd, state_dict, or raw tensor state dict.")


def load_model_state(model, state_dict, source_label):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing model keys from {source_label}: {len(missing)} (e.g. {missing[:5]})")
    if unexpected:
        print(f"⚠️  Unexpected checkpoint keys from {source_label}: {len(unexpected)} (e.g. {unexpected[:5]})")
    if len(missing) > 25 or len(unexpected) > 25:
        print("🚨 Large checkpoint/model mismatch detected; verify that the config matches the checkpoint.")


def average_state_dicts(state_dicts):
    """Average floating tensors while preserving non-floating buffers from the reference model."""
    if not state_dicts:
        raise ValueError("No state dictionaries were provided for SWA.")

    reference = state_dicts[0]
    averaged = {}
    for key, reference_value in reference.items():
        if torch.is_floating_point(reference_value) or torch.is_complex(reference_value):
            accumulator_dtype = torch.complex64 if torch.is_complex(reference_value) else torch.float32
            accumulator = reference_value.detach().to(dtype=accumulator_dtype).clone()
            for state_dict in state_dicts[1:]:
                accumulator.add_(state_dict[key].detach().to(dtype=accumulator_dtype))
            averaged[key] = (accumulator / len(state_dicts)).to(dtype=reference_value.dtype)
        else:
            averaged[key] = reference_value.detach().clone()
    return averaged


def looks_like_probabilities(tensor):
    if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
        return False
    if tensor.numel() == 0:
        return False
    if tensor.detach().min().item() < -1.0e-6 or tensor.detach().max().item() > 1.0 + 1.0e-6:
        return False
    sums = tensor.sum(dim=-1)
    return torch.allclose(sums, torch.ones_like(sums), atol=1.0e-3, rtol=1.0e-3)


def normalized_sequence_confidence(raw_score, prediction_length, is_ctc):
    score = float(raw_score)
    if not math.isfinite(score):
        return 0.0
    if is_ctc:
        # The repository's CTC decoder normally returns a probability-like score.
        return min(max(score, 0.0), 1.0) if score >= 0.0 else math.exp(score / max(1, prediction_length))
    return min(max(math.exp(score / max(1, prediction_length)), 0.0), 1.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal low-resolution license-plate recognition models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", required=True, choices=["val", "test"])
    parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging")
    parser.add_argument("--num_swa", type=int, default=5, help="Number of top checkpoints to average")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--output", default="submission.txt")
    parser.add_argument("--in_images", type=int, default=None, help="Override the number of temporal frames")
    parser.add_argument(
        "--fusion",
        type=str,
        default="logit_average",
        choices=["bayes", "average", "majority", "logit_average"],
        help="Temporal fusion strategy",
    )
    args = parser.parse_args()

    if args.num_swa < 1:
        parser.error("--num_swa must be at least 1")

    utils.setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty YAML config: {args.config}")

    cls_loss_type = str(config.get("cls_loss", "SmoothPoly1")).upper()

    print("Building Architecture...")
    model = models.make(config["model_g"]).to(device)

    ckpt_dir = Path(args.checkpoints)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if args.swa:
        print(f"\n⚖️  SWA ENABLED: Targeting Top {args.num_swa} Models...")
        checkpoint_paths = sorted(ckpt_dir.glob("*acc_*.pth"), key=checkpoint_accuracy, reverse=True)
        if not checkpoint_paths:
            print("⚠️  No accuracy-named checkpoints found for SWA. Falling back to last.pth")
            checkpoint_paths = [ckpt_dir / "last.pth"]
        checkpoint_paths = checkpoint_paths[: args.num_swa]
        if not checkpoint_paths[0].exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_paths[0]}")

        valid_state_dicts = []
        preferred_key = None
        reference_keys = None
        reference_shapes = None

        for index, checkpoint_path in enumerate(checkpoint_paths):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if index > 0 and preferred_key and preferred_key not in checkpoint:
                    print(f"  ⚠️ [SKIP]    {checkpoint_path.name} (Missing {preferred_key})")
                    continue
                state_dict, source_key = state_dict_from_checkpoint(checkpoint, preferred_key=preferred_key)
                if index == 0:
                    preferred_key = source_key if source_key != "raw_state_dict" else None
                    reference_keys = set(state_dict)
                    reference_shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
                    valid_state_dicts.append(state_dict)
                    print(f"🧠 SWA Target Source: '{source_key}'")
                    print(f"  ✅ [REF]     {checkpoint_path.name}")
                    continue

                compatible = set(state_dict) == reference_keys and all(
                    tuple(value.shape) == reference_shapes[key] for key, value in state_dict.items()
                )
                if compatible:
                    valid_state_dicts.append(state_dict)
                    print(f"  ✅ [INCLUDE] {checkpoint_path.name}")
                else:
                    print(f"  ⚠️ [SKIP]    {checkpoint_path.name} (State-dict keys or shapes differ)")
            except Exception as exc:
                print(f"  ❌ [ERROR]   {checkpoint_path.name}: {exc}")

        if not valid_state_dicts:
            raise RuntimeError("No compatible checkpoints found for SWA.")

        swa_state = average_state_dicts(valid_state_dicts)
        load_model_state(model, swa_state, f"SWA of {len(valid_state_dicts)} checkpoint(s)")
        print(f"✅ SWA weights loaded from {len(valid_state_dicts)} compatible checkpoint(s).")
    else:
        checkpoint_paths = sorted(ckpt_dir.glob("*acc_*.pth"), key=checkpoint_accuracy, reverse=True)
        if checkpoint_paths:
            best_checkpoint = checkpoint_paths[0]
            print(f"\nLoading Best Checkpoint: {best_checkpoint}")
        else:
            best_checkpoint = ckpt_dir / "last.pth"
            print(f"\nLoading Fallback Checkpoint: {best_checkpoint}")

        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint}")

        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        state_dict, source_key = state_dict_from_checkpoint(checkpoint)
        if source_key == "model_ghost_sd":
            print("👻 Found Ghost EMA weights! Promoting Ghost to Primary Inference Model.")
        elif source_key == "model_g_sd":
            print("👤 No Ghost weights found. Loading standard Student model.")
        else:
            print(f"📦 Loading model weights from '{source_key}'.")
        load_model_state(model, state_dict, best_checkpoint.name)

    model.eval()

    print(f"\nPreparing Data from {args.split}...")
    if "val_dataset" not in config:
        raise KeyError("The config must contain a val_dataset section for this evaluator.")

    if args.in_images is not None:
        config["val_dataset"]["wrapper"]["args"]["in_images"] = args.in_images
        print(f"🔄 TEMPORAL OVERRIDE: Forced val_dataset in_images to {args.in_images}")

    dataset_spec = config["val_dataset"]["dataset"]
    dataset_spec["args"]["path_split"] = args.split
    dataset_spec["args"]["phase"] = "test"
    base_dataset = datasets.make(dataset_spec)

    wrapper_spec = config["val_dataset"]["wrapper"]
    wrapper_spec["args"]["dataset"] = base_dataset
    wrapper_spec["args"]["test"] = True
    val_dataset = datasets.make(wrapper_spec)

    num_workers = int(config["val_dataset"].get("num_workers", 4))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=val_dataset.collate_fn,
    )

    # Keep the default alphabet synchronized with train_utils.strLabelConverter (no hyphen).
    true_converter = strLabelConverter(config.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    track_names_ordered = []
    metadata_path = Path(args.split) / "metadata.pkl"
    if metadata_path.exists():
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        for item in metadata:
            match = re.search(r"(track_\d+)", str(item))
            if match:
                track_names_ordered.append(match.group(1))
    track_names_ordered = list(dict.fromkeys(track_names_ordered))

    correct_plates = 0
    correct_6plus = 0
    correct_5plus = 0
    correct_char_positions = 0
    total_char_positions = 0
    total_edit_errors = 0
    total_gt_characters = 0
    total_plates = 0

    correct_mercosur = 0
    total_mercosur = 0
    correct_brazil = 0
    total_brazil = 0

    failures = []
    submission_lines = []
    confidence_tracking = []
    warned_bad_gt = False

    use_fp16 = bool(config.get("use_fp16", True)) and device.type == "cuda"

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            desc=f"Evaluating ({args.mode.upper()} Mode | Fusion: {args.fusion.upper()})",
        )
        for batch in progress:
            lr_sequences = batch["lr_seq"].to(device, non_blocking=True)
            raw_gt = batch["gt"][0] if "gt" in batch and batch["gt"][0] else ""
            gt_text = normalize_plate(raw_gt)

            if args.mode == "val" and not gt_text:
                raise ValueError("Validation mode requires non-empty ground-truth labels in every batch.")
            if args.mode == "val" and len(gt_text) != PLATE_LENGTH and not warned_bad_gt:
                print(f"⚠️  Found non-{PLATE_LENGTH}-character ground truth ({raw_gt!r}); fixed-slot metrics still use 7 positions.")
                warned_bad_gt = True

            track_name = None
            for value in batch.values():
                if isinstance(value, (list, tuple, str)):
                    match = re.search(r"(track_\d+)", str(value))
                    if match:
                        track_name = match.group(1)
                        break

            if not track_name and track_names_ordered and total_plates < len(track_names_ordered):
                track_name = track_names_ordered[total_plates]
            if not track_name:
                track_name = f"track_{total_plates:05d}"

            batch_size, sequence_length, channels, height, width = lr_sequences.shape
            flat_images = lr_sequences.view(batch_size * sequence_length, channels, height, width)
            flat_images = flat_images.contiguous().to(memory_format=torch.channels_last)

            view_logits = []

            def get_logits(output_dict):
                if not isinstance(output_dict, dict):
                    raise TypeError(f"Expected model output dict, got {type(output_dict).__name__}")
                if cls_loss_type == "EVIDENTIAL":
                    if "expected_prob" not in output_dict:
                        raise KeyError("Evidential model output is missing 'expected_prob'.")
                    return torch.log(output_dict["expected_prob"].clamp_min(1.0e-8))
                if "logits" not in output_dict:
                    raise KeyError("Model output is missing 'logits'.")
                return output_dict["logits"]

            with torch.amp.autocast("cuda", enabled=use_fp16):
                base_output = model(flat_images, epoch=100)
                if isinstance(base_output, tuple):
                    base_output = base_output[0]
                view_logits.append(get_logits(base_output))

                if args.tta:
                    positive_images = TF.rotate(
                        flat_images,
                        angle=2.5,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                    positive_output = model(positive_images, epoch=100)
                    if isinstance(positive_output, tuple):
                        positive_output = positive_output[0]
                    view_logits.append(get_logits(positive_output))

                    negative_images = TF.rotate(
                        flat_images,
                        angle=-2.5,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                    negative_output = model(negative_images, epoch=100)
                    if isinstance(negative_output, tuple):
                        negative_output = negative_output[0]
                    view_logits.append(get_logits(negative_output))

            if args.fusion in {"bayes", "average", "logit_average"}:
                accumulated_log_probs = None
                accumulated_probabilities = None
                accumulated_logits = None

                for logits_tensor in view_logits:
                    if logits_tensor.ndim != 3:
                        raise ValueError(f"Expected logits [B*frames, chars, classes], got {tuple(logits_tensor.shape)}")
                    if logits_tensor.size(0) != batch_size * sequence_length:
                        raise ValueError(
                            "Model output batch does not match B*sequence_length: "
                            f"{logits_tensor.size(0)} vs {batch_size * sequence_length}"
                        )

                    _, num_characters, num_classes = logits_tensor.shape
                    logits_sequence = logits_tensor.view(
                        batch_size,
                        sequence_length,
                        num_characters,
                        num_classes,
                    )

                    if looks_like_probabilities(logits_sequence):
                        probabilities = logits_sequence.clamp_min(1.0e-8)
                        log_probabilities = torch.log(probabilities)
                    else:
                        probabilities = torch.softmax(logits_sequence, dim=-1)
                        log_probabilities = F.log_softmax(logits_sequence, dim=-1)

                    if args.fusion == "bayes":
                        temporal_fused = log_probabilities.sum(dim=1)
                        accumulated_log_probs = (
                            temporal_fused
                            if accumulated_log_probs is None
                            else accumulated_log_probs + temporal_fused
                        )
                    elif args.fusion == "average":
                        temporal_fused = probabilities.mean(dim=1)
                        accumulated_probabilities = (
                            temporal_fused
                            if accumulated_probabilities is None
                            else accumulated_probabilities + temporal_fused
                        )
                    else:
                        temporal_fused = logits_sequence.mean(dim=1)
                        accumulated_logits = (
                            temporal_fused
                            if accumulated_logits is None
                            else accumulated_logits + temporal_fused
                        )

                if args.fusion == "bayes":
                    fused_pseudo_logits = accumulated_log_probs / len(view_logits)
                elif args.fusion == "average":
                    fused_pseudo_logits = torch.log(
                        (accumulated_probabilities / len(view_logits)).clamp_min(1.0e-8)
                    )
                else:
                    fused_pseudo_logits = accumulated_logits / len(view_logits)

                if cls_loss_type == "CTC":
                    predictions, scores = ctc_greedy_decoder(
                        fused_pseudo_logits,
                        true_converter,
                        return_scores=True,
                    )
                else:
                    predictions = decode_batch_logits(fused_pseudo_logits, true_converter)
                    scores = F.log_softmax(fused_pseudo_logits, dim=-1).max(dim=-1).values.sum(dim=1)

                final_prediction = normalize_plate(predictions[0])
                raw_confidence = scores[0].item() if torch.is_tensor(scores[0]) else float(scores[0])

            else:  # majority
                decoded_strings = []
                decoded_confidences = []

                for logits_tensor in view_logits:
                    if cls_loss_type == "CTC":
                        frame_predictions, frame_scores = ctc_greedy_decoder(
                            logits_tensor,
                            true_converter,
                            return_scores=True,
                        )
                    else:
                        frame_predictions = decode_batch_logits(logits_tensor, true_converter)
                        frame_scores = F.log_softmax(logits_tensor, dim=-1).max(dim=-1).values.sum(dim=1)

                    decoded_strings.extend(normalize_plate(value) for value in frame_predictions)
                    if torch.is_tensor(frame_scores):
                        decoded_confidences.extend(float(value) for value in frame_scores.detach().cpu().tolist())
                    else:
                        decoded_confidences.extend(float(value) for value in frame_scores)

                if not decoded_strings:
                    raise RuntimeError("Majority fusion produced no decoded frame predictions.")

                vote_counts = defaultdict(int)
                vote_confidences = defaultdict(list)
                for decoded, confidence in zip(decoded_strings, decoded_confidences):
                    vote_counts[decoded] += 1
                    vote_confidences[decoded].append(confidence)

                maximum_votes = max(vote_counts.values())
                tied_candidates = [
                    decoded for decoded, count in vote_counts.items() if count == maximum_votes
                ]
                final_prediction = max(
                    tied_candidates,
                    key=lambda decoded: sum(vote_confidences[decoded]) / len(vote_confidences[decoded]),
                )
                raw_confidence = sum(vote_confidences[final_prediction]) / len(
                    vote_confidences[final_prediction]
                )

            normalized_confidence = normalized_sequence_confidence(
                raw_confidence,
                len(final_prediction),
                is_ctc=cls_loss_type == "CTC",
            )

            if args.mode == "val":
                match_count = positional_match_count(final_prediction, gt_text)
                distance = edit_distance(final_prediction, gt_text)

                correct_char_positions += match_count
                total_char_positions += PLATE_LENGTH
                total_edit_errors += distance
                total_gt_characters += len(gt_text)

                if match_count >= 5:
                    correct_5plus += 1
                if match_count >= 6:
                    correct_6plus += 1

                is_correct = final_prediction == gt_text
                if is_correct:
                    correct_plates += 1
                else:
                    failures.append(
                        f"{track_name} | Pred: {final_prediction} | GT: {gt_text} | "
                        f"Conf: {normalized_confidence:.4f} | Matches: {match_count}/{PLATE_LENGTH} | "
                        f"EditDistance: {distance}"
                    )

                layout = classify_brazilian_layout(gt_text)
                if layout == "old":
                    total_brazil += 1
                    correct_brazil += int(is_correct)
                elif layout == "mercosur":
                    total_mercosur += 1
                    correct_mercosur += int(is_correct)

                confidence_tracking.append(
                    {
                        "correct": is_correct,
                        "char_correct": match_count,
                        "char_total": PLATE_LENGTH,
                        "conf": normalized_confidence,
                    }
                )

                processed = total_plates + 1
                progress.set_postfix(
                    {
                        "SeqAcc": f"{correct_plates / processed:.1%}",
                        "CharAcc": f"{correct_char_positions / max(total_char_positions, 1):.1%}",
                    }
                )
            else:
                submission_lines.append(f"{track_name},{final_prediction};{raw_confidence:.4f}")

            total_plates += 1

    if args.mode == "val":
        sequence_accuracy = 100.0 * correct_plates / max(total_plates, 1)
        character_accuracy = 100.0 * correct_char_positions / max(total_char_positions, 1)
        partial_6_accuracy = 100.0 * correct_6plus / max(total_plates, 1)
        partial_5_accuracy = 100.0 * correct_5plus / max(total_plates, 1)
        character_error_rate = 100.0 * total_edit_errors / max(total_gt_characters, 1)

        brazil_accuracy = 100.0 * correct_brazil / max(total_brazil, 1) if total_brazil else None
        mercosur_accuracy = 100.0 * correct_mercosur / max(total_mercosur, 1) if total_mercosur else None

        print(f"\n🏆 FINAL EVALUATION METRICS ({args.fusion.upper()} FUSION)")
        print(f"   Full Sequence (7/7):       {sequence_accuracy:.2f}% ({correct_plates}/{total_plates})")
        print(
            f"   Character Accuracy (slots): {character_accuracy:.2f}% "
            f"({correct_char_positions}/{total_char_positions})"
        )
        print(
            f"   Character Error Rate (CER): {character_error_rate:.2f}% "
            f"({total_edit_errors}/{total_gt_characters})"
        )
        print(f"   Partial Match (≥6/7):      {partial_6_accuracy:.2f}% ({correct_6plus}/{total_plates})")
        print(f"   Partial Match (≥5/7):      {partial_5_accuracy:.2f}% ({correct_5plus}/{total_plates})")
        print("-" * 48)
        if brazil_accuracy is None:
            print("   🇧🇷 Old Brazilian (LLL-NNNN): N/A (0 samples)")
        else:
            print(
                f"   🇧🇷 Old Brazilian (LLL-NNNN): {brazil_accuracy:.2f}% "
                f"({correct_brazil}/{total_brazil})"
            )
        if mercosur_accuracy is None:
            print("   🌎 Mercosur Layout (LLL-NLNN): N/A (0 samples)")
        else:
            print(
                f"   🌎 Mercosur Layout (LLL-NLNN): {mercosur_accuracy:.2f}% "
                f"({correct_mercosur}/{total_mercosur})"
            )

        print("\n📈 RECOGNITION RATE VS. CONFIDENCE THRESHOLD")
        print("-" * 82)
        print(
            f"| {'Minimum Confidence':<20} | {'Retained (Coverage)':<22} | "
            f"{'Plate Acc':<10} | {'Char Acc':<10} |"
        )
        print(f"|{'-' * 22}|{'-' * 24}|{'-' * 12}|{'-' * 12}|")

        thresholds = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
        for threshold in thresholds:
            retained = [item for item in confidence_tracking if item["conf"] >= threshold]
            if not retained:
                continue

            coverage = 100.0 * len(retained) / max(len(confidence_tracking), 1)
            retained_plate_accuracy = 100.0 * sum(item["correct"] for item in retained) / len(retained)
            retained_char_correct = sum(item["char_correct"] for item in retained)
            retained_char_total = sum(item["char_total"] for item in retained)
            retained_character_accuracy = 100.0 * retained_char_correct / max(retained_char_total, 1)

            print(
                f"| ≥ {threshold:<18.2f} | {len(retained):<6} ({coverage:>6.2f}%)         | "
                f"{retained_plate_accuracy:>7.2f}%   | {retained_character_accuracy:>7.2f}%   |"
            )
        print("-" * 82)

        correct_confidences = [item["conf"] for item in confidence_tracking if item["correct"]]
        incorrect_confidences = [item["conf"] for item in confidence_tracking if not item["correct"]]
        mean_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
        mean_incorrect = (
            sum(incorrect_confidences) / len(incorrect_confidences)
            if incorrect_confidences
            else 0.0
        )
        confidence_gap = mean_correct - mean_incorrect

        print("\n🧠 MODEL CALIBRATION (CONFIDENCE GAP)")
        print("-" * 45)
        print(f"   Mean Conf (Correct):   {mean_correct:.4f}")
        print(f"   Mean Conf (Incorrect): {mean_incorrect:.4f}")
        print(f"   Confidence Gap:        {confidence_gap:.4f}")
        print("-" * 45)

        if failures:
            failure_path = Path("validation_failures_sequence.txt")
            failure_path.write_text("\n".join(failures), encoding="utf-8")
            print(f"\nSaved {len(failures)} failures to {failure_path}")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(submission_lines), encoding="utf-8")
        print(f"✅ Submission saved to {output_path}")


if __name__ == "__main__":
    main()
