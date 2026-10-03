# Training-protocol audit for the OJ-ITS revision

This note records the training settings that are actually active in the current
repository so the manuscript and revision response do not overstate uniformity
across baselines.

## What is common across the controlled baseline comparison

All paper baselines are retrained on the same LRLPR train/validation split,
use 32x96 low-resolution inputs, use the repository LR-only augmentation/data
pipeline, select checkpoints on the validation split, and use no external
training data or pretrained checkpoints in the controlled comparison.

The architectures and task-specific losses are adapted from the authors'
public/official implementations. The optimizer/scheduler recipe is not
identical across all models.

## Active baseline optimization settings

| Model | Optimizer | LR | Weight decay | Scheduler | LR warmup | Max epochs | Train batch | Early-stop patience |
|---|---|---:|---:|---|---:|---:|---:|---:|
| SVTRv2-CTC | AdamW | 1e-4 | 0.0 | CosineAnnealingLR | none | 3000 | 64 | 25 (trainer default) |
| OTE | AdamW | 1e-4 | 0.01 (PyTorch default) | OneCycleLR | 1.5 ep | 500 | 256 | 300 |
| LISTER | AdamW | 1e-4 | 0.01 (PyTorch default) | OneCycleLR | 1.5 ep | 500 | 64 | 25 (trainer default) |
| IGTR | AdamW | 1e-4 | 0.05 | OneCycleLR | 1.5 ep | 500 | 64 | 25 (trainer default) |
| CPPD | AdamW | 1e-4 | 0.01 (PyTorch default) | OneCycleLR | 1.5 ep | 500 | 64 | 25 (trainer default) |
| MDiff4STR | AdamW | 1e-4 | 0.05 | OneCycleLR | 1.5 ep | 500 | 64 | 25 (trainer default) |
| SVTRv2-AR (revision) | AdamW | 1e-4 | 0.05 | OneCycleLR | 10 ep | 100 | 64 | 100 (full budget; best-val checkpoint) |

The proposed submitted model is separate from this baseline family: AdamW
with LR 1e-4 and weight decay 0.001, ReduceLROnPlateau, no LR warmup,
batch 64, maximum 1000 epochs, and validation early stopping.

## Warmup terminology

There are several similarly named config fields and they must not be conflated:

- `LRScheduler.warmup_epoch` controls the OneCycleLR warmup fraction in
  `train_gan.py`. The historical non-CTC baselines use 1.5 epochs; the added
  SVTRv2-AR baseline uses the literature-reported LRLPR schedule of 10 warm-up
  epochs over a 100-epoch run.
- `ema_warmup_epochs` controls only the proposed model's EMA hard-copy phase.
  It is not learning-rate warmup.
- Legacy top-level `warmup_epochs` entries in some baseline YAML files are not
  read by the current LR scheduler and must not be described as LR warmup.
- `force_lr` appears in historical YAML files but is not read by the current
  training script; it has no effect on the runs.

## SVTRv2-AR provenance and adaptations

Architectural provenance is pinned to:

- `Topdu/OpenOCR@1ccfc6ee6161f7133b192e16af3a9265273997ba`
- `configs/rec/nrtr/svtrv2_nrtr.yml`
- `openrec/modeling/encoders/svtrnet.py`
- `openrec/modeling/decoders/nrtr_decoder.py`
- `openrec/preprocess/ar_label_encode.py`
- `openrec/losses/ar_loss.py`

The local encoder and decoder reused by
`models/svtrv2/svtrv2_ar_bridge.py` match the pinned OpenOCR sources after
normalizing the repository-local import path and whitespace.

The current controlled port contains 22.722M parameters in the smoke test,
whereas the competition report describes the team's competition model as
approximately 22.42M parameters. Therefore the revision manuscript should
describe this baseline as an OpenOCR-derived SVTRv2/NRTR autoregressive
configuration adapted to the controlled LRLPR protocol, not as an exact
reproduction of the competition team's private training artifact.

Controlled adaptations are limited to the LRLPR experiment:

1. input size 32x128 -> 32x96;
2. output vocabulary -> 36 alphanumeric characters plus EOS/BOS/PAD;
3. maximum plate content length -> seven characters;
4. training from scratch in the same LRLPR data/augmentation pipeline;
5. LRLPR-specific SVTRv2-AR schedule from the ICPR 2026 competition report:
   OneCycleLR over 100 epochs with 10 warm-up epochs;
6. controlled local optimizer scale (LR 1e-4, weight decay 0.05, batch 64)
   and validation selection on the paper's 1k-track validation split;
7. multi-frame inference -> BJP/product-rule only.

For the AR decoder, BJP is applied with one shared fused prefix. At step t,
every frame evaluates the next-token distribution conditioned on the same
prefix; the log probabilities are summed, the fused token is selected, and
that token is appended to the shared prefix. The frame-conditioned decoder
calls are batched together, so this preserves the paper's batched multi-frame
execution semantics without mixing different AR histories.

## Reproducibility rule for the revision run

The ICPR 2026 competition report states that the OpenOCR team trained
SVTRv2-AR with AdamW and OneCycleLR for 100 epochs including 10 warm-up
epochs, used PARSeq augmentation, and monitored a 25-track validation holdout.
For the controlled paper comparison we preserve only the architecture and the
reported 100/10 scheduler shape while keeping the manuscript's common 19k/1k
split, 32x96 input, and repository augmentation pipeline.

Do not change the SVTRv2-AR config after the training run begins. The generated
`config_snapshot.yaml`, training log, validation-selected checkpoint name,
seed, and final F1/F3/F5 test outputs should be retained as revision evidence.
