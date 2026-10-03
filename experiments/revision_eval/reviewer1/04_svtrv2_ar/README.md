# Reviewer 1.4 — SVTRv2-AR baseline

This directory tracks the additional autoregressive SVTRv2 baseline requested
during the OJ-ITS major revision.

The implementation uses the OpenOCR SVTRv2/NRTR configuration as the
architectural reference, adapted to 32x96 LRLPR inputs and the seven-character
Brazilian/Mercosur alphabet. The upstream source is pinned to
`Topdu/OpenOCR@1ccfc6ee6161f7133b192e16af3a9265273997ba`, specifically
`configs/rec/nrtr/svtrv2_nrtr.yml`,
`openrec/modeling/encoders/svtrnet.py`, and
`openrec/modeling/decoders/nrtr_decoder.py`.

The local encoder and decoder sources reused by the bridge are
`models/cppd/svtrnet.py` and `models/ote/nrtr_decoder.py`; after normalizing
the repository-local import path and whitespace, they match the pinned OpenOCR
sources. No external dataset or pretrained checkpoint is enabled by the
revision config.

## Controlled training protocol

The architecture and AR loss follow the pinned OpenOCR implementation, while
training is deliberately run inside the same controlled LRLPR data pipeline
used for the paper baselines. The locked protocol is:

- training from scratch on the same 19k/1k LRLPR train/validation split;
- 32x96 LR inputs and the same repository augmentation wrapper;
- AdamW with learning rate 1e-4 and weight decay 0.05;
- OneCycleLR over 100 epochs with a 10-epoch warmup;
- batch size 64;
- validation-selected checkpoint over the 100-epoch budget;
- no EMA, external pretraining, or external training data.

The 100-epoch/10-warmup schedule follows the LRLPR-specific SVTRv2-AR
training description in the ICPR 2026 competition report. We intentionally
retain the paper's common LRLPR split, input size, and augmentation wrapper
instead of reproducing the competition team's 25-track validation holdout,
PARSeq augmentation, external-data variants, or pretrained variants.

## Temporal fusion policy

Only BJP/product-rule temporal fusion is used, matching the original paper's
multi-frame comparison protocol.

Because the decoder is autoregressive, frame probabilities cannot be fused
after independent greedy histories without changing their conditioning.  The
implementation therefore maintains one shared fused prefix.  At decoding step
t, every frame computes p(y_t | y_<t, x_f) under that same prefix, the
per-frame log probabilities are summed, and the fused argmax token becomes the
shared prefix for the next step.

## Files

- `baselines_configs/SVTRV2_AR_BASELINE.yaml`: controlled training config.
- `models/svtrv2/svtrv2_ar_bridge.py`: baseline bridge, AR loss, and shared-prefix BJP.
- `scripts/smoke_test.py`: architecture/training/BJP sanity check.

## Smoke test

```bash
python3 experiments/revision_eval/reviewer1/04_svtrv2_ar/scripts/smoke_test.py \
  --device cuda:0
```

The test should report finite teacher-forced loss and finite BJP outputs for
F=1, F=3, and F=5 before a full training run is started.
