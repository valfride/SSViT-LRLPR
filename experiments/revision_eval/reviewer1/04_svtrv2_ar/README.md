# Reviewer 1.4 — SVTRv2-AR baseline

This directory tracks the additional autoregressive SVTRv2 baseline requested
during the OJ-ITS major revision.

The implementation uses the OpenOCR SVTRv2/NRTR configuration as the
architectural reference, adapted to 32x96 LRLPR inputs and the seven-character
Brazilian/Mercosur alphabet.  No external dataset or pretrained checkpoint is
enabled by the revision config.

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
