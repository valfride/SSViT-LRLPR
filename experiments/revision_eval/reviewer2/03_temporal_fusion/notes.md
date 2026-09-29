# Reviewer 2.3 — Temporal-fusion audit

## Submitted-model comparison

All results below use the frozen submitted checkpoint, the LRLPR-26 test split
(3,000 tracklets), no TTA, no SWA, and the structured `test.py` evaluator.

| Frames | Fusion | 7/7 (%) | Char. acc. (%) | CER (%) | >=6/7 (%) | >=5/7 (%) |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Probability average | 72.00 | 92.37 | 7.62 | 87.57 | 93.57 |
| 3 | Product/log-probability | 74.17 | 92.96 | 7.03 | 88.30 | 94.27 |
| 3 | Logit average | 74.17 | 92.96 | 7.03 | 88.30 | 94.27 |
| 3 | Sequence majority | 67.30 | 90.87 | 9.13 | 84.87 | 91.63 |
| 5 | Probability average | 76.47 | 93.60 | 6.40 | 89.53 | 94.77 |
| 5 | Product/log-probability | 78.47 | 94.15 | 5.84 | 90.60 | 95.03 |
| 5 | Logit average | 78.50 | 94.16 | 5.83 | 90.63 | 95.03 |
| 5 | Sequence majority | 71.97 | 91.92 | 8.07 | 86.63 | 92.37 |

## Product rule versus logit averaging

For the proposed fixed-position softmax decoder, product/log-probability fusion and
logit averaging are expected to make the same class decision in exact arithmetic:
the per-frame softmax normalization term is independent of the candidate class, so
summing log-softmax scores differs from summing logits only by a class-independent
constant.

The empirical outputs support this interpretation:

- F=3: all 3,000 tracklet predictions are identical.
- F=5: only 3/3,000 predictions differ. One of those three changes from an
  incorrect prediction under product/log-probability fusion to a correct prediction
  under logit averaging, producing 78.47% versus 78.50%. The tiny discrepancy is
  consistent with finite-precision/numerical tie effects rather than a materially
  different fusion rule.

Therefore the revised manuscript should avoid presenting this operation as a
distinct Bayesian inference mechanism. A more accurate description is
**product-rule (sum-log-probability) temporal pooling**.

## Reviewer concern that remains open

The submitted-model comparison does not resolve the reviewer's concern for CTC
baselines. Pre-collapse pooling of CTC timestep logits assumes correspondence of
CTC timesteps across frames, which is not guaranteed. The next experiment will use
SVTRv2 (CTC) to compare pre-collapse pooling against post-decoding sequence-level
and character-position-level fusion.

This CTC experiment should be completed before Reviewer 2.3 is marked resolved.
