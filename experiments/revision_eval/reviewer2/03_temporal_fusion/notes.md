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

## SVTRv2 CTC comparison

The manuscript SVTRv2 checkpoint was identified by reproducing the submitted
single-frame result. The canonical checkpoint is:

`model_acc_0.7030_ep_156.pth`

SHA-256:

`6590d3da20d204c235e88a65871b8bc04b85ca3dd6d52ad81f73face8b68f231`

It reproduces the submitted SVTRv2 F=1 result exactly: 54.20% (1626/3000).

| Frames | Fusion | 7/7 (%) | Char. acc. (%) | CER (%) | >=6/7 (%) | >=5/7 (%) |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Probability average | 65.93 | 89.96 | 9.10 | 83.87 | 91.40 |
| 3 | Product/log-probability | 68.57 | 91.02 | 8.40 | 85.60 | 92.43 |
| 3 | Post-decoding character majority | 63.33 | 89.67 | 10.18 | 82.43 | 90.73 |
| 3 | Post-decoding sequence majority | 62.50 | 88.95 | 10.60 | 80.77 | 90.23 |
| 5 | Probability average | 71.20 | 91.56 | 7.65 | 86.97 | 92.80 |
| 5 | Product/log-probability | 73.80 | 92.42 | 6.97 | 88.07 | 93.67 |
| 5 | Post-decoding character majority | 69.37 | 91.66 | 8.27 | 85.83 | 93.03 |
| 5 | Post-decoding sequence majority | 67.13 | 90.33 | 9.30 | 83.40 | 91.43 |

The product-rule results reproduce the submitted manuscript values after rounding:
68.6% at F=3 and 73.8% at F=5.

### Interpretation for the revision

The reviewer's alignment concern is valid conceptually: CTC timesteps are latent
alignment positions, and corresponding pre-collapse timesteps from different
frames are not guaranteed to encode exactly the same alignment. We therefore
should not claim that pre-collapse temporal multiplication is theoretically exact.

However, the controlled comparison shows that the submitted product-rule pooling
is empirically stronger on this SVTRv2 checkpoint than the tested post-decoding
alternatives. At F=5 it obtains 73.80%, compared with 69.37% for character-position
majority and 67.13% for whole-sequence majority.

The revised manuscript should therefore:
1. describe BJP as product-rule / sum-log-probability pooling rather than a strict
   Bayesian posterior;
2. state that frame conditional independence is an approximation because tracklet
   frames are correlated;
3. state explicitly that pre-collapse CTC pooling assumes timestep correspondence;
4. report the post-decoding control experiment as evidence that, on this dataset,
   the submitted pooling rule nevertheless gives the strongest measured CTC result.

This completes the experimental component of Reviewer 2.3. The manuscript wording
and response-letter text still need to be revised.
