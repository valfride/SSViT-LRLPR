# Temporal-fusion and frame-order protocol check (2026-10-10)

This note records source-level checks for the OJ-ITS revision. It does **not**
replace a full 3,000-track re-evaluation.

## Evaluator behavior and the historical results

- The paper's multi-frame test results use test.py with --fusion bayes.
  This sums per-frame log-probabilities for each native output timestep/slot
  before greedy decoding (SVTRv2-AR is handled through shared-prefix BJP).
- The common SROCR_VAL training-validation routine instead averages
  per-frame softmax outputs and selects checkpoints based on validation
  sequence accuracy. Selection uses the validation split, not test results.
  For SVTRv2 CTC, its inference head already returns probabilities, so this
  routine applies an additional softmax during checkpoint selection. That
  historical selection rule is documented but not silently changed here.
- SVTRv2's classifier returns softmax probabilities in eval mode. The legacy
  logit_average branch therefore averaged those probabilities directly.
  This is not an independent raw-logit control and the column was removed from
  the manuscript.
- On 2026-10-10 the evaluation-only logit_average branch was corrected to
  average log-probabilities when a model emits probabilities, preserving the
  expected product-rule argmax (up to numerical effects). The default fusion
  option was changed from logit_average to bayes. Historical results were
  generated with an explicit --fusion and are not changed by this default.
- On 2026-10-10 the CTC **post-decoding voting** path was corrected: a CTC
  probability tensor is converted to log-probabilities before entering
  ctc_greedy_decoder(), which itself performs softmax. The old path had applied
  softmax twice and distorted per-frame confidence-based tie-breaking.
  Frame-level argmax predictions are otherwise expected to remain unchanged;
  majority-voting results might change when there are ties. Re-run F=3/F=5
  CTC sequence-majority and character-majority controls **before submission**.
  Keep newly generated results separate from historical artifacts.
- No weights were retrained or changed by these evaluation-only corrections.

## Non-chronological metadata order: confirmed for 100 saved tracks

Source: reviewer1/01_efficiency/results/benchmark_inputs.json.

Its 100 sample tracklets all have the recorded LR order:

    lr-005.jpg, lr-001.jpg, lr-002.jpg, lr-003.jpg, lr-004.jpg

The VSR_Sequence_collate_fn wrapper uses the existing order of metadata
entries within a track without sorting by frame number or timestamp.
Consequently the historical evaluator uses LR #5 at F=1 and LR #5, #1, #2 at
F=3 on these sampled tracks. If numbered filenames correspond to actual
capture order, these are not the first chronological F=1/F=3 observations.
The F=5 fused result uses all five and is order-independent in exact
arithmetic for the product rule.

Do **not** silently sort images in the wrapper and retain the reported F=1
and F=3 results. Sorting would create a new evaluation protocol and requires
new F=1/F=3 measurements (and corresponding paper tables/plots), including
the controlled ablations and every baseline if keeping a common protocol.

### Read-only audit for all 3,000 tracks

Run on the workstation holding the test LMDB metadata:

    python3 experiments/revision_eval/shared/scripts/audit_frame_order.py \
      --metadata /path/to/CompetitionDataset_LMDB_TEST_3k/metadata.pkl \
      --split test \
      --output experiments/revision_eval/shared/notes/full_frame_order_audit.json

To fail when any metadata order disagrees with ascending LR frame indices,
add --require-chronological.

The check can also be repeated from the committed 100-track benchmark manifest:

    python3 experiments/revision_eval/shared/scripts/audit_frame_order.py \
      --benchmark-inputs experiments/revision_eval/reviewer1/01_efficiency/results/benchmark_inputs.json

The script assumes lr-NNN names encode chronological frame indices. Without
capture timestamps it cannot independently establish the physical capture
order. The 100-track evidence does not prove the count for all 3,000 tracks.

## Manuscript consistency

The manuscript revision now says frames are selected from the dataset metadata
order rather than claiming they are the first chronologically ordered frames.
It also removes the redundant/mislabeled logit-average control and states
that the uncorrected product of correlated frame posteriors is a heuristic,
not an exact calibrated Bayesian posterior.

## Corrected CTC voting results verified on 3,000 test tracklets

The user ran the corrected evaluator from revision/reviewer-eval-tooling on
daugman, using the frozen SVTRv2 checkpoint, F=3/5, and sequence-/character-
majority controls, with results written to a **new** directory:
`experiments/revision_eval/reviewer2/03_temporal_fusion/results/svtrv2_ctc_corrected_votes/`.

| Frames | Fusion | Original RR (%) | Corrected RR (%) | Correct/3000 before | Correct/3000 after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | majority | 62.5000 | 62.5000 | 1875 | 1875 |
| 3 | char_majority | 63.3333 | 63.3667 | 1900 | 1901 |
| 5 | majority | 67.1333 | 67.2667 | 2014 | 2018 |
| 5 | char_majority | 69.3667 | 69.2667 | 2081 | 2078 |

Corrected output was reported from the four JSON metrics objects:
F3_majority.json, F3_char_majority.json, F5_majority.json,
F5_char_majority.json. This audit note records the values provided from the
local run; the JSON files themselves have not been uploaded/checked through
this GitHub connection.

The revised paper now reports the corrected one-decimal controls:
F=3 sequence/character majority 62.5/63.4, F=5 67.3/69.3.
Both BJP results are unchanged (68.6 at F=3, 73.8 at F=5), so the experimental
conclusion is unchanged: product-rule fusion gives the best RR among the
tested alternatives. No checkpoint was retrained.

## Remaining protocol considerations

1. Retain the historical metadata-order F=1/F=3 evaluation protocol and
   document it consistently; reordering the **same** F frames does not affect
   BJP fusion, but replacing frames with chronological-first F frames changes
   the selected input subset and would require rerunning those evaluations.
2. The full 3,000-track frame-order audit remains optional for a complete
   metadata provenance check and has not been run as part of this note.
3. Preserve submitted checkpoints and original benchmark/fusion artifacts;
   never overwrite them with recalculations lacking separate provenance.
