# Controlled-ablation historical-run audit

The historical experiment inventory was compared against
`experiments/revision_eval/submitted_model/config_snapshot.yaml`.

## Classification

| Historical run | Classification | Reason |
|---|---|---|
| `ce_sfb_13-05-2026-final` | Reusable submitted/full-model evidence | Matches the submitted training protocol fields in the audit and contains both student and ghost checkpoints, but it is a continuation from an earlier checkpoint rather than a fresh independent run. |
| `ablation_linear_head_26-05-2026` | Informative only; retrain for controlled table | It changes the classifier as desired, but uses `ema_warmup_epochs=15` instead of the submitted value 5. |
| `ce_nosfb_nocos_noema_26-05-2026` | Not a one-factor ablation | It simultaneously removes SFB, cosine classification, and EMA, and also uses a different EMA warm-up setting. |
| `baseline_ce_09-05-2026` | Not reusable for the requested controlled table | It removes SFB and EMA together, lacks the submitted early-stopping setting, and the audited directory contains no retained checkpoints. |
| `ablation_no_masking_26-05-2026` | Outside the requested reviewer factors | It studies token masking and also uses a different EMA warm-up setting. |

## Consequence

No historical run provides a clean, one-factor controlled comparison for all of
Restormer, PixelShuffle, SFB, cosine classification, and EMA under one identical
training protocol.

The revised Table 2 should therefore use a new controlled experiment family. The
recommended design is a common fresh-start protocol with explicit architecture
flags and multiple seeds. The frozen submitted checkpoint remains the Table 1
reproduction reference and should not be silently substituted by a newly selected
model.

For the EMA factor, no separate training run is required: evaluate student and
ghost weights from the same full-model training run at the same selected epoch /
checkpoint rule.
