# OJ-ITS Major Revision Execution Plan

Last status review: 2026-10-03.

This checklist separates completed experimental/audit work from manuscript integration.
Reviewer-response wording and manuscript edits remain open until they are committed to
the paper repository.

## Frozen submitted model
- [x] Identify released EMA checkpoint
- [x] Verify checkpoint SHA-256
- [x] Reproduce F=1 result
- [x] Reproduce F=3 result
- [x] Reproduce F=5 result
- [x] Preserve frozen submitted-model configuration and provenance

## Reviewer 1

### R1.1 — Computational efficiency
- [x] Define one common benchmarking protocol and hardware/software conditions
- [x] Parameter count
- [x] FLOPs/MACs
- [x] Peak inference GPU memory
- [x] F=1 per-frame latency
- [x] F=5 per-tracklet latency
- [x] Run the same protocol on all principal baselines
- [x] Prepare machine-readable and Markdown efficiency tables
- [x] Verify strict checkpoint loading for all seven models
- [x] Record Quadro RTX 8000 / PyTorch 2.6.0+cu124 / CUDA 12.4 conditions
- [ ] Write efficiency interpretation into manuscript/response

### R1.2 — Cross-dataset evaluation
- [ ] Select one compatible external dataset
- [ ] Define label-normalization and sample-selection protocol
- [ ] Prepare the external dataset for the existing evaluator
- [ ] Run zero-shot evaluation of the frozen submitted model
- [ ] Report exact-match, character accuracy, and CER
- [ ] Document domain/annotation limitations

### R1.3 — Decoder/reproducibility specification
- [x] Audit complete feature-extractor topology
- [x] Confirm SFB locations
- [x] Confirm Restormer/PixelShuffle order
- [x] Fully document deformable projection details
- [x] Confirm 3 decoder layers
- [x] Confirm d_model=384
- [x] Confirm 12 attention heads
- [x] Confirm 1536-dimensional FFN
- [x] Confirm visual-token organization
- [x] Confirm seven learned output queries
- [x] Correct autoregressive interpretation: decoder is parallel/fixed-length
- [x] Confirm training-time query conditioning/token masking behavior
- [x] Confirm normalized cosine classifier
- [x] Write the audited decoder/architecture specification into the manuscript
- [ ] Add reviewer-response evidence/reference to implementation details

### R1.4 — Additional baseline / related work
- [x] Investigate SVTRv2-AR implementation and compatibility
- [x] Add an OpenOCR-derived SVTRv2-AR baseline with no external data/pretraining
- [x] Implement AR-aware BJP/product-rule decoding with one shared fused prefix
- [x] Batch all frame-conditioned AR decoder calls under the shared-prefix BJP rule
- [x] Lock and document the controlled SVTRv2-AR training protocol
- [x] Align the manuscript training/fusion description with the implementation
- [x] Re-run SVTRv2-AR smoke test after final protocol/BJP lock
- [x] Train the controlled SVTRv2-AR baseline
- [ ] Evaluate SVTRv2-AR at F=1/F=3/F=5 using BJP only
- [ ] Add SVTRv2-AR to the computational-efficiency comparison
- [ ] Integrate SVTRv2-AR results into manuscript/response
- [ ] Expand CCPD discussion in Related Work

## Reviewer 2

### R2.1 — Controlled ablation
- [x] Full model
- [x] -Restormer
- [x] -PixelShuffle
- [x] -SFB
- [x] Linear classifier instead of normalized cosine
- [x] Student instead of EMA
- [x] Evaluate all structural/classifier variants at F=1/F=3/F=5
- [x] Run three-seed controlled family (42, 123, 2026)
- [x] Run best-validation EMA-versus-student comparison
- [x] Preserve matched-final-epoch EMA analysis as supporting evidence

### R2.2 — Table consistency
- [x] Diagnose Table 1 / original Table 2 inconsistency
- [x] Generate a complete controlled full-model/ablation path under one protocol
- [x] Use identical F=1/F=3/F=5 evaluation protocol for controlled variants
- [x] Replace cumulative-only Table 2 in the manuscript
- [x] Explain explicitly how the revised ablation table relates to Table 1

### R2.3 — Temporal fusion
- [x] Product-rule / sum-log-probability F=3/F=5
- [x] Probability-average F=3/F=5
- [x] Logit-average F=3/F=5
- [x] Sequence-majority F=3/F=5
- [x] Add post-decoding character-majority control for CTC
- [x] Evaluate CTC baseline fusion behavior
- [x] Reproduce submitted SVTRv2 F=1/F=3/F=5 values with the canonical checkpoint
- [x] Establish product-rule/logit-average equivalence for fixed-position softmax up to numerical effects
- [ ] Rename/reframe BJP as product-rule / sum-log-probability temporal pooling
- [ ] Rewrite conditional-independence and CTC-alignment claims
- [ ] Add alternative-fusion control results to manuscript/response

### R2.4 — Statistical validation
- [x] Make training seed configurable
- [x] Run three independent seeds
- [x] Report mean +/- sample standard deviation
- [x] Paired track-level bootstrap 95% confidence intervals
- [x] Hierarchical paired-bootstrap 95% confidence intervals across seeds
- [x] Exact paired McNemar tests
- [x] Holm correction within comparison families
- [x] Analyze best-vs-best EMA statistics
- [x] Analyze matched-final-epoch EMA statistics
- [ ] Integrate statistics and appropriately cautious claims into manuscript/response

### R2.5 — Reproducibility
- [x] Audit exact optimizer configuration
- [x] Audit exact scheduler configuration
- [x] Verify LR-warmup behavior
- [x] Audit geometric/data augmentation protocol
- [x] Audit training length and stopping rule
- [x] Audit EMA warmup/update implementation
- [x] Record tested software environment in repository documentation
- [x] Make evaluation deterministic (cuDNN benchmark off; deterministic kernels on)
- [x] Verify repeated inference produces byte-identical prediction CSVs
- [x] Record evaluation reproducibility metadata in metrics JSON
- [x] Record final hardware details used for reported efficiency experiments
- [x] Correct EMA description in manuscript
- [ ] Consolidate audited training/architecture details into manuscript and response

## Editor-in-Chief
- [ ] Search recent OJ-ITS literature
- [ ] Select genuinely relevant recent OJ-ITS papers
- [ ] Integrate selected citations into Related Work
- [ ] Mention the added OJ-ITS literature explicitly in the response letter

## Remaining experiment priority
1. [x] R1.1 computational-efficiency benchmark for proposed model and principal baselines
2. [ ] R1.2 one external zero-shot cross-dataset evaluation
3. [ ] R1.4 SVTRv2-AR feasibility check and, if fair/compatible, baseline experiment
4. [ ] Freeze all remaining experimental tables before manuscript editing

No additional Reviewer-2 training experiment is currently planned unless a new
consistency issue appears during manuscript integration.

## Final revision
- [ ] Update architecture figure where needed
- [ ] Rewrite Methods using audited implementation details
- [x] Replace ablation table
- [ ] Add statistical reporting
- [ ] Add temporal-fusion control table/discussion
- [ ] Add efficiency table
- [ ] Add external-evaluation table
- [ ] Update Related Work, including recent OJ-ITS and CCPD discussion
- [ ] Write point-by-point response letter
- [ ] Produce unmarked revised manuscript
- [ ] Produce optional highlighted-changes manuscript
- [ ] Final reproducibility/consistency audit
