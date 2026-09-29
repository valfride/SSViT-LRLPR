# OJ-ITS Major Revision Execution Plan

## Frozen submitted model
- [x] Identify released EMA checkpoint
- [x] Verify checkpoint SHA-256
- [x] Reproduce F=1 result
- [x] Reproduce F=3 result
- [x] Reproduce F=5 result

## Reviewer 1

### R1.1 — Computational efficiency
- [ ] Parameter count
- [ ] FLOPs/MACs
- [ ] Peak inference GPU memory
- [ ] F=1 latency
- [ ] F=5 tracklet latency
- [ ] Run same protocol on principal baselines
- [ ] Prepare efficiency table

### R1.2 — Cross-dataset evaluation
- [ ] Select compatible external dataset
- [ ] Define label-normalization protocol
- [ ] Run zero-shot evaluation
- [ ] Report exact-match and character accuracy

### R1.3 — Decoder/reproducibility specification
- [ ] Confirm complete feature-extractor topology
- [ ] Confirm SFB locations
- [ ] Confirm Restormer/PixelShuffle order
- [ ] Document deformable projection
- [ ] Document 3 decoder layers
- [ ] Document d_model=384
- [ ] Document 12 attention heads
- [ ] Document 1536-dimensional FFN
- [ ] Document 16x48 = 768 visual tokens
- [ ] Document seven learned positional queries
- [ ] Correct autoregressive -> parallel decoding
- [ ] Document training-time query conditioning
- [ ] Document normalized cosine classifier

### R1.4 — Additional baseline / related work
- [ ] Investigate SVTRv2-AR implementation
- [ ] Add baseline if technically compatible
- [ ] Expand CCPD discussion

## Reviewer 2

### R2.1 — Controlled ablation
- [ ] Full model
- [ ] -Restormer
- [ ] -PixelShuffle
- [ ] -SFB
- [ ] Linear classifier instead of cosine
- [ ] Student instead of EMA
- [ ] Evaluate all at F=1/F=3/F=5

### R2.2 — Table consistency
- [ ] Reconcile Table 1 and Table 2
- [ ] Replace cumulative-only Table 2
- [ ] Use identical evaluation protocol

### R2.3 — Temporal fusion
- [x] BJP/product-rule F=3/F=5
- [x] Probability-average F=3
- [x] Probability-average F=5
- [x] Logit-average F=3/F=5
- [x] Sequence-majority F=3/F=5
- [x] Evaluate CTC baseline fusion behavior
- [ ] Rewrite BJP independence/alignment claims

### R2.4 — Statistical validation
- [x] Make training seed configurable
- [ ] Run at least 3 independent seeds
- [ ] Report mean +/- standard deviation
- [ ] Paired bootstrap 95% CI
- [ ] Paired significance analysis

### R2.5 — Reproducibility
- [ ] Exact optimizer configuration
- [ ] Exact scheduler configuration
- [ ] Verify absence/presence of LR warmup
- [ ] Exact augmentation parameters
- [ ] Exact training length/stopping rule
- [ ] Exact EMA update implementation
- [ ] Correct EMA description in manuscript
- [ ] Record software/hardware environment

## Editor-in-Chief
- [ ] Search recent OJ-ITS literature
- [ ] Select genuinely relevant papers
- [ ] Integrate citations into Related Work

## Final revision
- [ ] Update architecture figure
- [ ] Rewrite Methods
- [ ] Replace ablation table
- [ ] Add efficiency table
- [ ] Add external-evaluation table
- [ ] Update Related Work
- [ ] Write point-by-point response letter
- [ ] Final reproducibility audit
