# Reviewer 1.1 — Computational efficiency

This directory contains the common benchmarking protocol requested by Reviewer 1:

> Add an efficiency comparison (#params, FLOPs, memory, per-frame and
> per-tracklet latency) for your model and all baselines.

The benchmark covers the proposed model and the six principal baselines used in
the manuscript:

- Ours
- SVTRv2
- OTE
- LISTER
- IGTR
- CPPD
- MDiff4STR

## Protocol

All models are benchmarked on the same physical GPU, in separate fresh Python
processes, using their validation-selected checkpoint. Baseline experiment roots contain
timestamped run subdirectories (for example,
`experiments/baselines/CPPD_BASELINE/CPPD_BASELINE_<timestamp>/`), so the
worker searches recursively for `*acc_*.pth` and selects the highest encoded
validation accuracy. It falls back to `last.pth` only when exactly one such
file exists.

The runner prepares one fixed set of real LRLPR-26 TEST_3k tracklets with the
repository validation wrapper. The exact same preprocessed tensors are reused
for every architecture.

Default settings:

- input size: 3 x 32 x 96;
- real TEST_3k inputs;
- 100 fixed tracklets;
- 30 warm-up iterations;
- 100 timed iterations;
- batch size 1 for F=1;
- the five F=5 observations are flattened to a batch of 5, matching `test.py`;
- FP16 follows each model config when `--precision auto` is used;
- channels-last CUDA input layout, matching `test.py`;
- TF32 disabled explicitly;
- performance mode uses `cudnn.benchmark=True`;
- latency uses `time.perf_counter()` with CUDA synchronization immediately
  before and after every timed inference;
- each model runs in a fresh subprocess to avoid cross-model CUDA allocator
  contamination.

F=1 latency is one model forward pass.

Two F=5 timings are recorded. The primary deployment-oriented tracklet latency
runs five consecutive batch-size-one forwards and then applies product-rule /
sum-log-probability fusion. A second batched F=5 timing processes the five
observations in one batch of five, matching the current `test.py` evaluation
implementation. Reporting both avoids making the batched GPU result look like
five sequential frame inferences. Both exclude disk I/O, LMDB access, image
resize/normalization, host-to-device transfer, final Python string conversion,
and CTC collapse/string decoding.

Peak GPU memory is the maximum PyTorch allocated memory during inference and
includes model parameters and the active input tensor. F=1 and F=5 are measured
separately after the latency pool is released.

FLOPs are measured with PyTorch's `torch.profiler.profile(with_flops=True)`
using one common backend for every architecture. PyTorch's experimental
`FlopCounterMode` is not used because it fails on the proposed model's
`torchvision.ops.DeformConv2d` path in the tested environment. When the
profiler assigns zero FLOPs to that custom deformable-convolution operator, the
standard convolution arithmetic count is added analytically (multiply + add =
two FLOPs); bilinear sampling overhead is not included. The JSON records the
backend and this convention. For models with data-dependent inference paths,
FLOPs are measured over multiple real samples and summarized rather than
silently assuming a fixed graph.

## Checkpoint paths

The default paths follow the repository's documented baseline layout:

```text
Ours     experiments/revision_eval/submitted_model/
SVTRv2   experiments/baselines/SVTRV2_BASELINE/
OTE      experiments/baselines/OTE_BASELINE/
LISTER   experiments/baselines/LISTER_BASELINE/
IGTR     experiments/baselines/IGTR_BASELINE/
CPPD     experiments/baselines/CPPD_BASELINE/
MDiff4STR experiments/baselines/MDIFF_BASELINE/
```

Weights remain local/ignored; only benchmark results and manifests should be
committed.

## 1. Pull and syntax-check

```bash
git pull --ff-only

python3 -m py_compile \
  experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_one.py \
  experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_efficiency.py
```

## 2. Preflight

Inspect the planned commands without running GPU work:

```bash
python3 \
  experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_efficiency.py \
  --gpu 0 \
  --dry-run \
  --keep-going
```

The dry-run checks the expected config/checkpoint locations. The real input cache
is not created in dry-run mode.

## 3. Smoke-test the proposed model

Run the proposed model first with a short timing loop:

```bash
python3 \
  experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_efficiency.py \
  --models ours \
  --gpu 0 \
  --sample-count 20 \
  --warmup 10 \
  --iterations 20 \
  --flop-samples 3 \
  --rebuild-input-cache \
  --strict
```

This also creates the reusable real-input cache under:

```text
experiments/revision_eval/reviewer1/01_efficiency/results/benchmark_inputs.pt
```

The `.pt` cache is a generated local artifact and must not be committed.

## 4. Publication run

After the smoke test succeeds:

```bash
python3 \
  experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_efficiency.py \
  --gpu 0 \
  --sample-count 100 \
  --warmup 30 \
  --iterations 100 \
  --flop-samples 10 \
  --rebuild-input-cache \
  --keep-going \
  --strict
```

If GPU 0 is not the GPU intended for the paper, replace `--gpu 0` with the
chosen physical device and use that same GPU for every model.

Do not mix results from different GPU models in the final table.

## Outputs

Per-model machine-readable results:

```text
results/
  ours.json
  svtrv2.json
  ote.json
  lister.json
  igtr.json
  cppd.json
  mdiff.json
```

Aggregated outputs:

```text
results/efficiency_summary.csv
results/efficiency_summary.md
results/benchmark_manifest.json
results/benchmark_inputs.json
```

The generated `benchmark_inputs.json` records dataset provenance, tensor shape,
track selection, and the SHA-256 of the local tensor cache.

## Before committing results

Verify that no model weights or generated input cache are staged:

```bash
git status --short

git diff --cached --name-only | grep -E '\.(pth|pt|ckpt)$'
```

The second command should print nothing.

Then stage only the text/JSON/CSV benchmark evidence:

```bash
git add \
  experiments/revision_eval/reviewer1/01_efficiency/results/*.json \
  experiments/revision_eval/reviewer1/01_efficiency/results/*.csv \
  experiments/revision_eval/reviewer1/01_efficiency/results/*.md

git status --short
```

Do not add `benchmark_inputs.pt`.


## If your local checkpoint folders differ

The repository documentation records the historical default baseline paths, but
model weights are intentionally not tracked by Git. On a workstation where the
runs were stored under different names, locate the actual retained checkpoints:

```bash
find experiments -type f \( -name '*acc_*.pth' -o -name 'last.pth' \) \
  | grep -Ei 'SVTR|OTE|LISTER|IGTR|CPPD|MDIFF' \
  | sort
```

Do not copy checkpoints into Git just to satisfy the benchmark runner. Instead,
override the local path explicitly. The option is repeatable:

```bash
python3 \
  experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_efficiency.py \
  --gpu 0 \
  --checkpoint-override svtrv2=/actual/path/to/student_weights \
  --checkpoint-override ote=/actual/path/to/student_weights \
  --checkpoint-override lister=/actual/path/to/student_weights \
  --checkpoint-override igtr=/actual/path/to/student_weights \
  --checkpoint-override cppd=/actual/path/to/student_weights \
  --checkpoint-override mdiff=/actual/path/to/student_weights \
  --dry-run \
  --keep-going
```

Valid override keys are `ours`, `svtrv2`, `ote`, `lister`, `igtr`,
`cppd`, and `mdiff`. The resolved paths are recorded in
`benchmark_manifest.json`.

If FLOP counting is unsupported by one operation/model, the worker now prints the
profiler error explicitly and stores it in the per-model JSON rather than silently
leaving the table cell blank.
