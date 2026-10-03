# Reviewer 1.1 — computational-efficiency summary

F=1 latency is a single-frame model forward. The primary F=5 tracklet latency follows test.py: the five observations are processed as one batch of five and product-rule / sum-log-probability fusion is included. A supplementary serial-five-frame latency reports five consecutive batch-size-one forwards without cross-frame fusion. Disk I/O, preprocessing, host-to-device transfer, and final string/CTC decoding are excluded.

| Model | Params (M) | F1 GFLOPs | F1 latency (ms) | F1 peak mem (MiB) | F5 GFLOPs | F5 tracklet latency (ms) | F5 peak mem (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | 16.047 | 45.555 | 12.110 +/- 1.015 | 117.4 | 227.773 | 12.740 +/- 0.859 | 209.6 |
| SVTRv2 | 4.167 | 1.324 | 7.985 +/- 0.859 | 33.9 | 6.619 | 8.659 +/- 2.548 | 36.1 |
| OTE | 20.065 | 3.277 | 18.889 +/- 4.256 | 124.9 | 16.386 | 17.930 +/- 1.493 | 126.4 |
| LISTER | 5.991 | 1.678 | 13.800 +/- 0.260 | 44.5 | 8.390 | 14.722 +/- 1.473 | 48.2 |
| IGTR | 31.097 | 3.951 | 16.393 +/- 3.727 | 192.2 | 19.756 | 15.950 +/- 1.848 | 196.0 |
| CPPD | 26.892 | 3.956 | 14.866 +/- 1.403 | 165.8 | 19.779 | 15.737 +/- 0.695 | 167.4 |
| MDiff4STR | 31.906 | 5.930 | 17.803 +/- 5.278 | 202.2 | 29.652 | 18.825 +/- 5.604 | 204.7 |

## Shared benchmark conditions

- Device: Quadro RTX 8000
- PyTorch: 2.6.0+cu124
- CUDA runtime: 12.4
- Precision: fp16
- Warm-up iterations: 30
- Timed iterations: 100
- cuDNN benchmark: True
- cuDNN deterministic: False
- Input: real LRLPR-26 TEST_3k tracklets resized/normalized by the repository validation wrapper.
- Latency numbers are synchronous end-to-end device execution for the defined tensor operation, not throughput measurements.
