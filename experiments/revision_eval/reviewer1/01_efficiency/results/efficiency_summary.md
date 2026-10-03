# Reviewer 1.1 — computational-efficiency summary

F=1 latency is a single-frame model forward. The primary F=5 tracklet latency is deployment-oriented: five consecutive batch-size-one forwards plus product-rule / sum-log-probability fusion. A separate batched F=5 latency is also reported because test.py evaluates the five observations together as a batch of five. Disk I/O, preprocessing, host-to-device transfer, and final string/CTC decoding are excluded.

| Model | Params (M) | F1 GFLOPs | F1 latency (ms) | F5 seq. GFLOPs | F5 seq. latency (ms) | F5 batched latency (ms) | F1 peak mem (MiB) | F5 seq. peak mem (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours | 16.047 | 45.555 | 12.110 +/- 1.015 | 227.773 | 52.904 +/- 7.797 | 12.740 +/- 0.859 | 117.4 | 117.6 |
| SVTRv2 | 4.167 | 1.324 | 7.985 +/- 0.859 | 6.619 | 40.296 +/- 5.342 | 8.659 +/- 2.548 | 33.9 | 34.0 |
| OTE | 20.065 | 3.277 | 18.889 +/- 4.256 | 16.386 | 90.487 +/- 11.770 | 17.930 +/- 1.493 | 124.9 | 125.1 |
| LISTER | 5.991 | 1.678 | 14.815 +/- 0.652 | 8.390 | 73.710 +/- 9.727 | 14.679 +/- 0.740 | 44.5 | 44.7 |
| IGTR | 31.097 | 3.951 | 16.393 +/- 3.727 | 19.756 | 75.569 +/- 9.319 | 15.950 +/- 1.848 | 192.2 | 192.3 |
| CPPD | 26.892 | 3.956 | 14.866 +/- 1.403 | 19.779 | 67.642 +/- 8.826 | 15.737 +/- 0.695 | 165.8 | 166.0 |
| MDiff4STR | 31.906 | 5.930 | 17.803 +/- 5.278 | 29.652 | 93.114 +/- 16.080 | 18.825 +/- 5.604 | 202.2 | 202.3 |

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
