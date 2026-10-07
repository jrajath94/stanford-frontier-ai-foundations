# U06 lab key , execution-verified outputs

Produced by running `python3 u06_lab_run.py` (numpy, CPU, 2026-10-06).
Reference device assumptions are stated in the lesson.

```
T1 active threads: 221184 (2.0K per SM)
T1 10 blocks -> idle SMs: 98
T2 GQA-8 fit: 1220703 tokens
T2 MQA fit: 9765625 tokens (8x)
T3 knee: 156 FLOP/byte
T3 scores: 2.0 TFLOP/s memory-bound
T3 gemm: 312.0 TFLOP/s compute-bound
T4 blocks/SM at 32 regs, 48KB smem: 2, at 32 regs, 8KB smem: 8
T5 warmup excluded, median of rest: 3.1 ms
T6 M=1 intensity 1.00 (memory), M=4096 intensity 1365 (compute)
T7 MFU 8.1%, HFU with 30% remat 10.5%
T8 7B all-reduce over fast tier: 0.02 s
T8 7B all-reduce over slow tier: 0.07 s
T8 70B over slow tier: 1.40 s
T10 bf16 dense: 312 TFLOP/s
T10 fp8 dense: 624 TFLOP/s
T11 clock 1.4->1.0 GHz: throughput factor 0.71 (drop 29%)
T11 detection: baseline drift 8% -> flag: True
T12 sane profile passes: True
T12 120% roofline fails: True
```

Notes:

- T9 (vendor-claim template) is a written exercise, no numeric key.
- T2's dh=64 assumes 32 query heads at d=2048.
- T7's HFU adds 30 percent rematerialization FLOPs to the numerator.
