# U11 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u11_lab_run.py`, outputs
must match exactly.

```
T1 decode AI 1.0 FLOP/byte, prefill(2048) ~2048 FLOP/byte
T1 weight bytes per decode token: 14.0 GB
T2 per token: 131072 bytes (128.0 KB)
T2 b=16 T=4096: 8.59 GB
T3 unshared 2.62 GB, shared 0.26 GB, saving 2.36 GB
T4 static 800 steps, continuous 775 slot-steps
T5 contiguous waste 50.0%, paged(block=16) 0.0%
T6 E=2.53, speedup=1.95x
T7 accept probs: [1.  0.6]
T8 temp=1.0 top-1 0.644
T8 top-p=0.9 keeps 3
T9 turn reuse: prefill 1600 -> 100 token-equiv, saves 192 MB
T10 fp16 14.0 GB, int8 7.0 GB, int4 3.50 GB
T11 TTFT mean 120 ms, p99 533 ms
T12 10 RPS / 2 slots = 5 GPUs at mean load (+headroom)
```
