# U09 lab key , execution-verified outputs

Produced by running `python3 u09_lab_run.py` (numpy, CPU, 2026-10-06,
seed 0).

```
T1 column split sums to unsharded: True
T1 TP all-reduce per layer: 67.1 MB
T2 layers/GPU: 8
T2 total stage-times: 35 (m+p-1)
T3 bubble p=4,m=4: 42.9%, p=4,m=32: 8.6%, p=8,m=8: 46.7%
T3 p=1 -> 0%: True
T4 1F1B bubble ~9.4%, in-flight ~4 microbatches
T5 valid triples: [(8, 2, 4), (4, 4, 4), (2, 8, 4)]
T6 TP 2.1 GB/step (64 msgs), DP 14 GB/step (1 msg)
T7 naive balance max/mean: 1.14
T7 cost-based split beats naive: True
T8 activation memory full 100 vs checkpointed 32 (compute +33%)
T9 TP=4 per-token latency model: 2.5 ms
T9 PP=4 per-token latency model: 16.0 ms
T10 (8,2,4): params 0.875 GB/GPU, optimizer 14.0 GB/GPU
T10 (4,4,4): params 0.875 GB/GPU, optimizer 14.0 GB/GPU
T11 imbalance: stage-time skew, bubble: fill/drain gaps, tier: comm-dominated step
T12 (8,2,4)/64 passes: True
T12 (16,2,2)/64 fails tp<=node: True
```

Notes:

- T4, T8, T9, T11 are stated-model checks, not derived quantities,
  the lab verifies the arithmetic around them.
- T6's DP row is the 7B fp16 grads (14 GB) from U08.
