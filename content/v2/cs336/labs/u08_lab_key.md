# U08 lab key , execution-verified outputs

Produced by running `python3 u08_lab_run.py` (numpy, CPU, 2026-10-06,
seed 0).

```
T1 DP average == big-batch grad: True
T1 missing /N scales update by 4: True
T2 ring 14GB/8/200GBs: 0.122 s
T2 ring -> 2B/bw at large N: 0.140 s
T3 ZeRO-0/8: 84.0 GB/GPU
T3 ZeRO-1/8: 35.0 GB/GPU
T3 ZeRO-2/8: 22.8 GB/GPU
T3 ZeRO-3/8: 10.5 GB/GPU
T4 sequential 96 ms vs overlapped 36 ms
T4 exposed when comm>compute: 8.0 ms
T5 scaled grad 6.14e-05 representable, unscaled matches fp32: True
T5 overflow -> skip step, not nan: True
T6 toy layer 1M params over 4 ranks: gathered 2.0 MB bf16, resident 0.5 MB
T7 option A: one 84 GB file, option B: 8 x 10.5 GB
T7 reshard 8->4 round-trips: True
T8 exposed(3,12)=0.0 ms, exposed(20,12)=8.0 ms
T9 efficiency(100,14,8)=89%
T9 straggler: step bound by max rank 20.0 s
T10 expected lost work per failure: 500 s
T11 shards disjoint: True, covering: True
T12 invariants: dp==bigbatch True, shards disjoint True, step counts agree True, reshard exact True
```

Notes:

- T12's invariant checks run on the simulator's known-good state,
  the missing-/N failure is asserted in T1.
- All byte math assumes fp16 params/grads and fp32 Adam states.
