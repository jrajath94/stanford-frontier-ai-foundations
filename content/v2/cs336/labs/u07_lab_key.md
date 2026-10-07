# U07 lab key , execution-verified outputs

Produced by running `python3 u07_lab_run.py` (numpy, CPU, 2026-10-06,
seed 0).

```
T1 Bc(100KB,4,64,2)=200, blocks for T=4096: 21
T2 online matches naive: True, block-1 matches: True
T3 tiled matches naive: True (max dev 3.33e-16)
T3 traffic: naive 537 MB, tiled 25.2 MB
T4 tiled-style dQ vs finite diff max rel err: 3.41e-07
T5 blocks 21, kept 3, skipped 18
T6 separate 101 MB vs fused 34 MB
T7 masked block add correct incl tail: True
T8 protocol: tiny hand case, random small vs naive, edge shapes, dtype variants
T10 Bc: 3 tiles->266, 4 tiles->200, 5 tiles->160, dh=128 4 tiles->100
T11 fp32-accum vs fp16-accum scores max dev: 3.77e-03
T12 rows sum to 1: True, convex hull: True, block-size independent: True
```

Notes:

- T9 (benchmark-claim template) is a written exercise, no numeric key.
- T4 uses central differences with eps=1e-6 in fp64.
- T11 compares bf16-matmul scores accumulated in fp16 versus fp32.
