# U17 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u17_lab_run.py`, outputs
must match exactly.

```
T1 256 img + 512 text = 768 (33.3% image)
T2 stream: 200 text + 256 img + 300 text = 756 tokens
T3 data pipe 0.010 GB/s at 2.5 Mtok/s toy
T4 broadcast every 10 updates. KL under 0.01
T5 util 83.3%, backlog 1200/hr
T6 v2=ab12 vs v3=cd34: mismatch fires before GPUs burn
T7 controls: filter, monitor, 1% sample, kill at -2pts
T8 final artifact id: ea98188a5dad105d
T9 checkpoint 8.4 s, expected waste 500 steps at interval 1000
T10 5/6 invariants pass, template fails -> reject
T11 backward 38.5% of 143 ms. 2x -> 116 ms (19.2%)
T12 effects +0.10/+0.06/+0.02/-0.03: sign flips at 64x
```
