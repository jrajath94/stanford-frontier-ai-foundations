# U14 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u14_lab_run.py`, outputs
must match exactly.

```
T1 t=0.5: keep good 0.94, keep bad 0.06
T2 posture: recall-first, safety classes need recall>0.99 on probes
T3 unique 5840, dup rate 41.6%
T4 best k=6 FPR 0.0216
T5 s=0.8 detect prob 1.000
T6 13-gram index, drop docs with >80% eval n-gram overlap
T7 multiplier tgt/src, code 1.67
T8 argmin w=1.0 L=2.500
T9 1M generated, 80% pass check, 800k at 10% mix weight
T10 audit keep-rate per domain at fixed t, spread flags bias
T11 deltas: dedup +0.06, filter +0.03, reweight +0.01
T12 recall at 1/2/5/10: 0.18/0.34/0.61/0.78
```
