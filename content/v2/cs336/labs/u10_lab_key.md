# U10 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u10_lab_run.py`, outputs
must match exactly.

```
T1 budget 70B/1.4T: 5.88e+23 FLOPs, D/N=20.0
T1 budget 7B@20tpp: 5.88e+21 FLOPs
T2 best E on grid: 1.69 (RMS 0.0031)
T3 fitted a=0.340 (true 0.34)
T4 isoflop C=1e22: N*=8.32e+09, loss=2.874
T5 allocation: N=7.00e+10 D=1.40e+12 D/N=20.0
T6 transfer lr 256->1024: 7.5e-05
T7 residual RMS: 0.0031, max|resid|: 0.0074
T8 bootstrap CI: [0.336, 0.343]
T9 A loss 2.290 vs B loss 2.646, gap 0.36
T10 preregistered band [0.30, 0.38], observed 0.340: HELD
T11 7B @20 tpp: loss 3.003
T11 7B @200 tpp: loss 2.646
T12 transfer checks: form=pass(noise iid), noise=pass(CI narrow), range=FLAG(grid coarse)
```

Notes: T8 CI [0.336, 0.343] is the lab's 200-resample run (seed 2).
the lesson cites [0.337, 0.343] from compute_u10.py. Both are honest
Monte Carlo draws, the key records this run's values.
