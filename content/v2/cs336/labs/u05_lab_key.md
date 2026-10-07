# U05 lab key , execution-verified outputs

Produced by running `python3 u05_lab_run.py` (numpy, CPU, 2026-10-06,
seed 0).

```
T1 stable CE: 0.4076, naive: nan
T1 remediation [2,1,0] class 0: 0.4076 (expect 0.4076)
T2 inputs: [[5, 6, 7]] labels: [[6, 7, 8]]
T3 AdamW w: [0.99899, -0.498995]
T3 t=1 mhat==g: True vhat==g^2: True
T4 AdamW decay-only update -0.000500, L2-entangled differs: True
T6 norm 5.0 -> 1.0000, direction preserved (cos=1.000000)
T6 no-op when norm < c: True
T7 lr(0)=0.0e+00 lr(warm)=3.0e-04 lr(total)=3.0e-05
T7 WSD stable phase lr(500)=3.0e-04
T8 accumulation equivalence: True
T8 missing /a scales by 2: True
T9 SGD+m w: 0.999500, Lion w: 0.999000 (Lion step is +-lr)
T10 width 128: std-init 0.0884, muP hidden lr mult 1.0000, muP out init std 0.00781
T10 width 512: std-init 0.0442, muP hidden lr mult 0.2500, muP out init std 0.00195
T10 width 2048: std-init 0.0221, muP hidden lr mult 0.0625, muP out init std 0.00049
T11 RNG restore exact: True
T12 max rel err: 1.88e-10 (fp64)
```

Notes:

- T1's naive line prints RuntimeWarnings before the nan, the key line
  is the printed output.
- T4: the "L2-entangled differs" assertion compares the decay-only
  AdamW update against the full L2-style gradient step, which the lab
  implements directly.
- T5 (bias correction) is folded into T3's t=1 exactness checks.
