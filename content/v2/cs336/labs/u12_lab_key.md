# U12 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u12_lab_run.py`, outputs
must match exactly.

```
T1 loss 2.3 -> ppl 9.97
T1 loss 1.7 -> ppl 5.47
T2 overlap 100.0%, canary recall 1.00
T3 bits/byte A 0.687, B 0.605: B wins
T4 20 decisions at 5% FP: P(>=1 bogus)=0.64
T5 math: 0.80 +- 0.124
T5 code: 0.65 +- 0.121
T6 format deltas: 0.717/0.723/0.657, max-min=0.066
T7 config: model u-test, items sha256:abc, T=0, seed 0
T8 agree 0.75, kappa 0.50
T9 seed range 0.034
T10 gap 0.15, gap SE 0.088: not significant
T11 gaps: input len 40 vs 400, format MC vs free, grading auto vs user
T12 A: $0.0022 per correct
T12 B: $0.0056 per correct
```

Note: T8 uses the lab's own 4-item toy (agree 0.75, kappa 0.50).
the lesson C08 cites compute_u12.py's 100-item toy (0.83, 0.63).
Both are labeled toys, the key records this run's values.
