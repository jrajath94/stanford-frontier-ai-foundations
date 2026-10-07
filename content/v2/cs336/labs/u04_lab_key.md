# U04 lab key , execution-verified outputs

Produced by running `python3 u04_lab_run.py` (numpy, CPU, 2026-10-06,
seed 0).

```
T1 allowed per row: [1, 2, 3, 3, 3, 3, 3, 3] w=T recovers causal: True
T1 FLOP ratio 8.0
T2 g=h equals MHA: True
T2 g=8 cache 50.3 MB
T2 g=2 cache 12.6 MB
T2 g=1 cache 6.3 MB
T3 dc=d identity reconstructs: True
T3 MLA dc=128 cache 12.6 MB
T4 equivalence dev 2.78e-16, state shape (4, 4)
T5 h = [1.0, 0.9, 1.81] recurrence==conv: True
T6 alpha=beta=1 reduces to linear: True
T6 causal (early outputs unchanged): True
T7 dense==E1k1 structural: True
T7 total 25165824 active 6291456 ratio 4.0
T8 loads: [1, 6, 4, 3, 6, 5, 5, 2] k per token: True
T9 k=E equals weighted sum: True
T9 k=1 picks argmax: True
T10 cap 5 loads [3, 5, 2, 5, 5, 3, 2, 2] dropped 5, conserved: True
T11 toy aux 2.3745, uniform aux 2.0000 (== k)
T12 total 25165824 active 6291456 ratio 4.0
T12 k=E: active==total: True
T12 E=1,k=1 equals dense 3145728: True
```

Notes:

- T8/T10/T11 use the lab's own rng stream, so loads and drop counts
  differ from the lesson's `compute_u04.py` run, the invariants (k per
  token, conservation, uniform aux = k) are what the lab grades.
- T1 FLOP ratio 8.0 = T/w = 1024/128 exactly.
- T4 state shape (4,4) is independent of the sequence length 6.
- T5 hand check: h_0 = 1, h_1 = 0.9, h_2 = 0.9*0.9 + 1 = 1.81.
