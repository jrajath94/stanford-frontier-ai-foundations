# U02 lab key , execution-verified outputs

Produced by running `python3 u02_lab_run.py` (numpy, CPU, 2026-10-06).
Learner code must reproduce these values exactly. No wall-time claim
appears, all rates are arithmetic.

```
T1 contracts ok, wrong contract raises: True
T2 scores match: True mix match: True
T3 strides: (4, 24) predicted (4, 24): True
T3 view shares: True contig: False
T3 copy shares: False contig: True
T3 alias write visible: True
T4 fp16 MB: 134.2 terms: {'attn_in': 524288, 'qkv': 1572864, 'weights': 2097152, 'ffn': 4194304}
T4 2xB ratio 2.00 (expect 2.0), weights term 4x on 2xT: True
T5 sgd: 498.6 MB
T5 adam_fp32: 664.8 MB
T5 adam_mixed: 664.8 MB
T6 GFLOP 8.858 (expect 8.858), ffn share 72.7%
T6 attention passes FFN at T=6400
T7 mlp fwd 600 bwd 1200 ratio 2.00 in [1.8, 2.4]: True
T8 intensities: 170.7 204.8 1.0
T8 square scaling: 2.0
T8 M=1 flat: True
T9 rate 40.0 TFLOP/s cap 50.0 TFLOP/s MFU-vs-cap 0.80
T9 MFU>1 raises: True
T10 ledger: {'params+opt_MB': 664.8, 'act_MB': 134.2, 'total_MB': 799.0, 'margin15_MB': 918.9}
T10 fits 1GB: True fits 512MB: False
T11 fp16 sum 0.003906 fp32 sum 0.010000 true 0.01
T11 act fp16 134.2 MB fp32 268.4 MB ratio 2.00
T12 task 1: correctness evidence
T12 task 2: correctness evidence
T12 task 3: correctness evidence
T12 task 4: correctness evidence
T12 task 5: correctness evidence
T12 task 6: correctness evidence
T12 task 7: correctness evidence
T12 task 8: correctness evidence
T12 task 9: correctness evidence
T12 task 10: correctness evidence
T12 task 11: correctness evidence
T12 note: no wall-time claim appears, all rates are arithmetic.
```

Notes:

- T4 term counts are exact element counts for B=4, T=256, d=512,
  dff=2048, h=8, L=8. The 134.2 MB is an estimate (stated formula), not
  a framework measurement.
- T6 crossover T=6400 is the first multiple of 256 where the score term
  exceeds the FFN term, the exact crossover is T=6144 from
  2*T*d = 6*d*dff.
- T9 uses a synthetic step (26.6 GFLOP per layer-equivalent, 1000
  repeats at 40 TFLOP/s) to exercise the cap logic, the numbers are
  arithmetic, not measurements.
- T11 fp16 stalls at 0.00390625 because the fp16 ulp near 0.01 exceeds
  the 1e-6 increment, fp32 accumulates correctly.
