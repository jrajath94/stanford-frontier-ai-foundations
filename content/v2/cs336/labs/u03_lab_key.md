# U03 lab key , execution-verified outputs

Produced by running `python3 u03_lab_run.py` (numpy, CPU, 2026-10-06,
seed 0). Toy config: B=2, T=8, d=16, h=4, dh=4, dff=32, V=64.

```
T1 shape: (2, 2, 16) row0 match: True
T1 out-of-range raises: True
T2 shapes: (2, 8, 16) fused match: True
T3 rowsums max dev 2.22e-16
T3 uniform weights: True
T3 finite weights: True
T4 future max 0.00e+00 rowsums ok True pos0 self True
T4 after-softmax rowsums broken: True
T5 roundtrip: True heads disjoint: True
T5 bad h raises: True
T6 zero-block identity: True
T6 correction/input norm ratio 0.0103
T7 pre==input: True post==norm(input): True
T8 rmsnorm rms in [0.999,1.001]: True
T8 layernorm mean~0 std~1: True True
T8 zero row no NaN: True
T9 shapes: (2, 8, 16) (2, 8, 16) swiglu params 1536 relu48 params 1536
T10 norm dev 4.44e-16
T10 pos0 identity: True
T10 shift invariance dev 1.78e-15
T10 rope-on-V changes values: True
T11 logits shape: (2, 8, 64) rowsums ok: True
T11 tied (same object): True
T12 std 0.0625 target 0.0625 within 10%: True
T12 zero init identical heads: True
```

Notes:

- T3 uniform weights: with zero queries and keys, scores are all zero
  and softmax gives 1/4 = 0.25 per position.
- T4 demonstrates the after-softmax masking bug: zeroing future
  weights after normalization breaks the row sums.
- T6 ratio 0.0103: at init the correction is about 1 percent of the
  input norm, the healthy regime for residuals.
- T9: SwiGLU at dff=32 and ReLU at dff=48 both use 1536 parameters
  (the 2/3 rule).
- T10 shift invariance: the dot-product matrix has constant diagonals
  to 1.78e-15, proving relative-position dependence.
- T12: Xavier target for (256,256) is sqrt(2/512) = 0.0625.
