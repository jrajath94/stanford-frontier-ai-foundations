# Interview bank U05: Quantization, sparsity, structured operators

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Write the four linear quantization formulas (s, z,
x_q, x_hat).

B2. State the in-range rounding error bound. What is the
second error source?

B3. Define per-tensor, per-channel, per-group
quantization. When is per-tensor unsafe?

B4. Contrast weight-only vs W8A8 quantization, and
static vs dynamic activation quantization.

B5. Name three calibration range-picking methods. Which
fails with outliers, and why?

B6. Define 2:4 sparsity. Why does the pattern matter
more than the sparsity count?

B7. Define magnitude pruning. Name two upgrades.

B8. Write the butterfly parameter count. Give the
n=1024 number vs dense.

## Deep ladder 1: quantization economics

L1a. Define: what are s and z?
L1b. Toy: range [-1,1], 4 bits. Compute s, z, and the
round-trip of 0.5.
L1c. Derive: prove the in-range error bound s/2.
L1d. Implement and complexity: write
quantize/dequantize and state the one-time cost.
L1e. Compare: per-tensor vs per-channel on rows
[-100,100] and [-1,1]. What happens to the quiet
channel under per-tensor?
L1f. Debug: many small values map to one level. Name
the cause and the first fix.
L1g. Critique: state the range-covers-data assumption
and construct the outlier counterexample.
L1h. Design: propose the clip-percentile sweep. Name
the expected curve shape.

## Deep ladder 2: sparsity reality

L2a. Define: what is realized kernel speed vs the FLOP
ratio?
L2b. Toy: dense 100 us, 50% sparse. Compute realized
time for gather overhead 45 us and 5 us.
L2c. Derive: show when unstructured sparsity gives
~1x despite halved FLOPs.
L2d. Implement and complexity: write `realized()` and
state what it does not model.
L2e. Compare: unstructured vs 2:4 vs block at 50%
sparsity on speed and hardware needs.
L2f. Debug: a 2:4 kernel runs at 1.0x vs dense. Name
two causes.
L2g. Critique: state the pattern-support assumption.
What happens on hardware without sparse tensor cores?
L2h. Design: propose the three-kernel measurement.
Name the expected ranges.

## Analytical exercises

A1. W has rows in [-100,100] and [-1,1], 4-bit.
Compute per-tensor s. Compute the quiet channel's
quantization MSE under per-tensor (values uniform in
[-1,1], all snap near one level: approximate MSE as
variance of the uniform = 1/3) vs per-channel
(s=0.1333, MSE ~ s^2/12). Give the ratio.

A2. Two layers, per-layer error 0.02 each, Lipschitz
constants [1.0, 2.0]. Compute the error bound. Then
recompute with the order swapped ([2.0, 1.0]). Which
layer should keep more precision, and why?

## Implementation / debug task

D1. The quantizer below looks right but the zero point
property fails (quantize(0) does not decode to 0).
Find the bug, fix it, and state the invariant.

```python
import numpy as np
def make(s, b):
    z = 0  # bug is here
    def q(x):
        return np.clip(np.round(x / s).astype(int) + z, 0, 2**b - 1)
    def dq(xq):
        return (xq - z) * s
    return q, dq
```

## Changed-constraint scenarios

S1. Constraint change: scales are free (zero storage,
zero compute). What is the optimal quantization
granularity now, and what limits it?

S2. Constraint change: the error budget is zero but
memory must halve. Compare int8 quantization,
2:4 sparsity, and butterfly replacement on
exactness, memory, and risk. Which do you pick?

## Research critique

R1. A paper claims "4-bit weights with no quality loss
on all tasks." List five audit questions, and for each
state the answer that would invalidate the claim.
