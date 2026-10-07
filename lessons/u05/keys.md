# keys.md, U05 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

s = (max-min)/(2^b-1). z = round(-min/s). x_q =
clamp(round(x/s)+z, 0, 2^b-1). x_hat = s*(x_q-z).

## E02

s = 0.1333, z = 8. x=0.5: x_q=12, x_hat=0.5333, error
0.0333. x=-0.25: round(-1.875)+8 = -2+8 = 6, x_hat =
0.1333*(-2) = -0.2667, error 0.0167.

## E03

The range is set by an outlier (or the tensor mixes
channels of very different ranges). The scale is too
coarse for the small values.

## E04

Forced z=0: zero decodes to -min... precisely,
dequantize(0) = -z*s is no longer 0. Padded zeros
become nonzero noise, sparse patterns break.

## E05

Hypothesis: clipping the top 0.1% of the range lowers
output MSE vs full min/max. Report both MSEs.

## E06

|x - x_hat| <= s/2 for x inside [min, max].

## E07

Max rounding error: s/2 = 0.0667. x=2.0 clips to
x_hat=0.9333: error 1.0667.

## E08

Clipping dominates: huge max error with moderate mean
error is the signature of clipped outliers.

## E09

Infinite range: s grows without bound (the range is
unbounded), so rounding error explodes. Useless.

## E10

Hypothesis: total MSE vs clip percentile is U-shaped
with a clear minimum. Report the argmin percentile.

## E11

Per-tensor: one (s,z) for the tensor. Per-channel: one
per output channel. Per-group: one per g elements.

## E12

Per-tensor s = 200/15 = 13.3. Row 2 (in [-1,1]) maps
almost entirely to one level: destroyed.

## E13

First change: per-channel quantization (one ruler per
row).

## E14

Per-element scales (s per value) would be exact up to
rounding, the scale storage then costs as much as the
data, so per-group is the practical limit.

## E15

Hypothesis: per-channel MSE << per-tensor MSE,
per-group-128 MSE close to per-channel. Report all
three.

## E16

Weight-only: int8 weights, fp16 activations.
W8A8: both int8 with integer matmul.

## E17

fp32: 4096^2*4 = 67 MB. int8: 16.7 MB. 4x less
traffic.

## E18

Static ranges from calibration no longer cover the new
data: clipping explodes. Fix: recalibrate on the new
domain or switch to dynamic.

## E19

Dynamic per batch: exact ranges, no mismatch, tiny
overhead. Best choice.

## E20

Hypothesis: the static-dynamic quality gap grows with
domain shift size. Report gap vs shift.

## E21

Min/max, percentile, MSE search over ranges.

## E22

Min/max: s = 1001/255 = 3.925. 99.9 percentile: s ~
2/255 = 0.0078.

## E23

Calibration-deployment mismatch: the calibration data
did not represent production inputs.

## E24

MSE search over ranges on the deployment data: the
empirical optimum.

## E25

Hypothesis: final quality tracks the
calibration-deployment match. Report quality vs
match.

## E26

2:4: exactly 2 nonzeros per 4. Block: all-or-nothing
bxb tiles. Unstructured: arbitrary zeros.

## E27

[1,0,0,1,0,1,1,0]: valid (2 per group). Any pattern
with a group of 1 or 3 nonzeros: invalid.

## E28

The pattern is unstructured (or unsupported): index
overhead eats the FLOP saving. Need 2:4 or block.

## E29

Unstructured: skip every zero with zero overhead, so
the finest pattern wins.

## E30

Hypothesis: at 50% sparsity, 2:4 runs ~1.5-2x dense,
unstructured ~1x. Report all three timings.

## E31

Zero out the fraction of weights with smallest
absolute value.

## E32

Mask [0,1,0,1]. Pruned W: [0, 2.0, 0, 1.5].

## E33

Fixes: (1) gradual prune + retrain instead of
one-shot, (2) sensitivity-guided instead of pure
magnitude.

## E34

Gradual pruning to the target with full retraining
between steps.

## E35

Hypothesis: sensitivity-guided pruning beats magnitude
on quality at equal sparsity. Report both losses.

## E36

realized = dense_us * ((1-sparsity) + gather_frac).

## E37

100*(0.5+0.45) = 95 us. Speedup 100/95 = 1.05x.

## E38

Causes: (1) shapes do not qualify for sparse tensor
cores (fallback to dense), (2) the gather/index path
dominates (bad pattern or small shapes).

## E39

Unstructured: finest granularity, every zero skipped
free.

## E40

Hypothesis: 2:4 in [1.5,2]x, unstructured in
[0.9,1.1]x vs dense. Report measured ranges.

## E41

2n log n parameters (k=log2(n) factors, 2n each).

## E42

Dense: 1024^2 ~ 1M. Butterfly: 2*1024*10 = 20480.

## E43

The target matrix is far from the butterfly class
(e.g. random dense): the factorization cannot fit it.
Learned factors or a different structure needed.

## E44

Pad n up to the next power of 2, or use a mixed
factorization for the remainder.

## E45

Hypothesis: butterfly holds quality at 10x fewer
params on mixing-heavy tasks. Report quality and
param counts.

## E46

M = P L P^T R with block-diagonal L, R, params O(nb)
per factor pair.

## E47

2*4096*256 = 2.1M vs 16.7M dense: 8x fewer.

## E48

Fewer degrees of freedom and worse conditioning make
optimization harder, the model underfits.

## E49

b = n: one block = the full matrix, Monarch becomes
dense (no structure left).

## E50

Hypothesis: Monarch needs more steps than dense to
match quality but gets there. Report steps and final
quality.

## E51

total <= sum_i e_i * prod_{j>i} L_j.

## E52

0.01*1.5 + 0.01 = 0.025 (lips [1.0, 1.5]: the early
error is amplified by the later layer).

## E53

Early-layer errors amplify through later layers
(L>1), late-layer errors have nothing downstream to
amplify them.

## E54

Errors shrink downstream: the bound contracts
instead of growing.

## E55

Hypothesis: fitted per-layer amplification is larger
for early layers. Report the fitted amps.

## E56

Perplexity delta, task accuracy deltas,
latency/throughput targets.

## E57

ppl +0.8% (within 2%), task +0, lat 12 ms (within
15): ship.

## E58

It proves perplexity alone is not a sufficient gate:
task evals caught what perplexity missed.

## E59

No-ship on latency (or renegotiate the target):
gates are conjunctive. Great quality with missed
latency fails the serving contract.

## E60

Hypothesis: perplexity delta under-predicts
long-range task damage. Report per-task
correlations.
