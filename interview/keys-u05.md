# keys-u05.md: interview answer keys, U05

Date: 2026-10-06. Minimum sufficient explanation, strong
answer, red flags, rubric, remediation per item.

## B1

Minimum: s=(max-min)/(2^b-1), z=round(-min/s),
x_q=clamp(round(x/s)+z), x_hat=s*(x_q-z).
Strong: explains why z exists (exact zero).
Red flags: missing clamp.
Rubric: 2 points. Remediation: C01.

## B2

Minimum: |err| <= s/2 in range. Second source:
clipping (out-of-range values snap to the edge).
Strong: writes total MSE = rounding + clipping.
Red flags: "quantization error is uniform."
Rubric: 2 points. Remediation: C02.

## B3

Minimum: per-tensor one (s,z), per-channel one per
row, per-group one per g elements. Per-tensor unsafe
when channels span very different ranges.
Strong: gives the 100x-range example.
Red flags: "granularity does not matter."
Rubric: 2 points. Remediation: C03.

## B4

Minimum: weight-only = int8 weights, fp16 acts,
W8A8 = both int8. Static = fixed ranges, dynamic =
per-batch ranges.
Strong: maps to bandwidth vs integer-math goals.
Red flags: confusing static/dynamic with
weight/activation.
Rubric: 2 points. Remediation: C04.

## B5

Minimum: min/max, percentile, MSE search. Min/max
fails with outliers (one giant sets the scale).
Strong: quantifies the 500x scale difference.
Red flags: "min/max is the safest."
Rubric: 2 points. Remediation: C05.

## B6

Minimum: exactly 2 nonzeros per 4 consecutive
elements. Pattern matters because hardware skips
only supported patterns, unstructured pays index
overhead.
Strong: cites the ~1x vs ~1.8x measured gap.
Red flags: "50% sparse = 2x faster."
Rubric: 2 points. Remediation: C06, C08.

## B7

Minimum: zero the smallest-|w| fraction. Upgrades:
sensitivity-guided, gradual + retrain.
Strong: explains why magnitude is only a proxy.
Red flags: "pruning never hurts."
Rubric: 2 points. Remediation: C07.

## B8

Minimum: 2n log n. N=1024: 20480 vs ~1M dense
(51x fewer).
Strong: derives from log n factors of 2n.
Red flags: "O(n^2)."
Rubric: 2 points. Remediation: C09.

## Deep ladder 1

L1a. S: level spacing. Z: the integer level for 0.0.
L1b. S=0.1333, z=8, x_q=12, x_hat=0.5333.
L1c. Nearest mark is at most half a spacing away.
L1d. O(N) one-time, then 4x less traffic per step.
L1e. Per-tensor: quiet channel collapses to ~1 level
(MSE ~0.28). Per-channel: 16 levels (MSE ~0.0013).
L1f. Cause: range set by an outlier or mixed
channels. Fix: per-channel or clipped range.
L1g. Assumes the calibrated range covers the data,
one outlier at 1000 makes s 500x too coarse.
L1h. Sweep clip percentile, plot total MSE:
U-shaped, the argmin is the pick.
Scoring: 1 point per rung.

## Deep ladder 2

L2a. Realized = measured time, FLOP ratio ignores
index/gather overhead.
L2b. 95 us (1.05x) and 55 us (1.8x).
L2c. Math halves (50 us) but gather adds ~45 us:
net ~95 us ~ 1x.
L2d. Models math+gather, does not model cache
effects or occupancy.
L2e. Unstructured: ~1x, needs nothing. 2:4: ~1.8x,
needs sparse tensor cores. Block: varies, needs
block support.
L2f. Shapes do not qualify (fallback), or the index
path dominates (small shapes).
L2g. Assumes hardware support, without it, 2:4 runs
as dense (no win).
L2h. Time dense/2:4/unstructured at 50%: expect
[1.5,2]x and [0.9,1.1]x vs dense.
Scoring: 1 point per rung.

## A1

Per-tensor s = 200/15 = 13.33. Quiet channel
per-tensor MSE ~ 1/3 = 0.333 (crushed to ~1 level).
Per-channel s = 0.1333, MSE ~ s^2/12 = 0.00148.
Ratio ~225x.

## A2

[1.0, 2.0]: bound = 0.02*2.0 + 0.02 = 0.06. [2.0,
1.0]: bound = 0.02*1.0 + 0.02 = 0.04. Keep more
precision in the early layer when later layers
amplify (first case), the early error is the one
that gets multiplied.

## D1

Bug: z = 0 hardcoded. With min != 0 the true zero
point is round(-min/s) != 0, so quantize(0) lands on
the wrong level and dq(q(0)) != 0. Fix: compute z =
round(-min/s) from the range. Invariant:
dq(q(0.0)) == 0.0 exactly.

## S1

Per-element scales would be optimal (exact ranges),
but scale storage then equals data storage: the
limit is the metadata ratio. Practical optimum:
per-group with small g, bounded by scale overhead.

## S2

Memory must halve with zero error budget: int8
halves fp16 memory but is approximate (fails the
zero budget strictly), 2:4 is approximate too,
butterfly is exact only if the layer is exactly
representable (rare). Honest answer: no method meets
both literally, nearest is int8 with the error shown
to be below measurement noise, or store fp16 with
2:4... Pick: reframe, halve via int8 and prove the
task deltas are zero within eval noise, else the
constraints are inconsistent.

## R1

(1) Which tasks? Invalid: perplexity only.
(2) Baseline: fp16 or fp32, tuned? Invalid: weak
baseline.
(3) Granularity and calibration reported? Invalid:
no.
(4) Seeds and error bars? Invalid: one run.
(5) Long-range / outlier-sensitive tasks included?
Invalid: no.
