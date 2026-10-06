# Answer keys, lesson 04 (Wasserstein and improved adversarial training)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run4.py.

## E01

Rows sum to p = [0.5, 0.5], columns to q = [0.5, 0.5].
Legal: [[0.5, 0], [0, 0.5]]. Illegal: [[0.5, 0.5], [0, 0]]:
row 1 sums to 1.0, not 0.5.

## E02

Cost matrix [[1, 3], [1, 1]]. Diagonal plan cost =
0.5*1 + 0.5*1 = 1.0. Crossed plan cost = 0.5*3 + 0.5*1 =
2.0. W1 = min = 1.0.

## E03

Any coupling of delta_0 and delta_theta must put all
mass on the single pair (0, theta): gamma = delta_{(0,
theta)}. Cost = 1 * |0 - theta| = |theta|. No other
coupling exists.

## E04

W1 = 1.0, JS = 1 bit, dW1/dtheta = 1.0, dJS/dtheta = 0.
The generator needs dW1/dtheta: the only nonzero
direction signal.

## E05

(1) Ground cost is a metric. Breaks: c = 0 everywhere
gives W1 = 0 for all pairs. (2) f is 1-Lipschitz
everywhere. Breaks: f(x) = -2x gives gap 2.0 > W1 =
1.0. (3) The max runs over all such f. Breaks: the
clipped critic gives gap 0.01, far below 1.0.

## E06

Quote f(x) as the price at x. The 1-Lipschitz rule
forbids price differences larger than distance: no
arbitrage across space. The max gap is the fair
shipper's max profit, which equals the cheapest
transport cost.

## E07

sv(W1) = sqrt(4 + 1) = 2.2361. sv(w2) = sqrt(0.25 + 1)
= 1.1180. Product = 2.5000. ReLU contributes 1.

## E08

The product bounds the composition's slope from
above. The true constant is the max slope attained,
which can be strictly smaller (here 1.0).

## E09

Gap = 0.01 * 1 = 0.01. Bias factor 100x: the screen
shows 1 percent of the true W1.

## E10

With 90.9 percent of weights pinned at +-c, the
critic's effective parameters are nearly binary.
The function class collapses toward a small set of
extreme configurations, so the "max over f" in the
dual runs over a crippled set.

## E11

Two-sided: 10 * (0.3 - 1)^2 = 4.9. One-sided:
max(0, 0.3 - 1)^2 = 0.0. The two-sided form
punishes the flat critic. The one-sided form lets
it pass.

## E12

f(x) = x + 3 exp(-((x-5)/0.1)^2): mean penalty on
the [0,1] interpolation segment is 5.1e-22, max
||grad|| near x = 5 is 26.7. A zero penalty proves
nothing about slopes off the sampled segments.

## E13

Ratio = gap / W_exact. Optimal critic: 1.0 / 1.0 =
1.0. Suboptimal (w = -0.5): 0.5 / 1.0 = 0.5.

## E14

Ratio above 1.0 is impossible for a 1-Lipschitz
critic: the critic breaks the bound (C03). Check
the enforcement, not the distance.

## E15

n_critic = 1: 1*2 + 3 = 5 units. n_critic = 5:
5*2 + 3 = 13 units. n_critic = 10: 10*2 + 3 = 23
units.

## E16

Flaw one: clipping biases the gap by the clip
scale, so "perfect" is miscalibrated (C05). Flaw
two: n_critic = 100 overfits the critic to its
box. Frequency cannot repair a broken
enforcement (C08).

## E17

Minimize ||A z - x||^2. Normal equations:
(A^T A) z = A^T x. A^T A = 5, A^T x = 1.5 + 6.0 =
7.5. z* = 7.5 / 5 = 1.5.

## E18

E(x) = (A^T A)^{-1} A^T x = (x1 + 2 x2)/5 = x1/5 +
2 x2/5. E([1.5, 3.0]) = 0.3 + 1.2 = 1.5.

## E19

IIT_DGM_WGAN.ipynb, IIT_DGM_UDA.ipynb,
IITM_DGM_Vanilla_GAN.ipynb, wgan_gp_training.gif.
Promotion needs: opened notebook, executed
cells, extracted loss/critic code, blob hash
and date. For the GIF, extracted frames with
frame count and what is plotted.

## E20

Exact W1 = 2.0 on the mode-drop toy. The weak
critic's gap = 0.02. Training minimizes the
gap (0.02), while the number you want is the
distance (2.0). The 100x miss is invisible
without the calibration ratio.

## L01

W1 is the cheapest transport cost over all
couplings. Toy: diagonal 1.0 vs crossed 2.0.
Derivation: only coupling delta_{(0,theta)},
cost |theta|. Implement coupling_cost with
row/column asserts. Complexity O(K^2) memory.
Primal defines, dual computes. Debug: column
sums mismatch means not a coupling. Critique:
c = 0 kills the ruler. Design: sweep theta,
predict W2 = theta^2 and slope 2 theta at
theta = 1, then compute.

## L02

The dual is the max price gap over
1-Lipschitz f. Toy: 1.0 vs 2.0 vs 0.01.
Price story: prices, no arbitrage, max
profit. Implement the linear gap. The
f-divergence dual uses a conjugate scale.
the W dual uses the Lipschitz bound.
Debug: ratio 1.4 means the bound broke.
Critique: the critic class is the first
assumption to fail. Design: grid over w in
[-1, 1], predict convergence to 1.0, then
compute.

## L03

The Lipschitz constant is the max slope.
Toy: product 2.5, measured 1.0. Derivation:
composition of Lipschitz maps multiplies
constants. Affine maps take the top singular
value. Implement sv product plus the
finite-difference max-slope check. Spectral
bounds analyze, the penalty trains, clipping
baselines. Debug: slope above bound means
the slope code is wrong. Critique: 2.5 vs
1.0 is too loose to calibrate. Design:
spectral-normalize each layer, predict
product exactly 1.0, then compute.

## L04

The penalty pulls ||grad f|| to 1 on
interpolation lines. Toy: 4.9 vs 0.0 at w =
0.3. Blind spot 5.1e-22 vs 26.7. Derivation:
xhat = eps x + (1-eps) y, penalty on the
input gradient norm. Implement the
finite-difference penalty. Two-sided keeps
the critic sharp. One-sided only caps it.
Debug: zero penalty with spikes means the
spike is off-segment. Critique: zero penalty
certifies nothing. Design: sample xhat near
x = 5, predict the mean penalty jumps, then
compute.

## L05

The protocol fixes architecture, seeds,
budget, and metrics, varying only the loss
and enforcement. Toy: 5/13/23 units per
G step. Derivation: step counts hide the
n_critic multiplier. Implement the
methods/seeds/budget checklist. Method
claims need the protocol. Single-method
claims need only a deep dive. Debug:
best-of-5 vs 1 seed is selection bias.
Critique: the protocol does not control
implementation quality. Design: three
methods on the toy, same 5 seeds, fixed
flop budget, report gap plus calibration
plus coverage.

## Implementation and debug task

Reference:

```python
import numpy as np
def w1_dual_gap(f, xs_p, xs_q):
    fp = np.asarray(f(np.asarray(xs_p, dtype=float)))
    fq = np.asarray(f(np.asarray(xs_q, dtype=float)))
    assert fp.shape == np.shape(xs_p) and fq.shape == np.shape(xs_q)
    return float(fp.mean() - fq.mean())
```

Bug: f(x) = -2x has slope 2, breaking the
1-Lipschitz assumption, so the gap 2.0 is not
W1. Clipping the output values does not fix
the slope. Fix: f = lambda x: -x. Then
w1_dual_gap(f, [0.0], [1.0]) returns 1.0.

## S1

Surviving: the coupling definition and the
min over plans (still well-defined). Breaks
first: W1 = 0 for every pair, so the dual
sup is 0 and every gap statement is vacuous.
Smallest repair: restore a metric ground
cost. Without one there is no Wasserstein
distance.

## S2

The dual becomes W1 = (1/2) max_{L(f) <= 2}
gap, i.e. the max gap over 2-Lipschitz f
equals 2 * W1. Toy gaps double: 2.0, 4.0,
0.02. Sections to edit: C03 (assumption 2),
C04 (bound target), C05 (clip analysis),
C06 (penalty target becomes (||grad||-2)^2).

## Research-critique question

Four flaws. One: clipping at c = 0.01 scales
the gap by ~0.01, so "W1 = 0.001" is a
scaled artifact, exposed by recomputing with
the penalty (C05). Two: n_critic = 1 leaves
the critic stale, so the gap is a lower bound
far from W1, exposed by the calibration ratio
(C07, C08). Three: one seed cannot separate
method from luck, exposed by the same-seed
multi-seed rerun (C12). Four: the gap is not
a sample-quality certificate, exposed by the
coverage diagnostic catching dropped modes
the gap misses (C11, U03-C12).
