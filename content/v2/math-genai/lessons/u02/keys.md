# keys.md, U02 lesson answer keys

Date: 2026-10-06. Closed-book answers for lesson 02. Keep separate
from the lesson file. All numbers computed 2026-10-06, numpy
1.26.4, float64, seed 0 where RNG is used.

## E01

Jensen: for convex f, f(E[X]) <= E[f(X)]. Toy: E[X] = 0.55,
f(E[X]) = 0.3025, E[f(X)] = 0.5*(0.16 + 0.49) = 0.325. Gap =
0.0225.

## E02

The two-point case is the definition of convex applied once:
f(lambda x + (1-lambda) y) <= lambda f(x) + (1-lambda) f(y).
The definition is the whole proof. Induction extends it to n
points.

## E03

Concave: sqrt. Jensen reverses: g(E[X]) >= E[g(X)]. Toy:
0.7416 > 0.7346. Non-convex: x^3 - 3x. No fixed order: on
{-2, 1} with equal weight, f(E) = 1.375 and E[f] = -2.0,
reversing the Jensen direction.

## E04

phi = r, one number in [0, 1]. It cannot represent a mixture,
e.g. the 50/50 mix of Bernoulli(0.1) and Bernoulli(0.9): the
best single r is 0.5 and the two-hump shape is lost.

## E05

q = posterior [0.9, 0.1]. Bound = log p(x) - KL(q || post) =
-1.0 - 0 = -1.0 bits. Gap = 0. The bound is tight exactly when
q matches the posterior.

## E06

log is concave. Jensen for concave f flips the direction:
log E[Y] >= E[log Y]. That is where >= comes from.

## E07

r(0) = 0.3/0.6 = 0.5. r(1) = 0.7/0.4 = 1.75. E_q[r] = 0.6*0.5 +
0.4*1.75 = 0.3 + 0.7 = 1.0.

## E08

r = d/(1-d) = 0.8/0.2 = 4.0. Derivation: d = p/(p+q), so
d/(1-d) = p/q = r.

## E09

TV = 0.5*(|0.3-0.6| + |0.7-0.4|) = 0.5*(0.3 + 0.3) = 0.3.
H^2 = (sqrt(0.3)-sqrt(0.6))^2 + (sqrt(0.7)-sqrt(0.4))^2 =
0.0515 + 0.0417 = 0.0932.

## E10

Reverse KL: f(t) = -log2(t). f(1) = -log2(1) = 0. f''(t) =
1/(t^2 ln 2) > 0 for t > 0, so f is convex.

## E11

f(t) = t ln t. f*(s) = e^{s-1}. T*(x) = ln r(x) + 1.

## E12

T* = [ln 0.5 + 1, ln 1.75 + 1] = [0.3069, 1.5596].
E_p[T*] = 0.3*0.3069 + 0.7*1.5596 = 1.1838.
E_q[e^{T*-1}] = E_q[r] = 1.0. Dual = 0.1838 nats = KL(p||q) in
nats = 0.2651 bits.

## E13

KL = 0.7*log2(0.7/0) + 0.3*log2(0.3/1) = +infinity. Repair:
smooth, q(heads) = epsilon > 0. Cost: bias. The smoothed model
is not the model you meant. Epsilon = 0.01 gives 3.769 bits.

## E14

sigma^2 = E_q[(r-1)^2] = 0.6*0.25 + 0.4*0.5625 = 0.375, sigma =
0.6124. SE = 0.6124/sqrt(400) = 0.0306. Predicted before any
code runs.

## E15

Standard error scales as sigma/sqrt(N). sqrt(100) = 10, so 100x
the samples shrinks the error 10x.

## E16

dJ/db = 1 - E_q[e^{T-1}]. At (1,1): T = [1, 2], E_q[e^{T-1}] =
0.6*e^0 + 0.4*e^1 = 0.6 + 1.0873 = 1.6873. dJ/db = -0.6873.

## E17

Legal when the integrand is smooth in the parameters and the
derivatives are bounded by an integrable function (dominated).
Failure: a ReLU witness has a kink. The derivative does not
exist there and autodiff silently returns a subgradient.

## E18

Approximation gap: 0.0006 nats for the line class (0.1838 for
constants). Estimation gap: sigma/sqrt(N). Optimization gap:
whatever the grid or gradient steps leave behind.

## E19

Richness fails: constants cannot approach T*. Symptom: the
reported divergence stays 0 at every N. More samples never fix
a poor class.

## E20

Categorical with K outcomes adds K-term sums. It tests the
f-divergence table at scale: the formulas are unchanged, only
K grows.

## L01

Convex: every chord lies above the graph. Toy: gap 0.0225 as in
E01. Derivation: log p(x) = log E_q[p(x,z)/q(z)] >=
E_q[log p(x,z) - log q(z)] by Jensen (log concave). Code: the
C03 snippet. Assert bound + gap == log p(x) to 1e-12. O(|Z|).
Compare: q = [0.6,0.4] gives bound -1.4490, gap 0.4490. q =
posterior gives bound -1.0, gap 0. Debug: bound above log p(x)
means the support assumption broke (q misses mass where p(x,z)
> 0) or a sign flipped. Critique: the code divided by q
outside its support. Design: tighten by enriching the family
(more parameters) or by choosing q closer to the posterior.

## L02

f-divergence: D_f = E_q[f(r)] for convex f with f(1) = 0. Toy:
KL 0.2651, reverse 0.2771, JS 0.0667, TV 0.3, H^2 0.0932.
Derivation: D_f = E_q[f(r)] >= f(E_q[r]) = f(1) = 0 by Jensen.
Code: the C05 Df function. Assert each >= 0. O(K). Compare: KL
weights by p and punishes q for missing p mass. Reverse KL
weights by q, punishes q for extra mass. Toy: KL 0.2651 vs
reverse 0.2771. Debug: negative D_f means f is not convex (the
sqrt(t) trap) or f(1) != 0. Critique: check the generator
before the code. Design: move q(1) from 0.4 to 0.45,
recompute all five, rank by absolute change.

## L03

Dual: D_f = sup_T E_p[T] - E_q[f*(T)]. Toy: T* = [0.3069,
1.5596], dual 0.1838 nats. Derivation: pointwise sup over T(x)
gives f(r(x)). For KL, d/ds of s*r - e^{s-1} = 0 gives s = ln r
+ 1. Code: the C06 snippet. Assert |dual - KL| < 1e-12. Primal
needs densities. Dual needs samples only. Debug: empirical dual
above truth is finite-sample luck, not a broken population
dual. Report the SE. Critique: the population dual never
exceeds the truth. The empirical one can. Design: polynomial
witness of degree d = 0, 1, 2, and so on. Predict monotone
non-decreasing dual values approaching 0.1838.

## L04

Density ratio: r(x) = p(x)/q(x). Toy: 0.5 and 1.75.
Derivation: optimal classifier d(x) = p/(p+q). Algebra gives r
= d/(1-d). Code: both routes in C04. Assert E_q[r] = 1 to
1e-12. Compare: direct ratios need closed-form densities.
classifier needs only samples but can overfit. Debug:
memorized classifier gives d in {0, 1} on training points and
undefined ratios elsewhere. Regularize. Critique: a ratio fit
on training piles says nothing about fresh points. Design: fix
the toy, draw N = 50..5000, measure MSE of estimated ratios
against 0.5 and 1.75, plot against 1/N.

## L05

Three gaps: approximation (class too small), estimation (finite
N), optimization (maximizer not found). Toy: 0.0006 nats for
lines. Derivation: for each x, sup_T of the pointwise term is
f(r(x)). Taking sup over a smaller class can only lower the
value, so the population dual <= truth. Code: the C10 grid
search. Assert best <= truth + 1e-12. Compare: constants 0,
lines 0.1832, full 0.1838 nats. Debug: empirical dual above
truth. Finite-sample maximizer overshoot. Shrink with SE.
Critique: more data shrinks estimation only. Richness needs a
bigger class. Design: shippable report = number + SE +
support verdict (min q mass on p samples) + class description.

## Implementation and debug task

Correct version:

```python
import numpy as np
def f_divergence(p, q, f):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    assert np.all((p > 0) <= (q > 0)), "support violation: p > 0 where q == 0"
    assert abs(q.sum() - 1.0) < 1e-9 and abs(p.sum() - 1.0) < 1e-9
    r = p / q
    val = float(np.sum(q * f(r)))
    assert val >= -1e-12, "negative divergence: check f convex, f(1) == 0"
    return val

p = [0.3, 0.7]
q = [0.6, 0.4]
kl = f_divergence(p, q, lambda t: t * np.log2(t))
assert abs(kl - 0.26514844544032273) < 1e-9
```

The bug: the broken version computes sum(f(p/q)) without the
q weights, i.e. it sums f(r) instead of E_q[f(r)]. On the toy
it returns f(0.5) + f(1.75) instead of 0.6*f(0.5) +
0.4*f(1.75). The fix: multiply by q before summing. The
support assert also catches q(heads) = 0 before the division.

## Changed-constraint scenarios

S1. Still works: Jensen, the dual form, Monte Carlo, the
1/sqrt(N) law, the three gaps. Breaks: exact finite sums
become integrals. The Bernoulli table becomes a density ratio
function. First change: drop histograms, estimate r(x) with
the classifier route (C04). The dual objective is unchanged.

S2. Swap p and q. KL 0.2651 -> 0.2771 (reverse KL). TV 0.3,
H^2 0.0932, JS 0.0667 stay (symmetric). Ratios invert: 2.0 and
0.5714. The bound direction in C03 is unaffected (it bounds
log p(x), not a divergence), but any D_f(p||q) claim must be
recomputed: divergence direction is part of the claim.

## Research-critique question

Flaw 1: the empirical dual is not a lower bound. With N = 500
the maximizer overshoots on lucky draws (C10). 0.001 may be
finite-sample noise. Expose: report the SE and a
sample-split dual.
Flaw 2: a small critic has an unknown approximation gap
(C10). The dual measures critic weakness, not distribution
closeness. Expose: enlarge the critic and watch the number
move.
Flaw 3: near-zero JS on samples does not imply good samples.
Expose: inspect actual generated samples and run a held-out
evaluation (U01-C12 protocol).
