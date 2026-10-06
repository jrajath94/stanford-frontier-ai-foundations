# Answer keys, lab 03 (Wasserstein mechanics)

Date: 2026-10-06. Ground truth: compute_run4.py.
numpy 1.26.4, float64, seed 0.

## Task 1

(a) Cost matrix [[1, 3], [1, 1]]. Diagonal plan:
0.5*1 + 0.5*1 = 1.0. Crossed plan: 0.5*3 + 0.5*1
= 2.0.
(b) The asserts check abs(gamma.sum(1) - p).max()
and abs(gamma.sum(0) - q).max() below 1e-12.
Costs reproduce 1.0 and 2.0.
(c) Marginals force gamma(0, theta) = 1: the row
must sum to 1 at x = 0 and the column at y =
theta, and there is only one cell. W1 = |theta|.
(d) The column-sum assert raises AssertionError
before any cost is computed. The lesson: never
trust a cost from an unchecked plan.

## Task 2

(a) Predictions: 1.0, 2.0, 0.01.
(b) Measured: 1.0, 2.0, 0.01. All three match:
they are closed-form, so prediction is
derivation, not guessing.
(c) f(x) = -2x breaks the 1-Lipschitz
assumption (C03 assumption 2). The clipped
critic breaks the richness assumption (C03
assumption 3): the box is too small a class.
(d) "Your critic breaks the Lipschitz bound,
so 2.0 is not W1. It is an overestimate from
an illegal witness."

## Task 3

(a) P(|w| > 0.01) = 1 - 0.02/0.2 = 0.9. Predict
0.90.
(b) Measured 0.909 (seed 0, n = 2000). Within
sampling noise of 0.90: binomial SE =
sqrt(0.9*0.1/2000) = 0.0067.
(c) 0.01^5 = 1e-10.
(d) The box optimum is not the unconstrained
optimum: the "best in the box" gap is capped
by the box scale, so it underestimates W1 by
construction.

## Task 4

(a) Predictions: |w| = 0.3: two-sided 4.9,
one-sided 0.0. |w| = 2.5: both 22.5.
(b) Measured: 4.9, 0.0, 22.5, 22.5. Match.
(c) Finite difference gives 2.49999999996,
agreement to 1e-6.
(d) Mean penalty on the segment: 5.1e-22.
Max ||grad|| near x = 5: 26.7. A zero penalty
certifies nothing off the sampled segments.

## Task 5

(a) Exact W1 = 1.0. Gap at w = -0.5: 0.5.
Ratio = 0.5.
(b) Reproduced: 1.0, 0.5, 0.5.
(c) Gap = 1.4, ratio = 1.4. A ratio above 1.0
diagnoses a broken Lipschitz bound, not a
better critic.
(d) No: with a frozen generator W1 cannot move,
so the falling loss is the critic finding a
better witness. The ruler moved, not the
distance.
