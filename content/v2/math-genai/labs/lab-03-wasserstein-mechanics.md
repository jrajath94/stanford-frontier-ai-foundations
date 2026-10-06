# Lab 03, Wasserstein mechanics on the point-mass toy

Unit: math-genai-U04. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-03.md. Test-mode: solve closed-book, then
check. The toy: P0 = delta_0, P_theta = delta_theta,
theta = 1 unless stated. Ground truth: compute_run4.py.

## Task 1, coupling by hand then in code

(a) By hand: on p at {0, 2} (0.5, 0.5) and q at {1, 3}
(0.5, 0.5), write the cost matrix and the two
coupling matrices from C01. Compute both costs.
(b) In code: implement coupling_cost(gamma, xs, ys)
with row-sum and column-sum asserts. Reproduce 1.0
and 2.0.
(c) Prove on paper that delta_0 to delta_theta admits
exactly one coupling. State W1.
(d) Break it: pass a gamma with correct row sums and
wrong column sums. Record exactly what your assert
does.

## Task 2, dual gaps: predict first, measure second

(a) Predict: the dual gap at theta = 1 for f(x) = -x,
f(x) = -2x, and the linear critic with |w| <= 0.01.
Write all three numbers before computing.
(b) Measure: implement the three critics and the gap.
Record the numbers and judge the predictions.
(c) State which assumption each non-conforming
critic breaks.
(d) Write one sentence: what would you tell a
teammate who reports "W1 = 2.0" from the second
critic?

## Task 3, clipping capacity: predict first, measure second

(a) Predict: the fraction of U[-0.1, 0.1] weights
that land on the [-0.01, 0.01] boundary after
clipping. Show the arithmetic.
(b) Measure: seed 0, 2000 weights. Record the
measured fraction and judge the prediction.
(c) Compute the output scale of a 5-layer net with
all weights at +-0.01. State the number.
(d) Write one sentence: why does the clipped gap
underestimate W1 even when the critic is
"optimal in its box"?

## Task 4, gradient penalty and its blind spot

(a) Predict: the two-sided and one-sided penalties
at |w| = 0.3 and |w| = 2.5, lambda = 10. Write all
four numbers before computing.
(b) Measure: implement the finite-difference
penalty on the linear toy. Record the numbers.
(c) Verify ||grad f|| = 2.5 for f(x) = 2.5x by
finite differences. Record the agreement.
(d) Implement the spike critic f(x) = x + 3
exp(-((x-5)/0.1)^2). Measure the mean penalty on
6 interpolation points in [0, 1] and the max
gradient near x = 5. State what a zero penalty
certifies.

## Task 5, stale-critic calibration

(a) By hand: exact W1 at theta = 1, and the gap of
the critic with w = -0.5. Compute the calibration
ratio.
(b) In code: reproduce both numbers and the ratio.
(c) Set w = -1.4 (breaks the bound). Record the
gap and the ratio. State what the ratio above 1.0
diagnoses.
(d) Write one sentence: a run reports a falling
critic loss with a frozen generator. Is that
progress? Use the C07 failure case.
