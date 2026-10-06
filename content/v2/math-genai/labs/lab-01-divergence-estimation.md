# Lab 01, divergence estimation on the Bernoulli toy

Unit: math-genai-U02. Date: 2026-10-06. numpy 1.26.4, float64,
seed 0 where RNG is used. Keys in labs/keys-lab-01.md.
Test-mode: solve closed-book, then check. The toy: p =
Bernoulli(0.7) on outcomes {0, 1} with p = [0.3, 0.7], q =
[0.6, 0.4]. Log base 2 in bits unless stated.

## Task 1, f-divergence table by hand then in code

(a) By hand: compute the density ratios r(0), r(1).
(b) By hand: compute KL, reverse KL, JS, TV, Hellinger^2 from
the ratio form. Show at least two terms of one sum.
(c) In code: write f_divergence(p, q, f) and reproduce all
five numbers to 1e-9.
(d) Break it: pass f(t) = sqrt(t) (concave). Record the exact
failure (which assert fires, and why it must).

## Task 2, Jensen gap and its direction

(a) By hand: f(x) = x^2, X in {0.4, 0.7} with equal weight.
Compute f(E[X]), E[f(X)], and the gap.
(b) In code: verify the gap is 0.0225. Then verify the concave
reversal on sqrt: g(E[X]) = 0.7416 > E[g(X)] = 0.7346.
(c) Break it: f(x) = x^3 - 3x on {-2, 1}, equal weight. Show
the Jensen direction reverses and name the violated condition.

## Task 3, Monte Carlo: predict first, measure second

(a) Predict: sigma of r under q, then the standard error of
the Monte Carlo estimate of E_q[r] at N = 400. Write the
number before running code.
(b) Measure: seed 0, draw N = 400 from q, estimate E_q[r].
Record the estimate and the absolute error.
(c) Compare: is |error| within about 2 SE of zero? Repeat at
N = 10000 and record both numbers.
(d) Write one sentence: what would you tell a teammate who
reports a divergence estimate from N = 50 samples with no
error bar?

## Task 4, variational dual on a grid

(a) In code: implement J(a, b) = E_p[T] - E_q[e^{T-1}] in nats
for T(x) = a*x + b. Grid-search a, b over [-4, 4] in 81 steps.
Record the best value and where it sits.
(b) Verify the best value never exceeds KL in nats (0.1838).
Compute the approximation gap.
(c) Check the gradient: analytic dJ/da, dJ/db at (1, 1)
versus central finite differences with eps = 1e-6. Record
both pairs and the max discrepancy.
(d) Break it: restrict T to constants (a = 0). What is the
best dual value? Which gap explains the shortfall?

## Task 5, support mismatch

(a) Compute KL(p||q) for q(heads) = 0.01 by hand to 3 decimals.
(b) In code: set q(heads) = 0.0. Record exactly what numpy
does (warning text and resulting value).
(c) Write the support assert you would put in front of every
divergence function in this course, and say which task-1
failure it blocks.
