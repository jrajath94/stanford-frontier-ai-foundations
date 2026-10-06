# Lab 05, risk, regularization, and generalization

Unit: math-ml-U05. Date: 2026-10-06. numpy 1.26.4, float64, seed 7.
Keys in labs/keys-lab-05.md. Test-mode: solve closed-book, then check.

## Task 1, empirical risk on coin flips

A coin has true bias p = 0.6. Draw n = 50 flips with
numpy default_rng(7). The classifier predicts the majority class.
(a) Count the heads. Compute p_emp and the gap |p_emp - 0.6|.
(b) Compute the standard error sqrt(p_emp (1 - p_emp)/n). State in
one sentence what the gap in (a) does and does not prove.
(c) Repeat with n = 500 (same seed stream continued). State the new
gap and the new standard error. What shrank, and why?

## Task 2, ridge path and the selection problem

Data x = [0, 1, 2, 3], y = [0, 1, 3, 2]. Model y = w x.
Objective: sum of squared residuals + n lambda w^2, n = 4.
(a) For lambda in {0, 0.1, 0.5, 2, 10}, compute w and the
in-sample MSE. Tabulate.
(b) The in-sample MSE rises with lambda. Explain why that is
expected and why it cannot select lambda.
(c) From lesson C08, LOOCV picks n lambda = 2 (lambda = 0.5) over
lambda = 0, with CV means 1.1806 versus 1.8515. State the decision
rule and explain why CV can select what in-sample MSE cannot.

## Task 3, bias-variance by repeated sampling

Truth: y = 2x. Estimators of f(1) = 2: (A) line through the origin
fit by least squares. (B) the constant sample mean. Generate
B = 200 datasets of 10 points, x = linspace(0, 3, 10), noise
N(0, 1), seed 7.
(a) For each estimator compute bias^2, variance, and MSE at x = 1.
Verify bias^2 + var equals MSE to 6 digits.
(b) Which estimator wins on MSE, and which term explains the win?
(c) Break it: change the truth to y = 2x + 5. Recompute (a) for
estimator A only. Which term moves, and why?
