# keys-lab-06.md

Date: 2026-10-06. Reference outputs for lab 06.

Computed with numpy 1.26.x, float64, on this machine.
Tolerances below absorb platform float differences.

## Task 1

alpha-bar_50 = 0.7772. Theoretical mean of x_50:
2.6447. Theoretical std: 0.4720.
Single-step (seed 7, 20,000 draws): mean 2.6434,
std 0.4707. Closed form (seed 7, 20,000 draws):
mean 2.6417, std 0.4685. Tolerance: 0.02 on
means and stds. Both agree with theory and with
each other: the recursion and the closed form
are the same distribution.

## Task 2

beta-tilde_50 = 0.00960 < beta_50 = 0.00995.
Max absolute disagreement between (14.19) and
(14.22) over the 61-point grid: 1.78e-15
(tolerance 1e-10). mu-tilde_50 at x_t = 3.0420:
3.0394 (tolerance 0.001). The agreement proves
the noise rewrite of the posterior mean is an
exact algebraic identity, so epsilon-prediction
trains the same mean as x_0-prediction.

## Task 3

Score at x_t = 3.0420: analytic -1.7829,
via -epsilon-hat / sqrt(1 - alpha-bar_t):
-1.7829, central finite differences (h = 1e-5):
-1.7829. Pairwise gaps below 1e-4 (tolerance
1e-3). The noise predictor is really learning
the score of the conditional, rescaled by
-sqrt(1 - alpha-bar_t): predicting epsilon and
predicting the score are the same task.
