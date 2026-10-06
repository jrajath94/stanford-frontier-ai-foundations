# Keys, lab 05 risk, regularization, and generalization

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 7.

## Task 1

(a) 30 heads in 50 flips. p_emp = 0.6. Gap = 0.0.
(b) Standard error = sqrt(0.6 * 0.4/50) = 0.0693. The gap of 0.0
is one lucky draw. It does not prove the estimator is exact. The
standard error says typical gaps are ~0.07.
(c) Continued stream, n = 500: 294 heads, p_emp = 0.588, gap =
0.012, standard error = 0.022. The standard error shrank by about
sqrt(10) because the variance of the mean falls as 1/n.

## Task 2

(a)

| lambda | w | in-sample MSE |
|---|---|---|
| 0 | 0.928571 | 0.482143 |
| 0.1 | 0.902778 | 0.484471 |
| 0.5 | 0.812500 | 0.529297 |
| 2 | 0.590909 | 0.881198 |
| 10 | 0.240741 | 2.138032 |

(b) The penalty pulls w toward 0, away from the least-squares fit,
so the raw residual must rise. In-sample MSE cannot select lambda
because it always prefers lambda = 0: it measures fit, not
generalization.
(c) Decision rule: pick the lambda with the smallest LOOCV mean.
1.1806 < 1.8515, so lambda = 0.5 wins. CV can select because each
fold's error is measured on a point the fit did not see: it
estimates population risk, not training fit.

## Task 3

(a) Estimator A (line): bias^2 = 0.000489, var = 0.035717, MSE =
0.036206. Sum: 0.036206. Estimator B (constant): bias^2 =
0.921689, var = 0.115717, MSE = 1.037406. Sum: 1.037406. Both
splits add up to 6 digits.
(b) A wins on MSE (0.0362 versus 1.0374). The bias term explains
it: B's constant cannot track the slope, so its bias^2 is 0.92
against A's 0.0005.
(c) Truth y = 2x + 5, estimator A: bias^2 = 7.042053, var =
0.035717, MSE = 7.077770. The bias term moves: the line through
the origin cannot represent the intercept, so E[w]*1 = 4.346313
against truth 7.0. The variance is unchanged because the noise
model did not change. Computed in compute_run4.py Task 3(c) block
(fresh seed-7 stream, f(1) = 7.0). bias^2 + var = MSE to 6 digits.
