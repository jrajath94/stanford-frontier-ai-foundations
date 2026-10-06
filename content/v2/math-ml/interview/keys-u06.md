# Keys, U06 interview bank

Date: 2026-10-06. Rubrics: full marks need the measured number or
the named equation. All numbers computed 2026-10-06, numpy 1.26.4,
float64, seed 7.

## B1

X^T X w = X^T y. Invertible iff the columns of X are linearly
independent (full column rank). Minimum: the equation. Strong:
adds the rank condition. Red flag: "always invertible."
Rubric: both parts. Remediation: U02-C07.

## B2

Gaussian: squared loss, identity link, sigma2_hat = RSS/n.
Bernoulli: log-loss, logit link, sigmoid. Minimum: the two
losses named. Strong: names the links. Red flag: "same model,
different threshold." Rubric: loss + link. Remediation: C05.

## B3

Probabilities [0.665241, 0.244728, 0.090031]. Class 0 gets
0.665241. Cross-entropy -log(0.665241) = 0.407606. Minimum: the
three numbers. Strong: shows the max-shift step. Red flag:
"0.73" (forgot normalization). Remediation: C04.

## B4

phi(x) = [1, sqrt(2) x, x^2]. Check: 1 + 2*2*3 + 4*9 = 49 =
(1+6)^2. Minimum: the sqrt(2). Strong: derives from the
expansion. Red flag: [1, x, x^2] (gives 43). Remediation: C06.

## B5

Geometric margin: distance from boundary to nearest point,
2/||w||. Hinge: max(0, 1 - y(w.x + b)). C prices margin
violations: large C punishes slack, small C tolerates it.
Minimum: both definitions. Strong: the C trade stated.
Rubric: definitions + C. Remediation: C08.

## B6

sum alpha_i y_i = 0. Dropped, the two-point toy gives a = 0.5,
dual 0, primal 1.0: a duality gap. Minimum: the constraint.
Strong: cites the 0 vs 1.0 gap. Red flag: "alpha >= 0 is
enough." Remediation: C09.

## D1 ladder

D1.1. Minimum: minimizer of the sum of squared residuals.
Strong: projection language.
D1.2. w_hat [0.3, 0.8], RSS 1.8, residual sum 0.0 (4.44e-16
measured on the lab toy). The intercept's normal equation
forces the sum to zero.
D1.3. Gaussian log-density is quadratic. Maximizing the
likelihood minimizes the sum of squares.
D1.4. Separable data: no finite MLE exists, ||w|| diverges.
Fix: penalty or early stopping. Strong: predicts the log-like
growth of ||w||.
D1.5. Absolute loss. The "normal equations" become the median
condition: subgradient with sign(residuals). No closed form.
Rubric: 2 points per follow-up, 10 total. Red flags: "OLS is
unbiased so it is fine" (misses the noise change).

## D2 ladder

D2.1. Minimum: dot product in an implied feature space.
Strong: PSD stated as the existence condition.
D2.2. Eigenvalues 0.391064, 0.965601, 1.643335. PSD: yes.
D2.3. dL/dw = 0 gives w = sum alpha_i y_i x_i. dL/db = 0 gives
the balance constraint. The kernel enters when x_i . x_j
becomes k(x_i, x_j) in the dual.
D2.4. See T1.
D2.5. Attack: the Gram costs O(n^2) memory, and gamma = 1e-6
gives cond 3.227e6 with ||alpha|| ~ 9.4e5: capacity is paid in
numerics and memory, not free. Rubric: 2 points per follow-up.

## Q1

X^T X = [[4,6],[6,14]], det 20, inverse [[0.7,-0.3],[-0.3,0.2]].
w_hat [0.3, 0.8]. Residuals [-0.3,-0.1,1.1,-0.7]. RSS 1.8.
sigma2_hat 0.45. Predictive variance at x = 5: 1.665.
Rubric: full marks for all seven numbers. Half for w_hat and
RSS only.

## Q2

Dual(a) = 2a - 4a^2, a* = 0.25, w = [0.5, 0.5], primal 0.25 =
dual 0.25. The forgotten constraint sum alpha_i y_i = 0. The
careless version gives a = 0.5, dual 0, primal 1.0.
Rubric: full marks for the constraint named and the gap
quoted.

## T1

The solver is not broken: residual 4.12e-11 is a true small
residual. The problem is gamma = 1e-6 makes K nearly the ones
matrix (cond 3.227e6). The alphas are huge canceling numbers
(-750000, 500001, 250001.5) that fit b by cancellation. The
residual hides the cancellation: tiny residual, garbage
solution. The jitter halves ||alpha|| to 471405 but it stays
huge: a patch, not a fix. The real fix is gamma matched to the
data scale (gamma = 0.5 gives cond 4.2 and sane alphas). Minimum:
"gamma too small" + the fix. Strong: explains why the residual
lies. Red flag: "use a better solver." Rubric: diagnosis 3,
residual explanation 2, fix 2, jitter assessment 1. Remediation:
C07/C11.

## S1

Measure first: column scales and kappa(X^T X). Expect ~1e18
scale spread and kappa in the millions. The oscillation is the
gradient steps fighting the stretched curvature. The huge
weights are the thin direction exploding. Fix: standardize
columns (two lines: mean/std, transform). Then kappa ~ O(1)
and plain GD converges. Minimum: scaling named + fix. Strong:
kappa measured before and after. Red flag: "tune the learning
rate." Rubric: diagnosis 3, numbers 2, fix 2.

## S2

Check first: the Gram's off-diagonals and cond(K). Memorization
signature: K near identity (off-diagonals ~0), cond huge, and
alphas with huge magnitudes. Bigger C makes it worse: it prices
the training fit higher. Change instead: larger gamma
(smoother kernel), or fewer features, chosen by held-out
accuracy. Minimum: Gram check + gamma. Strong: predicts the
off-diagonal pattern. Red flag: "increase C." Rubric:
diagnosis 3, fix 2.

## R1

Position: the claim is false as stated. The lesson's numbers:
under 15 percent label noise, hinge 0.7833 beats logistic
0.7167. Smoothness buys differentiability and calibrated
probabilities. It costs unbounded influence of far points
(log-loss never saturates). The hinge's flat region caps
influence, which is exactly what noise needs. The experiment
that changes the verdict: rerun at 0 percent noise with a
calibration metric (ECE) added. If logistic wins on both
accuracy and ECE there, the position narrows to "logistic for
clean calibrated outputs, hinge for noisy boundaries."
Minimum: position + the two numbers. Strong: names the
falsifying experiment. Rubric: position 2, numbers 2,
trade stated 2, experiment 2.
