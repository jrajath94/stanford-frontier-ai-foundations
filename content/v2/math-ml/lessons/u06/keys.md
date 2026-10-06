# Keys, lesson 06 linear models, kernels, and margins

Date: 2026-10-06. Test-mode: solve closed-book first. All numbers
computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## E01

X = [[1, 0], [1, 1], [1, 2], [1, 3]]. Shape (4, 2): 4 points, 2
parameters (intercept, slope).

## E02

X^T X = [[4, 6], [6, 14]]. Determinant 4*14 - 36 = 20.

## E03

Inverse [[0.7, -0.3], [-0.3, 0.2]]. X^T y = [6, 13]. b = 0.7*6 -
0.3*13 = 0.3. w = -0.3*6 + 0.2*13 = 0.8. Matches [0.3, 0.8].

## E04

Predictions [0.3, 1.1, 1.9, 2.7]. Residuals [-0.3, -0.1, 1.1,
-0.7]. RSS 0.09 + 0.01 + 1.21 + 0.49 = 1.8. Sum 0.0: the
intercept column of ones makes the first normal equation read
sum of residuals = 0.

## E05

Without the intercept the first normal equation vanishes, so
nothing forces the residuals to sum to zero. On this toy the
no-intercept fit gives w = 13/14 = 0.9286 with residual sum
-0.2857, not zero.

## E06

sigma2_hat = 1.8/4 = 0.45. l_max = -2 log(2 pi 0.45) - 1.8/0.9 =
-4.0787. Steps: plug RSS = 1.8, n = 4 into the Gaussian
log-likelihood.

## E07

Inverse [[0.7, -0.3], [-0.3, 0.2]]. x_* = [1, 5]. x_*^T inv x_* =
0.7 - 3.0 + 5.0 = 2.7. Predictive variance 0.45 * 3.7 = 1.665.

## E08

Gaussian log-density is -(y - mu)^2/(2 sigma^2) plus constants.
Maximizing the likelihood minimizes the sum of squares. The
square is the log of the bell.

## E09

At w = 0, p = [0.5, 0.5], (p - y) x = [0, -0.5], mean -0.25.
w_new = 0 - 1*(-0.25) = 0.25. p_new = [0.5, sigmoid(0.25)] =
[0.5, 0.562177]. ll_new = log 0.5 + log 0.562177 = -1.2691,
up from -1.3863.

## E10

sigmoid(0.25) = 1/(1 + e^-0.25). e^-0.25 = 0.7788008. Result
0.5621765. Rounds to 0.562177.

## E11

l = y log p + (1-y) log(1-p), p = sigmoid(w x). dl/dw =
(y/p - (1-y)/(1-p)) p(1-p) x = (y(1-p) - (1-y)p) x = (y - p) x.
Negative gives (p - y) x.

## E12

The data separate at any w > 0 boundary between x = 0 and x = 1.
Larger w makes p_1 closer to 1 and p_0 closer to 0, so the
likelihood keeps rising. No finite maximizer exists. The MLE is
at infinity.

## E13

exp values [1, 0.367879, 0.135335], sum 1.503214. p = [0.665241,
0.244728, 0.090031]. Cross-entropy -log(0.665241) = 0.407606.

## E14

p_k = e^{z_k} / S with S = sum_j e^{z_j}. Sum_k p_k = S / S = 1.
The denominator is shared, so the sum always cancels.

## E15

exp(1000) overflows to inf in float64. inf/inf = NaN for every
class. With the shift, z becomes [0, -1, -2] and the answer is
the E13 triple.

## E16

logit(0.7) = log(7/3) = 0.847298. sigmoid(0.847298) = 0.7. The
canonical link maps the mean to the linear score. Its inverse
maps the score back to the mean.

## E17

Linear regression: Gaussian, identity link. Logistic: Bernoulli,
logit link. Softmax: multinomial, softmax (multivariate logit)
link.

## E18

phi(2) . phi(3) = 1 + 6 + 36 = 43. (1 + 2*3)^2 = 49. Mismatch:
the naive map lacks the sqrt(2) on the cross term. Corrected:
1 + 2*6 + 36 = 49.

## E19

(1 + xz)^2 = 1 + 2xz + x^2 z^2. Read off phi(x) = [1, sqrt(2) x,
x^2]. The sqrt(2) carries the coefficient 2.

## E20

K[0,1] = exp(-0.5 * ||(0,0)-(1,0)||^2) = exp(-0.5) = 0.606531.
K[0,2] = exp(-0.5 * 4) = exp(-2) = 0.135335.

## E21

Eigenvalues 0.391064, 0.965601, 1.643335, all positive: PSD
confirmed. The dual is a concave maximization (bounded above)
only over a PSD kernel. Otherwise the quadratic form can grow
without bound.

## E22

Margins: (-1,0): 1. (0,-1): 1. (1,1): 2. (2,2): 4. Support
vectors: (-1, 0) and (0, -1), the two with margin exactly 1.

## E23

Distance from w . x + b = 0 to the line w . x + b = 1 is
1/||w||. Two sides give 2/||w||. With ||w|| = sqrt(2): 1.414214.

## E24

Noisy point score 0.4, y = -1: hinge = 1 - (-0.4) = 1.4. Total
C = 1: 1.0 + 1.4 = 2.4. C = 100: 1.0 + 140 = 141.0.

## E25

Dual 2a - 4a^2 from the pairwise expansion. d/da: 2 - 8a = 0,
a* = 0.25. Dual value 0.25. Primal 0.5||[0.5,0.5]||^2 = 0.25.
Equal: strong duality.

## E26

w = 0.25*(1,1) + 0.25*(1,1) = [0.5, 0.5]. y(w . x) = 1*(1.0) = 1
for x+ and (-1)*(-1.0) = 1 for x-. Both tight, consistent with
positive alphas.

## E27

The forgotten constraint was sum alpha_i y_i = 0. Without it the
draft used a = 0.5 each, giving dual 0 and primal 1.0: a duality
gap that exposes the missing constraint.

## E28

Active hinge: subgradient -y x = [0.2, 0.2]. w_new = [1,1] -
0.5*[0.2,0.2] = [0.9, 0.9].

## E29

At the kink the subgradient flips sign across the boundary, so a
fixed step jumps from one side to the other forever. The fix is
a diminishing schedule like eta = 1/t, which the Pegasos
analysis justifies.

## E30

Logistic 0.7167, linear SVM 0.7833. The hinge ignores
well-classified points beyond the margin (zero loss), while the
logistic loss keeps penalizing every point's probability error.
Under 15 percent label flips, the flat region wins.

## L01, OLS geometry

(a) Minimum sufficient: OLS picks the (b, w) minimizing the sum
of squared residuals, equivalently the projection of y onto the
column space of X. Strong: adds the normal equations and the
Gauss-Markov license. Red flag: "it finds the true line."
Rubric: full marks for projection language plus the equations.
half for the formula alone. Remediation: U02-C04.

(b)-(h) follow the lesson numbers: w_hat [0.3, 0.8], RSS 1.8.
QR must agree to ~1e-12. Duplicated column makes X^T X
singular, predict the LinAlgError then run. Non-zero residual
sum means no intercept or a coding bug in the residual. The
attack is one outlier at (2, 30). The experiment is OLS vs
Huber on contaminated toys with matched seeds.

## L02, probabilistic reading

(a) Minimum: y = Xw + eps, eps IID N(0, sigma^2). Strong: writes
the likelihood and names sigma2_hat = RSS/n. Red flag:
"Gaussian assumption is harmless." Rubric: likelihood written
correctly. Remediation: U04-C07.

(b)-(h): sigma2_hat 0.45, predictive variance at x = 5 is 1.665.
grid search on sigma must peak at 0.45. Laplace noise predicts
absolute loss. Tight bands mean sigma2_hat missed the true noise.
The model is wrong. Stock returns break Gaussian tails first.
transfer maps l(w) to the U04 likelihood with w as theta.

## L03, logistic

(a) Minimum: P(y=1|x) = sigmoid(w x), fit by Bernoulli MLE.
Strong: writes the log-likelihood and the gradient. Red flag:
"the sigmoid is just a squashing trick." Rubric: likelihood
present. Remediation: U04-C06.

(b)-(h): reproduce 0.25 and -1.2691. 50 steps show ||w||
growing ~log t. Penalty 0.5 caps ||w||. NaN from overflowed exp
or log(0). Calibration attack cites the separability pathology.
sigmoid saturation mirrors the margin flat region: both stop
caring about easy points.

## L04, softmax

(a) Minimum: p_k = e^{z_k}/sum e^{z_j}, loss -log p_true.
Strong: names the simplex and the log-sum-exp trick. Red flag:
"softmax gives calibrated probabilities." Rubric: simplex
stated. Remediation: U04-C08.

(b)-(h): the E13 triple. Jacobian derivation via quotient rule.
[1000,999,998] gives NaN without shift. T = 0.5 sharpens to
[0.867, 0.117, 0.016] (computed 2026-10-06, loss falls to 0.1429). Row sum 1.7
means the normalization was skipped. Calibration attack cites
temperature scaling. Transfer links cross-entropy to U04-C08.

## L05, kernels

(a) Minimum: a kernel is a dot product in some feature space,
computed without visiting it. Strong: states the PSD
requirement. Red flag: "any similarity is a kernel." Rubric:
PSD named. Remediation: C07.

(b)-(h): 43 vs 49, corrected map. Expansion read-off. Gram PSD
test code. N = 1e5 predicts the O(n^2) memory wall (~80 GB for
float64). Unbounded dual means non-PSD kernel. O(n^2) attack on
"kernels solve everything". The experiment fixes wall-clock
budget and compares exact vs Fourier accuracy.

## L06, Gram PSD

(a) Minimum: symmetric with all eigenvalues >= 0. Strong:
equivalently x^T K x >= 0 for all x. Red flag: "diagonal ones
imply PSD." Rubric: eigenvalue or quadratic-form test. Fix: the
U02-C10 bridge.

(b)-(h): eigenvalues as listed. Cholesky fails on the first
non-positive pivot. Gamma = 1e-6 predicts cond ~3e6 (measured
3.227e6) and ||alpha|| ~1e6. The T1 debug solution: gamma too
small, K near ones, fix by scaling gamma to the data
(0.5 works, cond 4.2). "Positive eigenvalues" attack uses the
measured cond. Transfer to U02-C10.

## L07, margins

(a) Minimum: the distance from the boundary to the nearest
point, width 2/||w||. Strong: derives from the projection
formula. Red flag: "margin is the distance between classes."
Rubric: 2/||w|| derived. Remediation: U02-C04.

(b)-(h): margins [1,1,2,4], width 1.4142. Distance formula
derivation. C sweep {0.01: 1.014, 0.1: 1.14, 1: 2.4, 10: 15.0}.
C = 0.01 predicts the boundary rotating toward the majority
geometry (violations cheap). Infeasible means non-separable,
fix is soft margin. Attack cites the noisy toy where max margin
chases the outlier. Transfer to U05-C09 capacity.

## L08, dual

(a) Minimum: maximize over alphas >= 0 with sum alpha_i y_i =
0. W recovered as the weighted sum. Strong writes the dual
objective. Red flag: "alphas are the weights." Rubric:
constraint stated. Remediation: U03-C10.

(b)-(h): a* = 0.25, 0.25 = 0.25. Lagrangian derivation. Grid
search peaks at 0.25. The (0.5,0.5) point predicts alpha 0
(inside the margin). Dual 0 vs primal 1.0 diagnoses the missing
balance constraint. Support-vector attack notes C = 100 makes
almost every noisy point a support vector. Transfer compares
with U03-C10 KKT.

## L09, conditioning

(a) Minimum: kappa = sigma_max/sigma_min, the worst-case error
amplifier. Strong: relative error bound ||dx||/||x|| <= kappa
||db||/||b||. Red flag: "small residual means accurate."
Rubric: amplifier language. Remediation: U02-C12.

(b)-(h): 3.35e6 raw, 2.618 standardized. Bound stated. Float32
predicts the raw solve losing ~7 digits. Cholesky fails on
non-PSD or exactly singular pivots. "Ridge fixes it" attack
cites 1.25e6 still bad. Transfer to U02-C12.

## L10, model choice

(a) Minimum: match the loss to the product: probabilities,
boundary, or nonlinear boundary. Strong: cites the noise and
curvature evidence. Red flag: "SVM is always better." Rubric:
evidence cited. Remediation: U05-C02.

(b)-(h): 0.7167 vs 0.7833 reproduced. Hinge flat region.
Justification. Held-out protocol must pick the winner, not
train. 18 flips predicts both degrade and the gap shrink.
train 1.0/test 0.55 diagnoses memorization via tiny gamma.
"train accuracy picks" attack cites U05-C07. The rerun design
fixes seed and protocol across flip rates.
