# U10 interview key

## B1-B6

B1. 6ND: 2 forward + 4 backward per param per token.
B2. The data entropy floor, loss -> E as compute -> inf.
B3. Log-log line, slope negated is the exponent.
B4. Fixed C, the minimum is the optimal N/D split.
B5. Written prediction + band + n + rule + date, before the run.
B6. Form change, noise regime, range.

## L1

L1.1 log(L-E) = log A - a log(C/C0).
L1.2 a=0.340, A_hat=0.499, RMS 0.0031.
L1.3 Normal equations on [1, x], slope = -a.
L1.4 lstsq, noiseless data recovers a exactly.
L1.5 Nonlinear better when E uncertain, debug: wrong E biases
slope, critique: iid noise, experiment: perturb E +-0.05.

## L2

L2.1 N_opt(C) minimizes loss at fixed C. D/N the ratio.
L2.2 N=7.00e10, D=1.40e12, ratio 20.0.
L2.3 First-order condition on the joint toy law gives N^2 =
(N0/D0)(C/6).
L2.4 `allocate`. 6*N*D reproduces C.
L2.5 Overtrain when inference dominates, debug: data cap binds.
critique: symmetric toy, experiment: constrained allocation.

## E1

(a) 2 orders of magnitude. (b) Law may change form, noise is
correlated across runs. (c) Intermediate-scale runs at 10x with
registered bands before the 100x bet. Rubric: (a) 1 pt, (b) 2
pts, (c) 1 pt. Red flag: quoting the point prediction.

## E2

(a) X: 6*30e9*600e9=1.08e23. Y: 6*7e9*1.4e12=5.88e22. (b) Y:
inference cost dominates at 1e14 tokens. 0.2 nats buys 4x cheaper
serving. (c) The quality bar: if the product needs < 2.5, X wins.
Rubric: (a) 1 pt, (b) 2 pts, (c) 1 pt.

## D1

Bug: a = coef[1] takes the slope, not its negation. The exponent
is -coef[1]. Fix: a = -coef[1]. Rubric: find 2 pts, fix 1 pt,
state the sign convention 1 pt.

## S1

Allocate N = C/(6*5e11) = 1.96e11? No: D capped at 5e11 gives N =
5.88e23/(6*5e11) = 1.96e11, D/N = 2.55. Defend: the data boundary
binds, the 20-ratio optimum is unreachable, the constrained
optimum sits at the boundary.

## S2

Report a=0.34 with the bootstrap CI and the range warning:
"fitted over half an order, extrapolation not supported." Refuse
the bare point estimate.

## R1

Gaps: (1) no residuals: fit quality unknown, plot them. (2) no E
discussion: exponent confounded, fit E jointly. (3) 1 order of
magnitude: range too narrow, widen or bound claims.
