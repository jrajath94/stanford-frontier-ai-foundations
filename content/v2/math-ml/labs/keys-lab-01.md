# Lab keys 01, linear geometry computations

Date: 2026-10-06. Computed values, numpy 1.26.4, float64.
Executed by compute_run2_labs.py (fix builder, 2026-10-06). Every
number below is printed by that script.

## Task 1

(a) Singular values: [16.8481034, 1.06836951, 4.41842475e-16].
(b) Rank 2. Evidence: the sigma gap between 1.068 and 4.4e-16
(numerical dust). Two nonzero values, one dust.
(c) Nullspace direction [-0.40824829, 0.81649658, -0.40824829]
(= [-1, 2, -1]/sqrt(6)). A @ n residual 1.0175e-15, within 1e-12.

## Task 2

(a) p = [1, 0], r = [0, 1].
(b) r dot a = 0.0. P + r = [1, 1] = b. Both checks pass.
(c) a = [0, 0] gives [nan, nan] with RuntimeWarning (invalid value
in scalar divide), not an exception. The failure is silent NaN, which
is why the lesson demands an explicit guard.

## Task 3

(a) kappa = 4002.000750124839. X = [2, 0].
(b) Relative input move 3.535533906426927e-07. Relative output move
7.071067812854244e-04.
(c) Bound check: 7.07e-04 <= 4002.0 * 3.54e-07 = 1.414e-03. Holds.
(d) Strong answer: "The residual is zero because the solver did its
arithmetic right. The answer is still sensitive because kappa = 4002
amplifies input noise 2000-fold. Check kappa, not just the residual."
