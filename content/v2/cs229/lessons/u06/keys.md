# keys.md, U06 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate.

## E01

Functional margin: y^{(i)}(w^T x^{(i)} + b).
Geometric margin: functional margin / ||w||.

## E02

Scaling (w, b) scales the functional margin without
changing predictions, so maximizing it is meaningless
without a normalization.

## E03

min (1/2)||w||^2 s.t. y^{(i)}(w^T x^{(i)} + b) >= 1.

## E04

max_alpha sum alpha_i - (1/2) sum_{i,j} y^{(i)} y^{(j)}
alpha_i alpha_j <x^{(i)}, x^{(j)}> s.t. alpha_i >= 0,
sum alpha_i y^{(i)} = 0.

## E05

alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] = 0.

## E06

Only the box: 0 <= alpha_i <= C. The objective W is
unchanged.

## E07

By symmetry b = 0, w = [c, c]. Constraints give
2c >= 1. Minimize c^2: c = 0.5. w = [0.5, 0.5],
margin = 1/||w|| = sqrt(2) = 1.4142.

## E08

dL/dw = w - sum_i alpha_i y^{(i)} x^{(i)} = 0, so
w = sum_i alpha_i y^{(i)} x^{(i)}. dL/db = -sum_i
alpha_i y^{(i)} = 0, the linear constraint.

## E09

Causes: (1) a bug in the clipping bounds [L, H].
(2) stale cached errors used for the pair update.
W must not decrease on an exact pair step.

## E10

Team B generalizes better. Hard margin fits the
training gap exactly, including its noise. Small C
trades a little training margin for robustness. On
clean separable data the difference is small. On
noisy data B wins.

## E11

Setup: two Gaussians, separation s in {1, 2, 4},
n = 500. Fit hard-margin SVM, count alpha_i > 1e-6.
Claim: the support fraction falls as s grows.
Falsified if the fraction stays flat.

## E12

Reference: pair loop per notes 6.8.2. Checks: W
non-decreasing, sum alpha_i y^{(i)} = 0 after each
pair update, KKT within 0.01 at convergence, and the
(w, b) matches a QP solver on the same 20 points.

## L01 key

(1) Functional: signed score. Geometric: distance.
(2) Score -1, geometric -0.2.
(3) ||2w|| = 2||w|| cancels the factor 2 in the
numerator.
(4) Code: scale (w, b), check geometric margins
unchanged.
(5) Margin is a distance. Logistic gives a
probability. Both rank confidence.
(6) Missing: the normalization (||w|| = 1 or
functional margin = 1).
(7) Not always: a wide margin on mislabeled data
memorizes noise. Soft margin exists for this.
(8) Fraud: geometric margin with asymmetric costs.
consider class-weighted C.

## L02 key

(1) L = (1/2)||w||^2 - sum alpha_i [y^{(i)}(w^T
x^{(i)} + b) - 1].
(2) a = 0.5, w = 1, b = 0.
(3) Minimize L over w, b. Substitute w and the zero
sum back.
(4) Check alpha_i * (margin_i - 1) near zero for all
i.
(5) Same (w, b). Dual additionally shows which
examples matter.
(6) Cause: the linear constraint pins one alpha.
update pairs.
(7) Invalid kernel: the dual may be non-convex. SMO
can diverge. Test PSD first.
(8) n = 1e5: SMO with kernel caching, or primal SGD
on the hinge loss. Not a dense QP.
