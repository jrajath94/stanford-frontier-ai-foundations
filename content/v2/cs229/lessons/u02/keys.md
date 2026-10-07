# keys.md, U02 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate.

## E01

theta_j := theta_j + alpha (y^{(i)} - h_theta(x^{(i)}))
x^{(i)}_j. Factors: learning rate, prediction error, feature
value.

## E02

X^T X theta = X^T y. Needs X^T X invertible: n independent
columns, at least n independent examples.

## E03

The residuals are orthogonal to every feature column. The
prediction is the closest point in the column space.

## E04

(1) Duplicate or linearly dependent feature columns. (2) Fewer
independent examples than features.

## E05

Gaussian noise gives a likelihood whose log is a negative
squared error plus a constant. Maximizing it minimizes the
squared error.

## E06

w^{(i)} = exp(-(x^{(i)} - x)^2 / (2 tau^2)). tau sets the
neighborhood width: small tau trusts only close points.

## E07

X^T X = [[2, 3], [3, 5]], X^T y = [8, 13]. det = 10 - 9 = 1.
theta = [[5, -3], [-3, 2]] @ [8, 13] = [40 - 39, -24 + 26] =
[1, 2]. Check: 1 + 1 = 2? Predictions: 1+1*1 = 2, not 3.
Hmm, recompute: X @ [1,2] = [1+2, 1+4] = [3, 5] = y. Correct.

## E08

h = 0. Error = 5 - 0 = 5. x = [1, 2].
theta_0 := 0 + 0.5 * 5 * 1 = 2.5.
theta_1 := 0 + 0.5 * 5 * 2 = 5.0.

## E09

Diagnosis: the two columns are nearly collinear (feet vs
meters differ by a constant factor 3.28), so kappa explodes
and theta splits the weight arbitrarily. Fix: drop one
column or standardize, then use lstsq.

## E10

Team B is safer: lstsq uses a stable factorization and never
forms the inverse, so kappa = 1e13 is survivable. Team A
should check kappa of X^T X before shipping and switch to a
factorization if it is large.

## E11

Setup: X with exact rank r, y = X beta + noise. Compare pinv
(minimum-norm, zero penalty on the null space) vs ridge with
lambda from validation. Claim: ridge wins when the true beta
has large null-space components that validation can shrink.
pinv wins when the truth is exactly minimum-norm. Falsified
if one dominates across all synthetic truths.

## E12

```python
import numpy as np

def lwlr_predict_vec(xq, X, y, tau):
    d = np.linalg.norm(X[:, 1:] - xq, axis=1)
    w = np.exp(-(d ** 2) / (2 * tau ** 2))
    W = np.diag(w)
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(np.r_[1.0, xq] @ theta)

rng = np.random.default_rng(0)
X = np.column_stack([np.ones(50), rng.normal(size=(50, 2))])
y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.1, 50)
th_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
p_big = lwlr_predict_vec(X[0, 1:], X, y, tau=1e9)
p_ref = float(X[0] @ th_lstsq)
assert abs(p_big - p_ref) < 1e-6, (p_big, p_ref)
```

## L01 key

(1) J(theta) = (1/2m) sum (h - y)^2.
(2) See E08.
(3) dJ/d theta_j = (h - y) x_j by the chain rule. Subtract
alpha times it.
(4) Batch code in SL-01. Check cost falls and gradient norm
shrinks.
(5) Batch: O(mn) per step, steady descent. Stochastic: O(n)
per step, fast early progress, oscillates near the minimum.
(6) Cause: alpha too large. Fix: shrink alpha, or normalize
features.
(7) The convexity claim is false outside quadratic costs, for
example neural nets (U07).
(8) m = 1e9: stochastic or mini-batch. A full batch pass per
step is unaffordable.

## L02 key

(1) X^T X theta = X^T y.
(2) See E07.
(3) Expand J, use grad theta^T A theta = 2 A theta and grad
b^T theta = b, set to zero.
(4) Check norm(X.T @ (y - X @ theta)) near zero.
(5) On kappa = 1e8: normal equations via explicit inverse
fail. QR and SVD stay accurate. The inverse squares the
condition number.
(6) Repairs: drop dependent columns. Use pinv. Add ridge
penalty.
(7) Minimum norm is the wrong prior when the truth uses
large weights, for example a known physical constant. Then
ridge toward the prior, or keep the dependent features out.
(8) Pipeline: standardize, compute kappa via SVD, log it,
refuse or fall back to lstsq above threshold 1e10, verify
residual orthogonality.
