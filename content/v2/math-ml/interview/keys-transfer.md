# Keys, unfamiliar transfer sets

Date: 2026-10-06. All numbers computed 2026-10-06 unless derived
by hand below. Rubric: each item 4 points (claim 2, justification
2). No partial credit for the claim without the justification.

## A1

||w|| = 5. Geometric margin width 2/5 = 0.4. Scaling w, b by 1/5:
the functional margins change (divided by 5). The geometric
margin, the boundary, and all classifications stay identical.
The lesson: functional margin is a parameterization artifact.

## A2

Doubling features halves the optimal w (same boundary, smaller
weights) and quarters the Gram scale. The dual alphas rescale
accordingly but the boundary is unchanged in exact arithmetic.
In float64 the raw kappa 3.35e6 grows ~4x and the solve
degrades. Test accuracy barely moves. The alphas become
unreliable. Fix: standardize before fitting.

## A3

Gram shape (4, 4). Rank at most 2 (rank(X X^T) = rank(X) <= 2).
The U02 concept is rank/nullspace: two zero eigenvalues. The
fix is the pseudoinverse or ridge. Cost: ridge biases the
solution, pseudoinverse costs the SVD.

## B1

Below. Laplace 0.6667 pulls toward 0.5 (prior pseudo-counts).
The unpenalized logistic MLE on 3/4 heads is p_hat = 0.75, the
sample proportion: above 0.6667, and overconfident. The bias
direction: MLE has no shrinkage. Laplace does. (U05-C04: the
same bias-variance trade.)

## B2

Log odds: log p(x|1)/p(x|0) = 2x - 2. Rule: 2x - 2 = 0, so w =
2, b = -2, decide class 1 iff x > 1. Error: P(x < 1 | class 1)
= Phi(-1) = 0.1587, same for class 0 by symmetry. Total 0.1587,
matching the lesson's Bayes floor (U04c-SB18).

## B3

diag(p) - p p^T = Cov of a one-hot draw with probabilities p:
for one-hot e_K, E[e_K e_K^T] - E[e_K]E[e_K]^T. Covariances are
PSD (U02-C10: x^T Cov x = Var(x^T e) >= 0). Reuses U04a-SB04
(variance as expected squared deviation).

## C1

The failure mode is U05-C07/C08: selecting by train error
(in-sample optimism). The unsupervised analogue of the fix is
held-out inertia (or cross-validated inertia): fit centers on
train, score new points.

## C2

U05-C12: distribution shift (or plain overfit of the variance
cutoff). Keeping more components hurts: it fits train noise
directions that do not repeat. The fix: choose k on
held-out reconstruction error, not train variance share.

## C3

Eigenvalue 3.83 is the noisy sensor's variance, not signal:
k = 2 spends a direction modeling noise (std 3.0 on sensor 3).
Train MSE falls, but the second direction will not repeat on
new data. Falsification: held-out MSE at k = 2 exceeds k = 1
(the learner runs this. The prediction is the grade).

## D1

As ||w|| grows on separable data, p_i -> 0 or 1, so W_ii =
p_i(1-p_i) -> 0. The Hessian X^T W X goes singular. The Newton
step becomes unstable or undefined. Prediction: Newton breaks
exactly where gradient descent merely slows. (Measured
behavior on ill-conditioned solves: see C11.)

## D2

Subgradient set at z = 1: the interval [-1, 0] (left slope -1,
right slope 0, any convex combination). Constant eta jumps
across the kink: the sign flips each step, so the iterate
bounces instead of settling.

## D3

Coordinate descent (U03, the Gauss-Seidel analogue for
optimization). Per-update cost O(n): the update touches one
row/column of the Gram.

## E1

U01-C07 (axes/shapes): the contract says (n, d). U02-C05
(matrix transforms): X^T w is defined with shape (d,) and runs
without error, silently transposing the problem.

## E2

U01: log(0) from underflow (probabilities rounded to zero).
U04: the data have zero probability under the model (support
mismatch). U06: softmax without the max shift overflowed to
inf/inf = NaN, and log(NaN) propagates.

## E3

What changes: the shuffle consumes the RNG stream first, so the
train and validation draws differ from the original run. What
stays: reruns of the new script are still reproducible (same
seed, same order). U01-C12: seed fixes the sequence, but the
sequence depends on call order.

## F1

The limit is shared spherical variance sigma^2 -> 0 (all
components, equal). In that limit the responsibilities
(C06) become 0/1: the E-step is trivial (nearest center), and
EM becomes Lloyd's algorithm.

## F2

Shared: the Gaussian bump exp(-||x - x_i||^2/(2h^2)) as a
similarity. Differs: Parzen fixes the bandwidth h and the
weights (1/n) with nothing optimized. The kernel machine
optimizes the alphas (and selects gamma) against a margin
objective.

## F3

Mixture likelihoods are nested probabilistic models: the
likelihood ratio has a (nonstandard but real) statistical
footing via held-out likelihood comparison. Inertia is not a
likelihood: k+1 centers always fit train better by
construction, with no probabilistic penalty for complexity, so
only the held-out elbow (a heuristic) is available.
