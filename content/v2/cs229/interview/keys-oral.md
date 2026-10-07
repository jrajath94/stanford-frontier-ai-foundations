# Keys: oral defenses

## O01

F1: J = (1/2m) sum (h - y)^2,
theta := theta - alpha (1/m) X^T
(X theta - y).
F2: Points (1,1),(2,2),(3,3),
alpha = 0.1: grad at 0 is
-(14/3), theta_1 = 0.467, loss
falls.
F3: Set X^T(X theta - y) = 0:
theta = (X^T X)^{-1} X^T y.
F4: Both reach theta = 1 within
tolerance.
F5: GD O(n d) per step, many
steps, normal equations O(n d^2 +
d^3), one shot but unstable when
X^T X is ill-conditioned.
F6: alpha too large, or features
unscaled (one coordinate
dominates).
F7: Least squares is the MLE only
under Gaussian noise with constant
variance. Real noise is rarely
either, the justification is a
modeling choice, not a fact.
F8: Time GD to 1e-4 tolerance vs
one Cholesky solve, compare
residuals and wall clock. GD wins
on memory, Cholesky on exactness.

## O02

F1: sigmoid(z) = 1/(1+e^-z),
L = prod h^y (1-h)^(1-y).
F2: Points (0,0),(1,1), theta = 0:
h = 0.5, grad = sum (y - h) x =
0.5 * [1] on the second point.
F3: d/dtheta log L = sum (y_i -
h_i) x_i via dh/dz = h(1-h).
F4: loss = max(z,0) - z y +
log(1 + e^-|z|), test z = +-1000
for finiteness.
F5: Logistic: discriminative, no
input model, needs more data.
GDA: generative, Gaussian
assumption, better on small data
when the assumption holds.
F6: Separable data: the MLE is at
infinity. Fix: regularization or
early stopping.
F7: They are calibrated only if
the model is right. Logistic
outputs are scores until a
calibration study says otherwise.
F8: Bin predicted probabilities,
plot fraction of positives per
bin, report ECE.

## O03

F1: Functional margin y(w^T x +
b), geometric margin divides by
||w||.
F2: Points -1 (y=-1), +1 (y=+1):
w = 1, b = 0, margin 1.
F3: L = (1/2)||w||^2 - sum
alpha_i [y_i(w^T x_i + b) - 1],
set derivatives to zero, get the
dual in alpha.
F4: alpha = [0.5, 0.5], check
y_i(w^T x_i + b) = 1 for both.
F5: Primal scales with d, dual
with n. Dual wins for kernels and
small n, primal wins for large n
with linear kernels.
F6: C too small (everything is a
margin violator) or massive
overlap/noise. Raise C or accept
the noise.
F7: Support vectors are the points
the solution depends on, not the
most informative ones. An outlier
can be a support vector.
F8: Grid over (C, gamma) with
nested CV, report the selected
pair and the CV spread.

## O04

F1: Given upstream gradient dL/dy,
return dL/dx and dL/dW.
F2: Net y = v * relu(w x), x = 1,
w = 2, v = 3: forward 6,
backward dy/dv = 2, dy/dw = 3,
dy/dx = 6.
F3: dL/dW = (dL/dY)^T X form via
differentials: dY = dW X gives
dL = tr((dL/dY)^T dW X).
F4: Relative error below 1e-5 on
random small nets.
F5: Reverse-mode: O(1) forward
cost for scalar loss, needs stored
activations. Forward-mode: O(d)
per input direction, no storage.
F6: Dead ReLUs (zero gradient
everywhere below) or a detached
graph (no_grad / wrong wiring).
F7: Depth helps only with the
architecture and optimization to
support it: residuals, init,
normalization. Depth alone gives
vanishing gradients.
F8: Train a shallow net to the
same loss, if it matches, the deep
net's failure is optimization,
not capacity. Then add one
residual link and watch the
gradient norms per layer.

## O05

F1: E[(y-h)^2] = bias^2 +
variance + noise.
F2: Small k: low bias, high
variance. Large k: high bias, low
variance.
F3: Add and subtract E[h]: cross
term vanishes, three terms
remain.
F4: Resample the training set B
times, fit, measure spread of
predictions at fixed x.
F5: Classical: U-curve in
capacity. Double descent: second
descent past the interpolation
threshold in overparameterized
regimes.
F6: Overfitting (fix: regularize,
early stop, more data) or
distribution shift (fix: nothing
in the training loop, fix the
data pipeline).
F7: More data from the wrong
distribution hurts. More data
helps when it matches deployment.
F8: Learning curves (error vs n),
ablated capacity, and a noise
floor estimate from repeated
labels.

## O06

F1: log p(x) >= E_q[log p(x,z)]
- E_q[log q(z)]: Jensen on the
marginal.
F2: Points {0, 10}, init means
{2, 8}: E step assigns 0 to comp
1, 10 to comp 2, M step moves
means to {0, 10}.
F3: The E step tightens the bound
at the current theta, the M step
maximizes the bound, the true
likelihood is above the bound, so
it cannot decrease.
F4: Likelihood sequence is
nondecreasing within tolerance.
F5: k-means: hard assignments,
spherical. EM: soft, full
covariances, likelihood-based.
F6: One component collapses onto
a point: likelihood goes to
infinity. Fix: variance floor or a
prior (MAP).
F7: EM finds a local maximum. The
MLE claim needs the global
maximum, which EM does not
guarantee.
F8: Multiple random inits plus
k-means init, keep the best
likelihood, report the spread.

## O07

F1: q(x_t|x_{t-1}) = N(sqrt(1 -
beta_t) x_{t-1}, beta_t I).
F2: beta = 0.5 both steps, x_0 =
2: x_1 ~ N(1.41, 0.5), x_2 ~ N(x_1
scaled, ...): variance
accumulates, mean shrinks.
F3: The ELBO's per-step KL terms
reduce to squared denoising errors
errors under the Gaussian
parameterization.
F4: Loss finite and positive,
gradient flows to the noise net.
F5: Epsilon-prediction: stable
targets across t. x0-prediction:
harder at high noise. Same
optimum in theory, different
optimization in practice.
F6: Schedule bug (signal gone too
early) or the reverse net
predicting the mean (mode
collapse to gray).
F7: The ELBO is a bound, and the
weighting across t is a choice.
Models with worse ELBO can make
better samples.
F8: FID or precision/recall on
held-out data, plus bits-per-dim
on a held-out set with the exact
ELBO weighting stated.

## O08

F1: softmax(QK^T/sqrt(d_h))V
with the causal mask.
F2: Rows [1,0,0],
[0.3302,0.6698,0],
[0.4011,0.4011,0.1978].
F3: Stack the per-position rules:
rows of QK^T are the score
vectors, softmax acts row-wise.
F4: Upper triangle exactly zero,
rows sum to 1.
F5: MHA: full quality, full
cache. GQA: n_g groups, cache
scales with n_g. MQA: one KV
head, smallest cache, quality
risk.
F6: The causal mask has a gap or
applied after the softmax. Future
positions leak.
F7: It assumes independent q, k
coordinates at init. Real q, k
are correlated, the scaling is a
heuristic that works, not a
derived constant.
F8: Per-layer cache = 2 T n_g d_h
bytes, multiply by layers and
batch, provision for the max
concurrent T, or page the cache.

## O09

F1: V^pi(s) = E[sum gamma^t R(s_t)
| s_0 = s, pi].
F2: V(B) = 1/(1-0.9) = 10, V(A) =
0.9 * 10 = 9.
F3: Split t = 0: R(s) + gamma
E_{s'}[V^pi(s')].
F4: Errors 10, 9, 8.1, ... ratio
0.9 per sweep.
F5: Value iteration: cheap
sweeps, asymptotic. Policy
iteration: linear solve per
iteration, exact in finite time.
F6: gamma >= 1, or a sign error
in the max (minimizing reward).
F7: Real problems never give you
P_sa. Estimation error dominates,
and the 0/0 fallback is dangerous
at scale.
F8: Coverage policy rollouts,
count accumulation per (19.5),
warm-started value iteration,
held-out start-state evaluation.

## O10

F1: r_t = pi_theta(a_t|s_t) /
pi_old(a_t|s_t).
F2: (1, 1.3): min(1.3, 1.2) = 1.2.
(-1, 0.5): min(-0.5, -0.8) = -0.8.
F3: grad E[f] = E[(grad log P)
f] via grad P = P grad log P.
F4: Monte Carlo mean 0.25 within
tolerance at theta = 0.
F5: Vanilla: one update per
batch. PPO: several updates per
batch via the ratio, clipped to
stay near the old policy.
F6: The surrogate is local:
stale data moved theta off the
trust region. Check KL and the
clipped fraction.
F7: PPO reuses near on-policy
data with a bounded excursion,
the state distribution is never
corrected. "Approximately
on-policy" is the honest label.
F8: Few epochs per batch (3-4),
resample often, guards are KL
from theta_old and the clipped
fraction.
