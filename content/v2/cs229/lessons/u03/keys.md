# keys.md, U03 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate.

## E01

g(t) = 1/(1 + e^{-t}). g'(t) = g(t)(1 - g(t)).

## E02

l(theta) = sum_i [y^{(i)} log h^{(i)} + (1 - y^{(i)})
log(1 - h^{(i)})], h^{(i)} = g(theta^T x^{(i)}).

## E03

The update looks the same but h_theta is a nonlinear
function of theta^T x here. LMS is linear least squares.
the objectives, the geometry, and the convergence stories
differ.

## E04

d l_ce / d t = phi - e_y.

## E05

Natural parameter eta: the dial inside the exp. Sufficient
statistic T(y): the function of y the dial multiplies.

## E06

(1) Family: Bernoulli. (2) Goal: E[y|x] = phi. (3) Link:
eta = theta^T x, so phi = g(theta^T x).

## E07

Exponentials: e^3 = 20.0855, e^1 = 2.7183. Sum 22.8038.
phi = [0.8808, 0.1192]. Loss: -log(0.1192) = 2.1269.
Gradient: phi - [0, 1] = [0.8808, -0.8808].

## E08

1 = int b(y) exp(eta T(y) - a(eta)) dy. Differentiate in
eta: 0 = int b(y) exp(...) (T(y) - a'(eta)) dy =
E[T(y)] - a'(eta). So a'(eta) = E[T(y)].

## E09

Diagnosis: sigmoid saturation underflows 1 - g to 0, then
log 0 = -inf, then nan gradients. Fixed line: use
y * softplus(-t) + (1 - y) * softplus(t) with
softplus(t) = max(t,0) + log1p(exp(-|t|)).

## E10

Team A: n = 200, Newton forms and inverts a 200x200
Hessian cheaply and converges in few steps. Team B:
n = 200,000, the Hessian never fits in memory, so L-BFGS
approximates curvature from gradients. Swapped: A with
L-BFGS still works but slower per step count. B with
Newton runs out of memory.

## E11

Setup: two Gaussian blobs with center distance d (the
margin knob), labels separable for all d > 0. Run
gradient ascent with fixed alpha, log ||theta|| per
iteration. Claim: ||theta|| grows like log(iterations)
and the rate increases as d shrinks. Falsified if the
norm plateaus on any separable setting.

## E12

```python
import numpy as np

def ce_loss(t, y):
    t = t - np.max(t)
    log_z = np.log(np.sum(np.exp(t)))
    return log_z - t[y]

def ce_grad(t, y):
    t = t - np.max(t)
    e = np.exp(t)
    p = e / e.sum()
    g = p.copy()
    g[y] -= 1.0
    return g

t = np.array([3.0, 1.0, 0.5])
assert abs(ce_loss(t + 7.0, 1) - ce_loss(t, 1)) < 1e-12
h = 1e-6
num = np.array([(ce_loss(t + h * ei, 1) - ce_loss(t - h * ei, 1)) / (2 * h)
                for ei in np.eye(3)])
assert np.allclose(num, ce_grad(t, 1), atol=1e-6)
```

## L01 key

(1) P(y=1|x) = g(theta^T x).
(2) See SL-01 computed example.
(3) Chain rule through the log likelihood. The h(1-h)
cancels to (y - h) x.
(4) Stable loss code in SL-02. Check finiteness at
t = +-1e6 and gradient vs finite differences.
(5) Newton: O(mn^2 + n^3) per step, few steps. Gradient
ascent: O(mn) per step, many steps.
(6) Cause: naive log(sigmoid(t)) underflow. Fix: stable
softplus form.
(7) Email labels are not IID: threads, bursts, and
adversarial senders correlate.
(8) n = 1e6, m = 1e8: stochastic gradient, never Newton.

## L02 key

(1) p(y, eta) = b(y) exp(eta^T T(y) - a(eta)).
(2) eta = log(0.7/0.3) = 0.847.
(3) eta = log(phi/(1-phi)) inverts to phi = 1/(1+e^{-eta}).
(4) h(x) = exp(theta^T x). Predict with the log link.
(5) Canonical link keeps the (y - h) x gradient. Other
links add a link-derivative factor.
(6) Broken step: the link. Identity link lets the mean go
negative. Use the log link.
(7) Linear eta fails when the true log-odds bends with x.
then add features (U05) or go nonlinear (U07).
(8) Waiting times are positive continuous: exponential or
gamma family with log link.
