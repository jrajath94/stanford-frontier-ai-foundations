# Keys: Lesson 12, Diffusion models

## Breadth recall

E01: q(x_t | x_{t-1}) = N(x_t, sqrt(1 - beta_t)
x_{t-1}, beta_t I), beta_t in (0, 1).

E02: x_t = sqrt(alpha-bar_t) x_0 + sqrt(1 -
alpha-bar_t) epsilon-hat_t, epsilon-hat_t drawn
from N(0, I), alpha-bar_t = product of alpha_s.

E03: p_theta(x_{t-1} | x_t) = N(x_{t-1},
mu_theta(x_t, t), sigma_t^2 I), the mean from a
neural network, the variance fixed.

E04: x_1, ..., x_T are jointly Gaussian given
x_0 (every step is a linear Gaussian map), so
conditioning on x_t and x_0 gives a Gaussian
by the conditional-Gaussian formula.

E05: L_0 (final reconstruction), L_1 ...
L_{T-1} (denoising terms), L_T (dropped, no
parameters).

E06: L_t(theta) = ||epsilon - epsilon_theta(x_t,
t)||^2. Steps: sample x_0, sample t, sample
epsilon, form x_t, gradient step.

## Deep oral ladders

L01: (1) q(x_t | x_{t-1}) = N(sqrt(alpha_t)
x_{t-1}, beta_t I). (2) alpha-bar_50 = 0.7772,
mean 2.6447, std 0.4720. (3) Unroll (14.3):
products of sqrt(alpha_s) give sqrt(alpha-bar_t),
noise terms are independent Gaussians with
total variance 1 - alpha-bar_t. (4) Draw by
single-step recursion and by the closed form,
histograms match. (5) Fixed forward: half the
problem disappears, x_T is known. Learned
forward (VAE): more flexible, needs a learned
prior. (6) alpha_t + beta_t != 1 somewhere, or
beta added without the sqrt(alpha_t)
shrinkage. (7) It is exact only in the limit,
a short schedule starts sampling from the
wrong prior. (8) Assert alpha-bar monotone
decreasing and x_T covariance near I on a
batch before any training.

L02: (1) log p_theta(x_0) >= E_q[log p_theta
(x_0:T) / q(x_1:T | x_0)]. (2) epsilon =
0.8417 recovered, loss = delta^2 under
perturbation. (3) Substitute x_0 = (x_t -
sqrt(1 - alpha-bar_t) epsilon) / sqrt(
alpha-bar_t) into (14.19) and simplify with
alpha-bar_t = alpha_t alpha-bar_{t-1}. (4)
The five steps of SL-03. (5) Exact weights:
correct ELBO, better likelihood. Unweighted:
better sample quality, not a bound. (6) The
unweighted loss ignores small-t precision, or
T is too small for the reverse steps to stay
near-Gaussian. (7) Fixed variance is a
restriction, learning it can tighten the
bound but adds instability. (8) Fix T, the
schedule, and the discretization, report the
weighted ELBO on held-out data for both.

## Analytical exercises

E07: log q(x_t | x_0) = -(1 / (2(1 -
alpha-bar_t))) ||x_t - sqrt(alpha-bar_t)
x_0||^2 + const. Gradient in x_t: -(x_t -
sqrt(alpha-bar_t) x_0) / (1 - alpha-bar_t).
Substitute x_t - sqrt(alpha-bar_t) x_0 =
sqrt(1 - alpha-bar_t) epsilon-hat_t. The
result is -epsilon-hat_t / sqrt(1 -
alpha-bar_t).

E08: beta-tilde_t = beta_t * (1 - alpha-bar_
{t-1}) / (1 - alpha-bar_t). Since alpha-bar_t
= alpha-bar_{t-1} * alpha_t < alpha-bar_{t-1},
we have 1 - alpha-bar_{t-1} < 1 - alpha-bar_t,
so the ratio is below 1 and beta-tilde_t <
beta_t.

## Failure diagnosis

E09: The t = 1 region is the fine-detail
regime. Likely causes: the unweighted loss
under-weights small t, or sigma_1 is too
large. Check the weighted L_0 term and try
the exact-ELBO weights, compare.

## Counterfactual comparison

E10: Team A wins on likelihood (exact ELBO
weights optimize the bound). Team B wins on
perceptual sample quality (the unweighted
loss emphasizes mid-range noise where
structure forms). The split is the
sampling-objective versus training-objective
distinction.

## Research question

E11: Falsifiable claim: sample quality
improves with the number of reverse steps up
to some threshold S* and then plateaus, while
compute grows linearly. S* depends on the
schedule's smoothness.

## Implementation task

E12: Verified by the identities: (14.19) and
(14.22) agree to 1e-10, and the finite-
difference score matches -epsilon / sqrt(1 -
alpha-bar_t) to 1e-6. See lab-06 keys.
