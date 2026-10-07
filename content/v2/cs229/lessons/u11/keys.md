# Keys: Lesson 11, Variational inference, PCA, ICA

## Breadth recall

E01: The posterior p(z|x) under parameters theta has no closed
form when the generative model is a neural net
(the normalizer integrates over a nonlinear
mapping).

E02: max_{Q in Q} max_theta ELBO(Q, theta), the
ELBO maximized over a tractable Q family and the
model parameters jointly.

E03: Write z ~ N(mu, sigma^2) as z = mu + sigma
xi with xi ~ N(0, 1). The expectation moves to
xi, which is parameter-free, so gradients flow
through the deterministic path.

E04: PCA maximizes the variance of the projected
data (equivalently, minimizes reconstruction
error) over unit vectors.

E05: x = A s: observed mixtures x are an unknown
linear mixing A of independent sources s. Find W
= A^{-1} to recover s = W x.

E06: Permutation (source order unrecoverable) and
scaling (source amplitudes unrecoverable).

## Deep oral ladders

L01: (1) z ~ N(0, I), x|z ~ N(g(z, theta),
sigma^2 I). (2) The encoder learns angle-ordered
codes. The decoder traces the circle. (3) ELBO =
E_Q[log p(x|z)] - KL(Q || p(z)). (4) Sample xi,
form z = mu + sigma xi, Monte-Carlo the ELBO,
backprop. (5) On a tractable mixture, VI with a
rich Q family approaches EM. With a diagonal
Gaussian Q it is looser. (6) KL collapse: the
decoder ignores z. (7) Diagonal Gaussians cannot
represent correlated posteriors. The bound stays
loose. (8) Track KL and reconstruction
separately. Alert on KL -> 0.

L02: (1) max_{||u||=1} u^T Sigma u. (2)
Eigenvalues 2.26, 1.62, 0.04. Keep 2, retain
98.9 percent. (3) Lagrangian gives Sigma u =
lambda u. (4) Standard implementation with the
variance-kept ratio. (5) ICA wins on
non-Gaussian mixed sources. PCA wins when the
goal is compression of Gaussian-ish data. (6)
Features were not standardized, so rescale. (7)
PCA is linear. Manifolds need nonlinear methods.
(8) Keep components until 95 percent (or a
domain-chosen) variance threshold.

## Analytical exercises

E07: ELBO = E_Q[log p(x, z) - log Q] = E_Q[log
p(z|x) + log p(x) - log Q] = log p(x) - KL(Q ||
p(z|x)).

E08: max_{||u||=1} u^T Sigma u. Lagrangian L =
u^T Sigma u - lambda (u^T u - 1). Gradient: 2
Sigma u - 2 lambda u = 0, so Sigma u = lambda u.
The maximum value is lambda, the largest
eigenvalue.

## Failure diagnosis

E09: Posterior collapse or a disconnected latent
space: the decoder memorized modes and the
latent geometry carries no smooth structure.
Check KL. Try beta < 1 warmup, KL annealing, or a
weaker decoder.

## Counterfactual comparison

E10: PCA wins when features are correlated and
Gaussian-ish and the goal is compression or
denoising. ICA wins when the data is a linear
mixture of non-Gaussian independent sources
(audio, EEG) and the goal is separation.

## Research question

E11: Falsifiable claim: as beta increases,
a disentanglement metric rises while
reconstruction error rises monotonically.

## Implementation task

E12: Verified by the eigenvalue identity:
reconstruction MSE equals the sum of dropped
eigenvalues within numerical tolerance, and the
whitened covariance is the identity.
