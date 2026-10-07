# Notation and shapes: math-genmodels

Date: 2026-10-06. Baseline: October 6, 2026.

Every symbol below is defined before its first lesson use. Lessons
cite this page instead of redefining symbols.

## Probability

- x: one data point. X: the random variable. x_i: the i-th sample.
- p(x): a probability density or mass function. p_data(x): the true
  data law. p_theta(x): the model law with parameters theta.
- q(z|x): an approximate posterior (encoder). p(z): a prior over
  latents. p(x|z): a likelihood (decoder).
- E_{x~p}[f(x)]: expectation of f under p. E_n: empirical mean over n samples.
- D_KL(p||q): KL divergence from p to q, in nats (natural log).
- H(p): entropy of p. H(p,q): cross-entropy.
- N(mu, sigma^2): Gaussian with mean mu and variance sigma^2.
- supp(p): support of p, the set where p(x) > 0.

## Shapes

- D: data dimension. x in R^D. d: latent dimension. z in R^d.
- n: sample count. X in R^{n x D}: one row per sample.
- theta: parameter vector in R^P. P: parameter count.
- W in R^{m x n}: matrix with m rows, n columns. b in R^m: bias.
- J in R^{D x D}: Jacobian of a map f: R^D -> R^D.
- T: diffusion step count. t in {1..T} or continuous t in [0,1].
- K: codebook size. e_k in R^d: the k-th code vector.
- V: vocabulary size for discrete tokens. L: sequence length.

## Calculus and linear algebra

- nabla_x f: gradient of f with respect to x, shape of x.
- nabla_x log p(x): the score, shape of x.
- det J or |det J|: Jacobian determinant, a scalar volume factor.
- diag(v): diagonal matrix with v on the diagonal.
- ||x||_2: Euclidean norm. <x,y>: inner product.

## Objectives

- L(theta): a loss (minimize). J(theta): an objective (context decides sign).
- ELBO: evidence lower bound. log p(x) >= ELBO.
- E_q: expectation under q. MC: Monte Carlo.

## Code conventions

- numpy as np. float64 for toy math unless stated.
- rng = np.random.default_rng(seed). Seeds are printed with every figure.
- Shapes appear in comments as # (n, D).

## Units

- Probabilities are unitless in [0,1]. Densities carry 1/[x] units and
  can exceed 1. Nats for log-base-e information. Bits for log-base-2.
- Compute cost in FLOPs. Memory in bytes. Time in wall seconds on CPU
  unless stated.
