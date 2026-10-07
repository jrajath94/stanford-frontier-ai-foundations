# Interview bank: U02 (questions)

Unit: math-genmodels-U02. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u02.md.

## Breadth (6)

Q1. A model has a hidden variable z. Write the formula for p(x) in
terms of p(z) and p(x|z). What changes when z is continuous?
Q2. Define the posterior. How does it differ from the likelihood?
Q3. State the ELBO. Name its two terms and what each rewards.
Q4. What is the variational gap, and when is it zero?
Q5. Forward KL versus reverse KL: which one does variational
inference minimize, and what fitting behavior does that imply?
Q6. What does the reparameterization trick buy you, and for which
latent types does it fail?

## Deep ladders (2 x 5)

L1 (latent models end to end):
1. Define a latent variable model with the two-bag toy.
2. Compute the marginal P(R) and the posterior P(A|R) by hand.
3. Derive the ELBO from Jensen. Point to where concavity is used.
4. Implement the ELBO for the toy in numpy and verify ELBO + gap =
   log p(R).
5. Compare exact marginalization against the ELBO on cost and on
   the n = 10,000 regime. When would you still marginalize exactly?

L2 (gradients and variance):
1. Define the reparameterization trick in one equation.
2. Compute the true gradient dE/dmu for E[z^2], z ~ N(1, 4), by hand.
3. Explain the score-function estimator and why its variance is
   higher.
4. State the Monte Carlo 1/sqrt(K) law and use the measured std
   values to pick K for std 0.05.
5. Debug: a reparameterized gradient returns exactly 0.0 every step.
   Name the root cause class and the verification test.

## Analytical exercises (2)

A1. Prove log p(x) = ELBO + KL(q(z)||p(z|x)) starting from the
definition of KL. State the support condition on q.
A2. A discrete latent has K = 10^6 states. The ELBO needs E_q over
all states. Propose a tractable estimator, name its bias and
variance properties, and state its per-step cost.

## Implementation and debug (1)

D1. The code below trains q on the toy but the ELBO never exceeds
-0.62 while the true log marginal is -0.5108. The family is q =
Bernoulli(t). Diagnose the ceiling and fix it.

    # q parameterized as Bernoulli(t), optimized over t in [0,1]

## Changed-constraint scenarios (2)

T1. The latent becomes a 50-dimensional correlated Gaussian and the
family stays mean-field. Predict the failure mode, name the
irreducible cost, and propose the cheapest family upgrade.
T2. New data points arrive one per second and each needs a posterior
before the next arrives. Per-point VI takes 2 seconds per point.
Redesign the inference pipeline and state what you sacrifice.

## Research critique (1)

R1. "Our VAE reports a higher ELBO than the baseline, so it is the
better generative model." Give two reasons this comparison can
mislead, and design the experiment that would settle which model is
better.
