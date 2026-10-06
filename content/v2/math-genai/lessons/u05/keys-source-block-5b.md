# Answer keys, source block 05b

Date: 2026-10-06. Ground truth: compute_run5a.py.

## E01

A latent variable model explains visible x through
hidden z. Joint: p(x, z) = p(x | z) p(z). Marginal on
the toy: p(x) = int N(x | W z, 0.25 I) N(z | 0, 1) dz.

## E02

log p(x) >= E_q[log p(x | z)] - KL(q || p). Jensen is
used at the log-E to E-log step. Toy gap: -1.9341 -
(-3.2382) = 1.3040 nats.

## E03

gamma_2 = 0.8802 (E14 of the lesson keys). Guarantee:
the log-likelihood never decreases across an EM step.

## E04

A VAE amortizes variational inference into a neural
encoder and trains the triple on the ELBO with
reparameterized sampling. Parts: prior p(z), encoder
q(z | x), decoder p(x | z). Objective: the ELBO.

## E05

Jensen is proved where it is used: the ELBO derivation
in W5L18 needs it. E-009: title order is a playlist
fact, not a course claim. The pairing is pedagogical,
not chronological.

## E06

Write the sample as a differentiable function of the
parameters plus fixed noise. Toy: reparameterized SE
0.1345 vs score SE 0.2622 at 64 samples. The
reparameterized estimator is about twice as precise.

## L01

ELBO: E_q[log p(x|z)] - KL(q||p), a lower bound on log
p(x). Toy: expected -3.2382 nats, marginal -1.9341,
gap 1.3040 nats. Derivation: the five-line proof
(C02). Implement: expected_elbo as in the lesson.
Compare: the titles promise the ELBO and its proof. No transcript was inspected, so the lecture's
derivation path and toy are unknown. Debug: a
1-sample ELBO can exceed the marginal. The bound is
in expectation. Average over many samples. Critique:
yes, "lower bound" overpromises for the 1-sample
estimate. It is a bound only in expectation. Design:
an inspected transcript or slide deck showing the
ELBO derivation would promote the row.
