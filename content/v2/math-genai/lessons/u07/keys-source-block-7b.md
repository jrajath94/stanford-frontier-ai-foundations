# Answer keys, source block 07b

Date: 2026-10-06. Ground truth: compute_run5a.py.

## E01

A DDPM destroys data with a fixed gradual noise
chain and learns to reverse it. Toy: alpha_bar_50 =
0.77718008, x_50 = 1.99917538 from x_0 = 2.0.

## E02

A formulation turns the idea into trainable algebra.
The three results: the closed form q(x_t | x_0),
the posterior q(x_{t-1} | x_t, x_0) with its
coefficients, and the reverse parameterization via
eps_hat.

## E03

The U-Net predicts the noise in x_t. Inputs: the
noisy x_t and the timestep t. Output: eps_hat.

## E04

The chain ELBO splits into L_T, T-1 middle KLs,
and L_0. Toy L_T = 0.7713 nats = 1.1127 bits.

## E05

Minimize ||eps - eps_hat||^2 over uniform t.
Toy mean: 0.0305. Title-level reading of "ELBO
Equivalence": the chain ELBO coincides with the
standard VAE ELBO on the latents x_{1:T} (marked
as title-level inference, not inspected fact).

## E06

From x_T ~ N(0, 1), iterate the reverse Gaussian T
times. Toy: 1.75690526, 1.74406522, 1.82846911.
Cost: T network evaluations per sample.

## L01

DDPM: a fixed forward noise chain plus a learned
Gaussian reverse chain. Toy: x_50 = 1.99917538.
Derivation: unroll two steps, merge the Gaussian
noise terms, induct. Implement: q_sample as in the
lesson. Compare: the titles promise the idea, the
formulation, and the implementation. No transcript
was inspected, so the lecture's schedule, network,
and derivation order are unknown. Debug: exploding
variance means the sqrt(alpha_t) scaling is
missing. Critique: a learned schedule adds
parameters for a choice the SNR curve already
guides. Fixed schedules are the reported standard.
Design: an inspected transcript or slide deck
showing the formulation derivations would promote
the row.
