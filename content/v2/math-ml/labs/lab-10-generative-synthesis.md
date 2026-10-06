# Lab 10, generative bridge and synthesis

Unit: math-ml-U10. Date: 2026-10-06. Keys in labs/keys-lab-10.md.
All numbers computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## Task 1, ELBO gap on a new discrete toy

z in {0,1}, p(z) = [0.5, 0.5], p(x=1|z=0) =
0.1, p(x=1|z=1) = 0.8. Observe x = 1.
Approximate posterior q = [0.4, 0.6].

(a) Compute p(x=1), log p(x=1), the ELBO, and
the gap. Reference: p = 0.45, log p =
-0.7985076962, ELBO = -1.0750556815, gap =
0.2765479853.

(b) Compute the true posterior p(z|x=1) and
verify the gap equals KL(q||posterior) to
6 digits.

(c) Assert ELBO <= log p(x) in code.

## Task 2, predicted versus measured, rerun

Rerun the C09 protocol with seed 12 instead of
11 (4000 trials, B = 4 vs 16, same 64 points).

(a) Report the measured ratio Var(B=4) /
Var(B=16). Reference: 3.9594 (seed 12).

(b) The predicted ratio is 4.0. Is the seed-12
measurement closer to or farther from 4.0 than
the seed-11 value 4.2887? What does that say
about E21's warning?

## Task 3, the beta sweep and the collapse

Reuse the C02 numbers: one-sample recon =
-0.9302, KL(informative) = 0.3981,
KL(collapsed, q = N(0,1)) = 0.0, recon
collapsed (MC, seed 7, 20000 trials) =
-1.6695.

(a) Compute the beta-VAE total (recon +
beta KL) for the informative q at beta in
{0, 1, 10} and for the collapsed q at beta =
10. Reference: -0.9302, -1.3283, -5.0713.
collapsed at 10: -1.6695.

(b) At beta = 10, which q does the optimizer
prefer, and what does that say about the
latent? Two sentences.
