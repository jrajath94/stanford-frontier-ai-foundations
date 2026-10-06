# Answer keys, lab 04 (VAE mechanics)

Date: 2026-10-06. Ground truth: compute_run5a.py.

## Task 1

(a) See lesson C02: the five lines. Jensen is used
at log E_q[...] >= E_q[log ...].
(b) elbo_1sample returns (-0.9108, -0.7342, 0.1766)
nats.
(c) q collapses toward a point mass at the best z
per x. The KL price vanishes and the model is a
plain autoencoder with a fake posterior.
(d) Prediction: the -0.5 log sigma^2 term explodes
and the KL goes to +infinity. Measured: KL =
0.5(0.16 + 1e-9 - 1 + 20.723) = 9.94 nats. The
prediction holds. Tiny sigma^2 is fatal.

## Task 2

(a) 2 mu = 0.8.
(b) Reparameterized mean 0.6244, SE 0.1345. Score
mean 0.5671, SE 0.2622. Both within 3 SE of 0.8.
(c) SE scales as 1/sqrt(n): at n = 4096 each SE
shrinks 8x (0.0168 and 0.0328). The ratio stays ~2.
(d) "It works for any distribution but pays double
the variance here. Use it only where
reparameterization is impossible, e.g. discrete
latents."

## Task 3

(a) Expand E_q[-0.5 log(2 pi sigma^2) - (z-mu)^2 /
(2 sigma^2)] - E_q[-0.5 log(2 pi) - z^2/2] and use
E_q[z^2] = mu^2 + sigma^2.
(b) Analytical 0.1766 nats. MC 0.1760 nats.
Agreement within 0.005.
(c) Finite difference: 0.4000000000. Analytical:
mu = 0.4. Match.
(d) The closed form changes: KL = 0.5(mu^2/s2p +
sigma^2/s2p - 1 - log(sigma^2/s2p)) with prior
N(1, 4) needs the shifted form 0.5(((mu-1)^2 +
sigma^2)/4 - 1 - log(sigma^2/4)) + log 2. At the
toy: 0.5((0.36 + 0.5)/4 - 1 - log(0.125)) + 0.6931
= 0.5(0.215 - 1 + 2.0794) + 0.6931 = 1.3404 nats.
All lesson numbers quoting 0.1766 change.

## Task 4

(a) Recon = -0.4516 nats (xhat = x), KL = 0, ELBO =
-0.4516 nats.
(b) Recon -0.4516 nats (-0.6515 bits), KL 0, ELBO
-0.4516 nats. Beats healthy -0.9108 by 0.4592 nats.
(c) The KL tax (0.1766 nats) exceeds the
reconstruction gain from using z, so the rational
optimum drops the code.
(d) Mean KL per dim over the 4 mus: computed in the
audit run. With mus (0.48, 0.15, -0.12, 0.54) and
logvar -0.7 each KL is 0.5(mu^2 + 0.4966 - 1 +
0.7) = 0.5 mu^2 + 0.0983: values 0.2135, 0.1096,
0.1055, 0.2441. Mean 0.1682 nats. All 4 dims
active at threshold 0.01.

## Task 5

(a) Beta = 2: -0.7342 - 0.3531 = -1.0874 nats. Beta
= 8: -0.7342 - 1.4126 = -2.1468 nats.
(b) 0.5: -0.8225. 1: -0.9108. 2: -1.0874. 4:
-1.4405. 8: -2.1468 nats.
(c) -0.4516 = -0.7342 - beta* 0.1766 gives beta*
= (0.7342 - 0.4516)/0.1766 = 1.6007. Above beta*
1.60 the collapsed model wins on this toy.
(d) Acceptable argument: at beta = 4 the healthy
ELBO (-1.4405) already loses to collapse (-0.4516)
by a full nat, and the boundary is beta* = 1.60. Beta = 4 buys no disentanglement on a 1-D toy and
pays certain collapse. Argue for beta <= 1.5 with
a traversal check, or for annealing instead of a
fixed beta.
