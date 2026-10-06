# Answer keys, lab 06 (DDPM mechanics)

Date: 2026-10-06. Ground truth: compute_run5a.py.

## Task 1

(a) beta_50 = 0.0001 + 49 * 0.0199/99 = 0.00994949.
alpha_bar_50 = product of (1 - beta_t) for t =
1..50 = 0.77718008.
(b) beta: 0.0001, 0.00994949, 0.02. alpha_bar:
0.9999, 0.77718008, 0.36356325. SNR: 9999.0,
3.4879, 0.5712.
(c) Bisection: SNR crosses 1 at t = 83 (SNR
0.9939 there).
(d) The chain never reaches pure noise, so
generation starts its reverse loop
out-of-distribution relative to the training distribution.

## Task 2

(a) sqrt(0.77718008) * 2.0 = 1.7631506. Sqrt(0.22281992) * 0.5 = 0.2360178. Sum
1.9991684 (audit 1.99917538).
(b) Chain simulation (seed 0): mean and std of
2000 endpoints match q_sample's N(1.99917538
mean over draws, sqrt(0.22281992) = 0.47204
std) within sampling noise. Both near mean
1.9992, std 0.4720.
(c) q_sample is O(1) per example. The chain is
O(T).
(d) With 1-indexed t, ab[t] reads alpha_bar_{t+1}:
x_50 uses alpha_bar_51 = 0.76929131 instead of
0.77718008. x_50 moves from 1.99917538 to
1.99434579. Every noise level shifts one step
noisier.

## Task 3

(a) beta_tilde_50 = 0.00960075. Coefficients:
0.03956209 on x_0, 0.96013574 on x_50.
(b) posterior_coeffs returns (0.00960075,
0.03956209, 0.96013574). Sum 0.99969783 ~ 1.
(c) x0_hat = 2.05354466, mu_rev = 2.00072225.
With eps_hat = 0.5: x0_hat = 2.0 exactly,
mu_rev = 1.99860391 (the true posterior mean).
(d) Upper (beta_t): 2.03064640. Lower
(beta_tilde_t): 2.03011727. Zero: 2.00072225.

## Task 4

(a) q(x_100 | x_0) = N(sqrt(0.36356325)*2,
1 - 0.36356325) = N(1.20596, 0.63644). KL =
0.5(1.45434 + 0.63644 - 1 - ln 0.63644) = 0.7713
nats.
(b) L_T_term gives 0.7713 nats. L_mid gives
0.0002337 nats.
(c) L_49 = 0 and L_0 = 0: the reverse matches
the posterior exactly when the noise prediction
is perfect.
(d) Sum over t = 2..100: 0.06292043 nats
(0.09077500 bits). Against L_T = 0.7713 nats:
the prior term dominates the toy ELBO by 12x.

## Task 5

(a) 0.01, 0.09, 0.0025, 0.04, 0.01. Mean 0.0305.
(b) The training step reproduces the five losses
with t = int(rng.integers(1, T+1)) and ab[t-1].
With the T1 bug (t in 0..T-1 passed to a
1-indexed net), every step conditions the network
on the wrong timestep label: the network applies
the step-t denoising rule to noise level t+1.
(c) The loop reproduces 1.75690526, 1.74406522,
1.82846911.
(d) With sigma = 0: x_99 = mu_rev(100),
x_98 = mu_rev(99), x_97 = mu_rev(98). The
trajectory is deterministic given the network.
The noise contributes the sampling diversity:
without it, one x_T gives one output.
