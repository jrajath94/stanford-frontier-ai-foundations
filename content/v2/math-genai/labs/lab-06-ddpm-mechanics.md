# Lab 06, DDPM mechanics on the scalar toy

Unit: math-genai-U07. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-06.md. Test-mode: solve closed-book, then
check. The toy: T = 100, linear betas 1e-4 to 0.02,
x_0 = 2.0, eps = 0.5, eps_hat = 0.4. Ground truth:
compute_run5a.py.

## Task 1, the schedule: predict first, measure second

(a) Predict: beta_50 and alpha_bar_50 on the linear
schedule. Write both before computing.
(b) Measure: build the schedule in code. Record
beta_1, beta_50, beta_100 and alpha_bar_1,
alpha_bar_50, alpha_bar_100, SNR at 1, 50, 100.
(c) Find the t where SNR crosses 1 by bisection.
Record it.
(d) Write one sentence: what does the endpoint SNR
0.5712 imply for generation?

## Task 2, direct noising vs chain simulation

(a) By hand: compute x_50 from x_0 = 2.0, eps = 0.5
using the closed form. Show both terms.
(b) In code: implement q_sample. Then simulate the
chain step by step for 50 steps (seed 0, fresh eps
per step). Compare the distribution of 2000 chain
endpoints against 2000 q_sample draws: record both
means and stds.
(c) State in one sentence why training uses
q_sample.
(d) Break it: index ab[t] instead of ab[t-1] for
1-indexed t. Record how far x_50 moves.

## Task 3, the posterior: predict first, measure second

(a) Predict: beta_tilde_50 and the two posterior
coefficients. Write all three before computing.
(b) Measure: implement posterior_coeffs. Record the
triple and verify c0 + ct against 1.
(c) Implement reverse_mean with eps_hat = 0.4.
Record x0_hat and mu_rev. Then set eps_hat = eps =
0.5 and record how mu_rev changes.
(d) Sample x_49 with z = 0.3 under all three
variance choices. Record the three values.

## Task 4, ELBO terms

(a) By hand: compute L_T from q(x_100 | x_0). Show
the mean, variance, and the KL arithmetic.
(b) In code: implement L_T_term and L_mid.
Reproduce 0.7713 nats and 0.0002337 nats.
(c) Set eps_hat = eps exactly. Record what happens
to L_49 and L_0. Explain in one sentence.
(d) Compute the full middle sum: sum of L_{t-1}
over t = 2..100 with the constant eps_hat = 0.4
network and z-free means. Record the total and
compare it against L_T.

## Task 5, the training step and the loop

(a) By hand: compute the five toy pair losses and
their mean.
(b) In code: implement the training step with
correct 1-indexed t. Reproduce the five losses.
Then introduce the T1 bug (0-indexed t into net)
and state exactly what goes wrong.
(c) Implement the three-step sampling loop from
x_100 = 1.7. Reproduce 1.75690526, 1.74406522,
1.82846911.
(d) Rerun the loop with sigma = 0 at every step.
Record the three values. State in one sentence
what the noise contributes.
