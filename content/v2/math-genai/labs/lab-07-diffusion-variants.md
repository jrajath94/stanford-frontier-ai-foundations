# Lab 07, diffusion-variant mechanics on the scalar toy

Unit: math-genai-U08. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-07.md. Test-mode: solve closed-book, then
check. The toy: T = 100, linear betas 1e-4 to 0.02,
x_0 = 2.0, eps = 0.5, eps_hat = 0.4. Ground truth:
compute_run5b.py.

## Task 1, the DDIM jump: predict first, measure second

(a) Predict: sigma for the 100 -> 90 jump at eta = 0
and eta = 1. Write both before computing.
(b) Measure: implement ddim_sigma and ddim_step.
Record x_100, x0_hat, x_90 at eta = 0.
(c) Verify the marginal identity: sigma^2 + (1 -
ab_90 - sigma^2) against 1 - ab_90 at eta = 1.
(d) Break it: set eta = 1.5. Record what fails and
which battery assert catches it.

## Task 2, determinism vs stochasticity

(a) Fix x_T from draw 0.7. Run the 10-step
trajectory at eta = 0 with seeds 0 and 1. Record
the max |diff|.
(b) Repeat at eta = 1. Record the max |diff|.
(c) State in one sentence what "deterministic"
means here and what it does not mean.
(d) Redraw x_T with a new seed at eta = 0. Does
the sample change? Explain.

## Task 3, the timestep embedding

(a) By hand: compute emb(50) entries 0 and 1 for d
= 8. Verify -0.262375 and 0.964966.
(b) In code: implement temb. Verify emb(0) =
[0, 1, 0, 1, 0, 1, 0, 1].
(c) Compute cos(emb(49), emb(50)) and cos(emb(10),
emb(90)). Record both.
(d) Zero the embedding. Explain in one sentence
what the net loses.

## Task 4, guidance

(a) Predict: x0_hat for g = 0, 1, 2, 3 with
eps_u = 0.35, eps_c = 0.45. Write all four before
computing.
(b) Measure: implement cfg and the x0_hat
conversion. Record the four values.
(c) Run the hit-rate sweep from C10 (500 draws,
seed 2, noise 0.15). Record the four hit-rates.
(d) Explain the inverted-U in two sentences:
what the linear blend does, and why the peak is
at g = 1 here.

## Task 5, the battery and the serving budget

(a) Implement the six-assert battery from memory.
Run it on your ddim_step. Report green or the
first failure.
(b) Introduce the E-018 bug (ab[t] instead of
ab[t-1] in x0_hat). Report which assert fails
first.
(c) Compute the serving FLOPs for three configs:
DDIM-10 eta=0 g=1, DDIM-10 eta=0 g=2 (CFG),
DDPM-100. Record all three.
(d) Write one paragraph: which config ships
under a hard latency budget, and what quality
evidence you still need before deciding.
