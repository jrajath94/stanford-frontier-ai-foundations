# Lab 04, VAE mechanics on the Gaussian toy

Unit: math-genai-U05. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-04.md. Test-mode: solve closed-book, then
check. The toy: x = [1.2, 0.4] in R^2, z in R^1,
p(z) = N(0, 1), encoder mu = 0.4, sigma^2 = 0.5,
decoder p(x | z) = N(W z, 0.25 I), W = [[1.0], [0.5]].
Ground truth: compute_run5a.py.

## Task 1, the ELBO by hand then in code

(a) By hand: write the five-line ELBO derivation from
log p(x). Name the line where Jensen is used.
(b) In code: implement elbo_1sample as in lesson C02.
Reproduce recon -0.7342 nats, KL 0.1766 nats, ELBO
-0.9108 nats with eps_draw = 0.6.
(c) Break it: drop the KL term and maximize recon
alone. State what q does and why the model stops
being variational.
(d) Predict first: what happens to your ELBO number
if sigma^2 is set to 1e-9? Measure it in code and
judge the prediction.

## Task 2, reparameterization: predict first, measure second

(a) Predict: d/dmu E[z^2] at mu = 0.4, var = 0.25.
Write the number before computing.
(b) Measure: implement both estimators with n = 64,
seed 0. Record both means and both SEs. Judge the
predictions.
(c) Increase n to 4096 (same seed stream). Record how
each SE changes. State the scaling law.
(d) Write one sentence: what would you tell a
teammate who proposes the score-function estimator
"because it works for any distribution"?

## Task 3, the analytical KL and its check

(a) By hand: derive KL = 0.5 (mu^2 + sigma^2 - 1 -
log sigma^2) from the two Gaussian log-densities.
(b) In code: verify 0.1766 nats at the toy values,
then Monte Carlo the KL with 200000 samples, seed 0.
Record the agreement.
(c) Finite-difference d KL / d mu at the toy with
h = 1e-6. Record the match with the analytical
derivative mu = 0.4.
(d) Break it: set the prior to N(1, 4). State which
lesson numbers change and recompute the KL.

## Task 4, posterior collapse: predict first, measure second

(a) Predict: the 1-sample ELBO of the collapsed model
(q = prior, xhat = x). Write the number before
computing.
(b) Measure: implement it. Record recon, KL, ELBO in
nats and bits. Compare against the healthy -0.9108
nats.
(c) State in one sentence why the objective prefers
the collapsed model.
(d) Implement the collapse diagnostic from C07 on a
batch of 4 encoder outputs (use the C11 mus). Record
the mean KL per dim and the active count.

## Task 5, beta sweep and the boundary

(a) By hand: compute the toy ELBO at beta = 2 and
beta = 8 from recon -0.7342 and KL 0.1766 nats.
(b) In code: sweep beta in {0.5, 1, 2, 4, 8}.
Record all five ELBOs.
(c) Solve the collapse boundary inequality for
beta*: the beta where the collapsed ELBO (-0.4516
nats) equals the healthy beta-ELBO. Give beta*.
(d) Write one paragraph: your team wants beta = 4
"for disentanglement". Use your numbers to argue
for or against on this toy.
