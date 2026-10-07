# Answer keys: U03

Unit: math-genmodels-U03. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: z = (1x1 + 2x2)/5 = 1.0. Reconstruction (1, 2). Squared
error 0. The clean point lies exactly on the line. Red flag:
reporting the code as the reconstruction. Rubric: the code, the
reconstruction, the zero.

## B2

Strong: representation: codes are features for downstream tasks.
Generative: prior over z plus decoder as a sampler makes new data.
The extra step is the prior: without it, decoding a random z is
undefined. Red flag: calling the deterministic autoencoder
generative. Rubric: both jobs plus the missing step named.

## B3

Strong: ELBO = E_q[log p(x|z)] - KL(q(z|x)||p(z)). Reconstruction
-2.1516 nats, KL 0.9013 nats, ELBO -3.0529 nats. Red flag: adding
the KL instead of subtracting. Rubric: all three numbers with
signs.

## B4

Strong: KL = -0.5 x (1 + log sigma^2 - mu^2 - sigma^2). At mu =
0.9, sigma = 0.4: -0.5 x (1 - 1.8326 - 0.81 - 0.16) = 0.9013.
Red flag: log of sigma instead of sigma squared. Rubric: the
formula and the value.

## B5

Strong: posterior collapse is q(z_d|x) = p(z_d) on some latent
dimensions: the encoder transmits nothing there. Signature:
per-dimension KL near zero across a validation batch while other
dims stay positive. Red flag: confusing it with a small total KL.
Rubric: the definition plus the per-dim diagnostic.

## B6

Strong: BPD = -log2 p(x) / D, bits per data dimension. From the toy
ELBO: (3.0529 / log 2) / 2 = 2.2022 bits/dim, an upper bound on the
true BPD. Red flag: quoting it without saying bound or exact.
Rubric: the formula, the number, the bound qualifier.

## L1 ladder

1. ELBO = E_q[log p(x|z)] - KL(q(z|x)||p(z)).
2. The encoder outputs q(z|x). The decoder defines p(x|z). The
   prior p(z) is fixed.
3. -2.1516 - 0.9013 = -3.0529 nats.
4. The KL prices latent information and pulls codes toward the
   prior so the prior stays a valid sampler.
5. Beta = 4 gives -5.7567. The encoder retreats toward the prior,
   rate falls, distortion rises, and the objective is no longer a
   likelihood bound.
Red flags: treating beta as a learning rate. Rubric: each rung with
the numbers.

## L2 ladder

1. Codebook posterior: softmax of -d^2/2 over the four codes =
   (0.1543, 0.1543, 0.0000, 0.6914).
2. Distances squared (5, 5, 45, 2) become logits (-2.5, -2.5,
   -22.5, -1). Softmax gives the posterior.
3. Argmin is piecewise constant: gradient zero almost everywhere,
   undefined at ties. No learning signal passes.
4. Straight-through copies the upstream gradient through the hard
   step as if it were the identity. Bias on the toy: 1.0 - 0.3989
   = 0.6011.
5. Experiment: set mu = 2.0. True gradient phi(2) = 0.0540.
   Straight-through still reports 1.0. Measured bias 0.9460
   confirms the bias grows under saturation.
Red flags: calling straight-through unbiased. Rubric: the posterior
numbers and the two bias measurements.

## A1

Strong: KL = integral q log(q/p). Insert the Gaussian densities,
take logs, use E_q[z] = mu and E_q[z^2] = mu^2 + sigma^2. The
integral reduces to -0.5 x (1 + log sigma^2 - mu^2 - sigma^2).
On the toy: -0.5 x (-1.8026) = 0.9013. Red flag: dropping the
-1/2. Rubric: the moment substitution and the verified number.

## A2

Strong: beta-objective = recon - beta x KL(q||p(z)). Write it as
ELBO_1 - (beta-1) x KL(q||p(z)). For beta >= 1 the subtracted term
is nonnegative, so beta-objective <= ELBO_1 <= log p(x): still a
bound, looser. For beta < 1 take q = posterior: beta-objective =
log p(x) + (1-beta) x KL(post||prior) > log p(x) when the KL is
positive. Not a bound. What it is instead: a Lagrangian for the
rate-distortion tradeoff, minimizing distortion subject to a rate
budget. Red flag: claiming beta > 1 breaks the bound. Rubric: the
rewriting, the counterexample for beta < 1, the Lagrangian reading.

## D1

Strong: mistake 1: Monte Carlo estimation of a closed-form
quantity. The Gaussian KL has an exact O(d) formula. Sampling it
injects pure variance into every gradient. Mistake 2: with one
sample the variance is maximal and swamps the reconstruction
signal, so the encoder retreats to the prior within tens of steps.
Fix: replace the sampler with the analytic KL. Independent check:
the loss curve smooths immediately and the per-dim KLs match the
formula within 1e-9. Red flag: "fixing" with more samples. Rubric:
both mistakes named with the fix and the check.

## T1

Strong: likelihood becomes Bernoulli per pixel: p(x|z) = product of
784 Bernoullis. Loss becomes binary cross-entropy summed over
pixels. Metric becomes bits per pixel over 784 dims. The Gaussian
KL term is unchanged: it depends only on q(z|x) and p(z), never on
the likelihood. Red flag: changing the prior because the data
changed. Rubric: the three rewrites plus the KL invariance.

## T2

Strong: 1 KB holds 256 float32 params. Codes in R^8: 8K <= 256, so
K <= 32. Posterior cost per point: 32 x 8 = 256 distance ops
versus 8 before. Tradeoff: finer quantization cuts distortion but
costs 32x the distance compute per point and 32 codes to learn
from the same data. Red flag: ignoring the learning cost of more
codes. Rubric: the K bound, the new cost, the tradeoff.

## R1

Strong: argument 1: the two numbers optimize different things. The
deterministic AE pays no KL price, so its MSE can always beat a
VAE that spends nats on a usable prior. MSE alone cannot rank a
representation against a generative model. Argument 2: MSE is not
a likelihood. The AE's 0.001 is not a bound on anything. The VAE's
ELBO is. Fair experiment: fix the decoder architecture and the
likelihood family, then report held-out ELBO or BPD for both,
fitting a prior over the AE codes so both are scored as generative
models, plus a sample-quality check. Red flag: proposing a bigger
AE as the fair baseline. Rubric: two valid arguments plus a matched
experiment.
