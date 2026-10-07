# Interview keys: U03

Unit: math-genmodels-U03. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers, red flags, and rubrics. Original practice material.

## Q1

Strong: near-zero error with a wide bottleneck means the net
learned the identity. The codes are the data in disguise: no
compression, no structure, no usable prior. Red flag: "low error
means good codes". Rubric: the identity-map diagnosis.

## Q2

Strong: a stochastic encoder q(z|x), a fixed prior p(z), and the
ELBO objective: reconstruction minus KL(q||p). The KL makes the
prior a valid sampler. Red flag: naming only the KL. Rubric: all
three parts with the sampler payoff.

## Q3

Strong: KL = -0.5 x (1 + log sigma^2 - mu^2 - sigma^2). Zero
exactly at mu = 0, sigma = 1, the prior itself. Red flag: log
sigma instead of log sigma^2. Rubric: the formula and the zero
condition.

## Q4

Strong: collapse is q(z_d|x) = p(z_d) on some dims: they carry no
information. Detect with per-dim KL near zero across a validation
batch. Red flag: diagnosing from the total KL alone. Rubric: the
definition plus the per-dim check.

## Q5

Strong: beta prices latent information: high beta buys
disentangling and blur, low beta buys fidelity and a weak prior.
Tuning on the beta objective is circular: it falls as beta rises
by construction. Tune on held-out likelihood or the downstream
task. Red flag: calling beta a learning rate. Rubric: the tradeoff
and the tuning rule.

## Q6

Strong: not exact. It is an upper bound from the ELBO: the true BPD
is lower by the variational gap. Exact only with the true log
marginal, e.g. from a flow model or an exact sum. Red flag:
reporting it unlabeled. Rubric: bound named with the exactness
condition.

## L1

1. ELBO = E_q[log p(x|z)] - KL(q(z|x)||p(z)). Encoder: q. Decoder:
   p(x|z). Prior: p(z).
2. Recon -2.1516, KL 0.9013, ELBO -3.0529 nats.
3. Insert Gaussian densities in the KL integral, use E[z] = mu and
   E[z^2] = mu^2 + sigma^2, simplify to the closed form.
4. Numpy: eq_sq formula from the lesson. Assert abs(kl - 0.9013) <
   1e-9 against the closed form.
5. VAE: one encoder/decoder pass plus KL, exact sampling from the
   prior, honest bound on likelihood. AE plus post-hoc prior:
   cheaper training, sampling needs a separately fitted prior, no
   bound. The VAE's number is honest. The AE's MSE is not a
   likelihood.
Red flags: mapping the decoder to q. Rubric: all five rungs.

## L2

1. Posterior over codes: softmax of -d^2/2.
2. (0.1543, 0.1543, 0.0000, 0.6914).
3. Argmin is piecewise constant: zero gradient almost everywhere,
   undefined at ties.
4. Straight-through copies the upstream gradient through the hard
   step. Bias: 1.0 - 0.3989 = 0.6011.
5. Straight-through: biased, zero extra variance, no knob.
   Gumbel-Softmax: biased with a temperature knob, lower variance
   than score-function, bias vanishes as temperature falls but
   variance rises. Switch when the straight-through bias visibly
   hurts: gradient magnitudes that ignore saturation.
Red flags: calling straight-through unbiased. Rubric: the bias
number and the switch condition.

## A1

Strong: beta-obj = ELBO_1 - (beta-1) KL(q||p(z)). For beta >= 1 the
correction is nonpositive, so beta-obj <= ELBO_1 <= log p(x): a
looser bound. For beta < 1, q = posterior gives beta-obj = log p(x)
+ (1-beta) KL(post||prior) > log p(x). Not a bound. Red flag:
claiming beta > 1 breaks the bound. Rubric: the rewriting plus the
counterexample.

## A2

Strong: diagonal q factorizes, so KL = sum_d KL_d with each term
the scalar closed form. The decomposition is the diagnostic
because collapse is per-dimension: total KL can look healthy while
half the dims sit at zero. Red flag: averaging instead of summing.
Rubric: the sum and the diagnostic argument.

## D1

Strong: the real problem is posterior collapse on dim 2: the
decoder reconstructs without it, so the KL gradient killed it.
Deleting the KL term destroys the prior and the sampler. Cheaper
fix first: KL annealing, start the KL weight near 0 and raise it
slowly so the dim gets used before it is priced. Check: per-dim KL
rises off zero on validation. Red flag: accepting the deletion.
Rubric: the collapse diagnosis with the cheaper fix.

## T1

Strong: 8-bit hardware wants a discrete likelihood: quantized
Gaussian or categorical over 256 levels per pixel, cross-entropy
loss. The bound stays a bound: the ELBO identity never assumed
floats. Exactness is unchanged in principle. In practice the
coarser likelihood changes the numbers. Red flag: keeping the
Gaussian and rounding after. Rubric: the likelihood redesign with
the bound status.

## T2

Strong: knob 1: beta down. Less KL pressure keeps more latent
detail, sharper samples, worse rate. Knob 2: likelihood choice,
e.g. a sharper or multimodal decoder noise model instead of fixed
Gaussian. Watch sample-based metrics (FID-style scores or human
eval), not BPD: BPD rewards covering mass, which is the blur
direction. Red flag: raising beta for sharpness. Rubric: both
knobs with directions and the metric switch.

## R1

Strong: reason 1: BPD from a VAE is a bound. The discrete model's
number may be exact or a tighter bound, so the comparison mixes
bound looseness with model quality. Reason 2: different latent
types change the effective capacity and the optimization
difficulty. The win may be optimization, not representation.
Settling experiment: matched decoder capacity and matched
inference budgets, report bound-vs-bound BPD with the gap
measured, plus a downstream probe on the codes and a sample
quality check. Red flag: declaring the latent type the winner from
one number. Rubric: two valid reasons plus a matched experiment.
