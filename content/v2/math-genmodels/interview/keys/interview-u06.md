# Interview keys: U06

Unit: math-genmodels-U06. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice. Strong answers, red flags, rubrics,
remediation.

## Q1

Strong: the score is grad_x log p. The normalizer is additive
in log space, so the gradient kills it. This buys training
without partition functions: energy models become tractable to
fit. Red flag: "the score is the density." Rubric: the
log-gradient mechanism plus the payoff. Remediation: U06-C01.

## Q2

Strong: expand E[0.5||s_theta - s||^2]. cross term E[s_theta .
s] = -E[div s_theta] by parts. drop constants. The trace is
div s_theta, the divergence of the model field. Red flag:
skipping the parts step. Rubric: three lines with the trace
named. Remediation: U06-C02.

## Q3

Strong: E[x_0 | x] = x + sigma^2 grad_x log p(x). It is a
posterior mean by construction (Bayes + Gaussian noise). A mean
is not a sample: under a bimodal posterior it sits between
modes. Red flag: using Tweedie as a sampler. Rubric: the
formula plus the mean-versus-sample split. Remediation:
U06-C03.

## Q4

Strong: q(x_t | x_0) is Gaussian in closed form, so any t is
O(1) to sample. Simulating the chain would cost O(t) per
sample and teach nothing extra. Red flag: "the chain must be
simulated." Rubric: the closed form and the cost argument.
Remediation: U06-C04.

## Q5

Strong: eta in [0,1] sets the noise per reverse step: 0 is the
deterministic ODE, 1 is DDPM. Striding keeps marginals only
approximately. too aggressive and colors shift, details wash
out. Red flag: treating striding as free. Rubric: the eta
meaning plus the striding cost. Remediation: U06-C07.

## Q6

Strong: each step evaluates both the conditional and the
unconditional score: 2x calls. w = 3 is too large when samples
show overshoot artifacts (saturation, distortion): fidelity up,
diversity and realism down. Red flag: "larger w is always
better." Rubric: the 2x cost plus the overshoot symptom.
Remediation: U06-C10.

## L1 ladder

1. s(x) = grad_x log p(x). N(0,1) gives -x, s(1) = -1.
2. x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps.
   toy x_4 = 1.5174.
3. Mean 0.6776 x_0 + 0.3194 x_2 = 1.8983, variance 0.0714.
4. Same step: DDIM 1.7630 deterministic. DDPM mean 1.8983
   plus noise. Different targets, different character.
5. Striding broke the marginal match. Fix: add steps back
   until quality recovers, or switch to a higher-order
   solver. Strong answers measure quality versus steps.
   Red flag: retraining the model.

## L2 ladder

1. dx = -0.5 beta(t) x dt + sqrt(beta(t)) dw, the VP-SDE.
2. dx = [f - 0.5 g^2 score] dt: same marginals, no noise.
3. The score must exist and be smooth at all t. Manifold data
   has no score at t = 0. the model learns a smoothed fiction.
4. Smoothed uniform sigma = 0.05: score 13.5 at x = 0.01,
   0.0 inside.
5. The likelihood integrates the smoothed law. the noise
   floor is a free parameter that moves the number. Name it
   or the comparison is void.

## A1

Strong: p(x_0 | x) proportional to N(x. x_0, sigma^2) p(x_0).
Write the posterior mean as an integral. differentiate log
p(x) = log integral N(x. x_0, sigma^2) p(x_0) dx_0 under the
integral. the derivative brings down (x_0 - x)/sigma^2.
rearrange to E[x_0 | x] - x over sigma^2. Red flag:
differentiating under the integral without noting the
Gaussian form. Rubric: the full chain with the rearrangement.

## A2

Strong: base: at t = T both are N(0, I) by construction.
Inductive step: assume x_t marginal matches. the DDIM eta =
0 update is x_{t-1} = sqrt(alpha_bar_{t-1}) x0_hat +
sqrt(1 - alpha_bar_{t-1}) eps_pred. With the true eps, x0_hat
= x_0 exactly and the update reconstructs the forward
marginal at t-1. Red flag: hand-waving the induction.
Rubric: base plus step with the exact-eps argument.

## D1

Strong: (1) sampler/schedule mismatch: sampling betas differ
from training, or strided without DDIM: test Tweedie
single-step sharpness. sharp there but mushy after sampling
proves the sampler. (2) Time embedding present but broken
(e.g. not added to the right blocks): test loss per t. flat
across t is the tell. (3) The loss 0.005 is on the training
noise distribution but sampling starts from pure noise the
model never saw at t = T: test the t = T marginal. Order:
sampler first (loss proves the model), then embedding, then
marginal. Red flag: adding capacity. Rubric: three causes in
order with one test each.

## T1

Strong: DDIM with 4 strided steps, or a distilled 4-step
student, or a consistency model. Cost: marginal mismatch.
measure a distributional distance to the 1000-step sampler at
matched seeds plus human sharpness ratings. Report quality
versus steps. Red flag: "4 steps are enough." Rubric: the
redesign plus the measured cost.

## T2

Strong: discrete diffusion with masking or uniform resampling
over the vocabulary. the "score" becomes probability ratios.
Breaks: C01 (no gradient on discrete sets), C03 Tweedie,
C09 SDE, C11 ODE solvers. Survives: chains, the loss shape,
striding. Red flag: Gaussian noise on token IDs. Rubric: the
new process plus the split.

## R1

Strong: attack 1: the ODE likelihood needs a noise floor on
discrete/bounded data. the ranking can flip with it (C12).
Attack 2: likelihood is one projection. "best generative
model" needs samples, coverage, and cost (U08). Fair
experiment: fixed dequantization, likelihood with seed
uncertainty, paired with precision/recall and sampling cost
at matched budgets. Red flag: the likelihood as verdict.
Rubric: both attacks plus the paired design.
