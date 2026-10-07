# Answer keys: U06

Unit: math-genmodels-U06. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: s(x) = grad_x log p(x), the uphill direction on
log-density. For N(0,1): s(x) = -x, so s(1) = -1. Red flag:
calling it the gradient of the density. Rubric: the definition
plus the number.

## B2

Strong: E_p[tr(grad s_theta) + 0.5 ||s_theta||^2]. For -a x:
-a + 0.5 a^2, min at a = 1, value -0.5. Red flag: minimizing
the Fisher divergence directly (it needs the true score).
Rubric: the objective and the closed-form optimum.

## B3

Strong: x0_hat = x + sigma^2 grad_x log p(x). Toy: 1.2 + 0.25
x (-4.8) = 0.0. Red flag: forgetting the sigma^2 factor.
Rubric: the formula and the number.

## B4

Strong: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t)
eps. Toy: sqrt(0.3024) x 2 + sqrt(0.6976) x 0.5 = 1.5174. Red
flag: simulating all intermediate steps. Rubric: the closed
form and the number.

## B5

Strong: L = E[||eps - eps_theta(x_t, t)||^2]. eps is the true
noise, eps_theta the network prediction, x_t the noisy sample,
t the time. Toy batch: 0.005. Red flag: calling it a
likelihood. Rubric: the formula with every symbol named.

## B6

Strong: eta = 0: predict x0_hat, then x_{t-1} = sqrt(alpha_bar)
x0_hat + sqrt(1 - alpha_bar) eps_pred. Toy: 1.70 to 1.7630,
deterministic. Red flag: adding fresh noise with eta = 0.
Rubric: the update and the number.

## L1 ladder

1. s(x) = grad_x log p(x). No normalizer needed.
2. Expand E[0.5||s_theta - s||^2]. The cross term becomes
   -E[div s_theta] by parts. The s-only term is constant.
3. Denoising: predict the added noise. Tweedie links the
   noise prediction to the noisy score: same optimum, no
   trace.
4. Toy: 1.2 -> 0.0 exactly for the point mass.
5. The trace came from the divergence of the model score. The
   denoising form never differentiates the model, so the trace
   never appears.

## L2 ladder

1. q(x_{t-1} | x_t, x_0) Gaussian with mean 0.6776 x_0 +
   0.3194 x_2 = 1.8983, variance 0.0714.
2. Same step: DDIM lands at 1.7630 exactly. DDPM lands near
   1.8983 plus noise. Deterministic versus stochastic.
3. Langevin: 1 + 0.05 x (-1) + sqrt(0.1) x 0.3 = 1.0449.
   Drift plus calibrated noise.
4. Higher-order solvers trade score calls per step for fewer
   steps: Heun halves the count at 2x per-step cost.
5. Striding breaks the marginal match: colors shift, details
   wash out. Validate quality versus step count.

## A1

Strong: E[0.5||s_theta||^2] - E[s_theta . s] + const. By parts:
E[s_theta . grad log p] = -E[div s_theta] + boundary term.
Boundary: p s_theta -> 0 at infinity. Result: E[div s_theta +
0.5||s_theta||^2] + const. Red flag: dropping the boundary
term without stating it. Rubric: the expansion, the parts
step, the stated assumption.

## A2

Strong: q(x_t|x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t),
q(x_{t-1}|x_0) = N(sqrt(alpha_bar_{t-1}) x_0, 1 -
alpha_bar_{t-1}). Product is Gaussian. Complete the square.
Mean coefficients: sqrt(alpha_bar_{t-1}) beta_t/(1 -
alpha_bar_t) on x_0 and sqrt(alpha_t)(1 - alpha_bar_{t-1})/(1
- alpha_bar_t) on x_t. Toy: 0.6776, 0.3194. Red flag:
memorizing the formula without the product. Rubric: the
product plus the evaluated coefficients.

## D1

Strong: bug: the network predicts the noise mean correctly but
the sampler uses the wrong variance schedule, or the time
embedding is present but the noise schedule at sampling
differs from training (e.g. strided steps without DDIM
correction). Most likely: the model learned E[eps | x_t]
fine (loss 0.005 proves it) but sampling adds too much noise
per step, washing out detail: the reverse variances are from
a different schedule than training. Fix: match the sampling
schedule to the training betas exactly, or switch to DDIM
with eta = 0. Test: single-step denoising quality (Tweedie
x0_hat on a grid) must be sharp. If sharp there but mushy
after full sampling, the sampler is the bug, not the model.
Red flag: retraining with more capacity. Rubric: the
sampler-versus-model split, the fix, the Tweedie test.

## T1

Strong: use DDIM with 4 strided steps (eta = 0), or distill to
a 4-step student, or a consistency model. Quality cost:
marginal mismatch grows as steps shrink: measure FID or a
toy distributional distance versus the 1000-step sampler at
matched seeds, plus a human sharpness check. Report the curve
of quality versus steps. 4 steps sits far down it. Red flag:
claiming 4 steps are free. Rubric: the sampler redesign plus
the measured cost.

## T2

Strong: discrete diffusion: replace Gaussian noise with a
masking or uniform-resampling process over the vocabulary.
the score becomes a ratio of probabilities, not a gradient.
Breaks: C01 (no gradient of log-density on discrete sets),
C03 Tweedie (needs Gaussian noise), C09 SDE (continuous
paths), C11 ODE solvers (no ODE). Survives: the chain idea,
the loss shape, DDIM-style striding. Red flag: adding
Gaussian noise to token IDs. Rubric: the new formulation plus
the broken/surviving split.

## R1

Strong: attack 1: the ODE likelihood measures the smoothed
law, and on discrete or bounded data the noise floor choice
moves the number (C12). The ranking can flip with
dequantization. Attack 2: likelihood is one projection.
"best generative model" needs sample quality, mode coverage,
and cost too (U08). A model can win likelihood and lose
every perceptual metric. Fair experiment: fix the
dequantization, report likelihood with uncertainty over
seeds, and pair it with sample precision/recall and
sampling cost at matched budgets. Red flag: accepting the
likelihood as the verdict. Rubric: both attacks plus the
paired experiment.
