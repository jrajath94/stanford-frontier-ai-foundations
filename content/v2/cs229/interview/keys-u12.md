# Interview keys, U12

Date: 2026-10-06. Each answer gives the minimum sufficient
explanation, a strong answer, common red flags, a rubric, and
remediation. Computed values: numpy 1.26.x, float64.

## B1

Minimum: q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I).
The two quantities are the shrinkage sqrt(1 - beta_t) (equivalently
alpha_t) and the noise variance beta_t.
Strong: adds the regime (beta_t small, 1e-4 to 1e-2) and the
consequence (many small steps, smooth corruption).
Red flags: calling the forward process learned. Swapping the mean
and variance roles.
Rubric: 1 pt density, 1 pt both quantities, 1 pt regime.
Remediation: lesson C01, SL-01.

## B2

Minimum: x_t = sqrt(alpha-bar_t) x_0 + sqrt(1 - alpha-bar_t)
epsilon, epsilon from N(0, I). It exists because each step is a
linear Gaussian map, so the composition stays Gaussian with
accumulated variance.
Strong: cites the unrolling (14.6-14.9) and the covariance
telescoping that pins the noise variance at 1 - alpha-bar_t.
Red flags: "the noise adds up linearly" without the variance
accounting. Treating epsilon-hat_t as independent across t for
joint statements.
Rubric: 1 pt formula, 1 pt Gaussianity argument, 1 pt variance
accounting.
Remediation: lesson C02, SL-01.

## B3

Minimum: p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I).
Reasonable because each reverse step is small (small beta), so the
true reverse conditional is near-Gaussian, the continuous-time
theorem (14.32) supports it.
Strong: adds that sigma_t^2 is fixed to beta-tilde_t so only the
mean is learned, and that the timestep is embedded as network
input.
Red flags: "Gaussian because the data are Gaussian". Missing the
small-step justification.
Rubric: 1 pt family, 1 pt justification, 1 pt fixed variance.
Remediation: lesson C03, SL-02.

## B4

Minimum: L_0 (final reconstruction), L_1 ... L_{T-1} (denoising
terms), L_T (prior matching). L_T is dropped because p_theta(x_T)
is fixed N(0, I), so it carries no parameters (and is small since
alpha-bar_T is near zero).
Strong: writes the KL decomposition (14.16) and explains each
denoising term as matching one true Gaussian posterior.
Red flags: dropping L_0. Claiming L_T is dropped "because it is
hard".
Rubric: 1 pt three terms, 1 pt drop reason, 1 pt denoising
reading.
Remediation: lesson C05, SL-03.

## B5

Minimum: L_t(theta) = ||epsilon - epsilon_theta(x_t, t)||^2.
Each step consumes one clean example x_0, one sampled time t, one
fresh noise epsilon, x_t is built by the closed form.
Strong: states the five steps verbatim and notes the supervision
is free (the forward chain creates it).
Red flags: reusing epsilon across updates. Sampling t only near T.
Rubric: 1 pt loss, 1 pt three sampled objects, 1 pt five steps.
Remediation: lesson C06, SL-03.

## B6

Minimum: draw x_T from N(0, I), for t = T down to 1, form
mu_theta(x_t, t) by (14.23) and set x_{t-1} = mu_theta + sigma_t
xi with xi from N(0, I), xi = 0 at t = 1.
Strong: explains why the noise injection is needed (keeps the
chain stochastic, matches the reverse SDE) and why the last step
is deterministic.
Red flags: starting from a data point. Adding noise at t = 1.
Rubric: 1 pt prior draw, 1 pt loop, 1 pt final-step rule.
Remediation: lesson C08, SL-04.

## D1

D1.1: transition (14.1), beta_t in (0, 1), alpha_t = 1 - beta_t,
alpha-bar_t = product of alpha_1..alpha_t.
D1.2: alpha-bar_50 = 0.7772, mean 2.6447, std 0.4720.
D1.3: linear maps of Gaussians stay Gaussian, independent noise
variances add, the telescoping sum gives exactly 1 -
alpha-bar_t. Strong answers show one unrolling step.
D1.4: the noise scale is wrong: the signature (variance 40)
means the shrinkage sqrt(alpha_t) is absent or beta is in the
wrong units. Fix: assert alpha-bar monotone decreasing and check
Var(x_T) is near 1 on a test batch before any training.
D1.5: assumptions 1 (normalized scale) and 4 (continuous
pixels) break first. Fix: normalize or standardize inputs and
apply the Ho et al. section 3.3 discretization handling before
any likelihood comparison, otherwise ELBO numbers are not
comparable.
Rubric: 1 pt each, 2 pts for D1.3.
Remediation: SL-01, SL-05.

## D2

D2.1: log p_theta(x_0) >= E_q[log p_theta(x_0:T) / q(x_1:T |
x_0)], latent path x_1:T, auxiliary q the forward chain.
D2.2: x_50 = 3.0420, loss = (0.8417 - 0.9417)^2 = 0.01.
D2.3: substitute x_0 = (x_t - sqrt(1 - alpha-bar_t) epsilon) /
sqrt(alpha-bar_t) from (14.5) into (14.19), simplify with
alpha-bar_t = alpha_t alpha-bar_{t-1} to get (14.22).
D2.4: (a) is more consistent: loss near zero means the network
fit the training noise draws, and fresh test-time noise defeats
a memorized mapping, giving garbage. (b) would give
biased-but-coherent samples, not noisy garbage. One-line fix:
evaluate the loss on fresh noise draws, if held-out loss is
high, the model memorized and needs more data or less
capacity.
D2.5: defend the practice, attack the phrasing: the unweighted
loss is a reweighting of the ELBO terms, so it is not "worse" in
an absolute sense. Exact weights win on likelihood, the
unweighted loss wins on perceptual sample quality (Ho et al.).
The experiment: fix T and schedule, train both, compare held-out
weighted ELBO and FID.
Rubric: 1 pt each, 2 pts for D2.3.
Remediation: SL-03, SL-04.

## Q1

beta-tilde_t = beta_t (1 - alpha-bar_{t-1}) / (1 - alpha-bar_t).
Since alpha-bar_t = alpha_t alpha-bar_{t-1} < alpha-bar_{t-1},
the numerator is smaller than the denominator, so the ratio is
below 1 and beta-tilde_t < beta_t. Numeric: beta-tilde_50 =
0.00960, beta_50 = 0.00995.

## Q2

log q(x_t | x_0) = -||x_t - sqrt(alpha-bar_t) x_0||^2 /
(2(1 - alpha-bar_t)) + const. Gradient: -(x_t - sqrt(alpha-bar_t)
x_0) / (1 - alpha-bar_t). From (14.5), the numerator equals
sqrt(1 - alpha-bar_t) epsilon-hat_t, giving -epsilon-hat_t /
sqrt(1 - alpha-bar_t).

## T1

Bug 1: the noise scale is np.sqrt(beta_t), the notes fix
sigma_t^2 = beta-tilde_t, so it must be np.sqrt(beta_tilde_t).
Beta_t exceeds beta-tilde_t, so the loop injects excess noise at
every step, worst where precision matters (t near 1).
Bug 2: noise is added at t = 1, the notes set xi = 0 at the final
step. Fix: guard the noise with `if t > 1`.
Check: with an oracle epsilon predictor, one reverse step from a
known (x_t, x_0) must reproduce the posterior mean (14.19) to
numerical precision, and a full run from x_T must land near the
data manifold instead of noise.
Rubric: 1 pt per bug, 1 pt per fix, 1 pt check.
Remediation: lesson C03, C08.

## S1

The small-step assumption breaks: with 20 coarse steps each
reverse kernel must cover a large denoising jump, and the true
reverse conditional is no longer near-Gaussian. Fixes: learn the
variance (or a richer reverse family), use a noise schedule
designed for few steps, or switch to a deterministic sampler
(DDIM-style) that does not need the stochastic kernel to be
exact. Evidence: sample quality collapses while the training
loss looks fine.

## S2

Replacements: discrete diffusion with a categorical forward
transition (masking or uniform corruption), or absorbing-state
diffusion. The replacement must keep a known, tractable
stationary distribution at time T (the analog of N(0, I)) and a
closed-form or easily sampled posterior, so the ELBO argument
and the denoising reduction survive.

## R1

The conclusion is not sound. Training loss is the unweighted
denoising objective, the baseline comparison needs the same
objective on both. FID is a perceptual evaluator, and the two
rankings often disagree (SL-05). Decide with one experiment:
evaluate both models with the exact weighted held-out ELBO
(the likelihood evaluator) and with FID (the perceptual
evaluator) on the same held-out set. If the new model wins on
ELBO but loses on FID, it is the objective split, not
overfitting.
