# Answer keys, lesson 07 (DDPM derivation and parameterizations)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run5a.py.

## E01

q(x_50 | x_49) = N(sqrt(0.9900505) x_49, 0.0099495).
Mean: 0.9950127 * 1.99860 = 1.98863. Variance:
0.0099495.

## E02

beta: 0.0001, 0.00994949, 0.02. alpha_bar: 0.9999,
0.77718008, 0.36356325.

## E03

x_2 = sqrt(alpha_2 alpha_1) x_0 + sqrt(alpha_2 beta_1)
eps_1 + sqrt(beta_2) eps_2. The noise terms merge:
variance alpha_2 beta_1 + beta_2 = 1 - alpha_2
alpha_1. General: q(x_t | x_0) = N(sqrt(alpha_bar_t)
x_0, (1 - alpha_bar_t) I).

## E04

sqrt(0.77718008) = 0.8815753, times 2.0 = 1.7631506.
sqrt(0.22281992) = 0.4720356, times 0.5 = 0.2360178.
Sum: 1.9991684 (audit 1.99917538 with full precision).

## E05

q(x_49 | x_50, x_0) = N(0.03956209 x_0 + 0.96013574
x_50, 0.00960075). beta_tilde_50 = 0.00960075.

## E06

x0_hat = (x_t - sqrt(1 - alpha_bar_t) eps_hat) /
sqrt(alpha_bar_t) = (1.99917538 - 0.4720356 * 0.4) /
0.8815753 = 2.05354466.

## E07

mu_rev = 0.03956209 * 2.05354466 + 0.96013574 *
1.99917538 = 2.00072225. x_49 = 2.00072225 +
sqrt(0.00960075) * 0.3 = 2.00072225 + 0.02939502 =
2.03011727.

## E08

q(x_100 | x_0) = N(1.20596, 0.63644). KL = 0.5 *
(1.20596^2 + 0.63644 - 1 - ln 0.63644) = 0.5 *
(1.45434 + 0.63644 - 1 + 0.45187) = 0.7713 nats.

## E09

0.5 * (1.99860391 - 2.00072225)^2 / 0.00960075 =
0.0002337 nats. Tiny because eps_hat = 0.4 is close
to eps = 0.5, so the two means nearly agree.

## E10

0.01, 0.09, 0.0025, 0.04, 0.01. Mean 0.0305.

## E11

No. Each count is Binomial(1000, 0.01): std =
sqrt(1000 * 0.01 * 0.99) = 3.146. Observed 3..16
spans mean 10 +- ~2.2 sigma. Ordinary sampling
noise.

## E12

Upper (beta_t): 2.03064640. Lower (beta_tilde_t):
2.03011727. Zero: 2.00072225. The theory forces
nothing: sigma_t^2 is underdetermined by the ELBO.

## E13

Three network evaluations for three steps. Full
sampling costs T = 100 evaluations per sample.

## E14

x0 error = eps error * sqrt(1 - alpha_bar_t) /
sqrt(alpha_bar_t) = 0.1 * 0.4720356 / 0.8815753 =
0.0535.

## E15

Training's noisiest x_100 still has SNR 0.57, so the
model never sees pure noise during training. At
generation we sample x_100 ~ N(0, 1): the first
reverse steps start out-of-distribution.

## E16

alpha_bar_10 = 0.5^10 = 0.0009766. The signal dies
in 10 steps. The remaining 90 steps add pure noise
to noise, wasting compute and giving the reverse
nothing gradual to invert.

## E17

Not a bug. The coefficient formulas are exact from
completing the square. Their sum has no identity
forcing it to 1. The near-1 value is a numerical
coincidence at t = 50, not a convex-combination
property.

## E18

The network never learns to denoise heavy noise.
Generation starts at x_T ~ N(0, 1) (t = 100) and
the first ~90 reverse steps run out-of-distribution.
Samples collapse or drift.

## E19

v = sqrt(alpha_bar_t) eps - sqrt(1 - alpha_bar_t)
x_0. Invert the rotation: eps_hat = sqrt(alpha_bar_t)
v_hat + sqrt(1 - alpha_bar_t) x0_hat_from_v, or
solve the 2x2 rotation directly. At t = 50 with the
toy numbers the rotation matrix is [[0.8815753,
-0.4720356], [0.4720356, 0.8815753]] on (eps, x_0).

## E20

DDPM: one U-Net evaluation per step dominates. The
noise algebra is O(1). VAE: one encoder plus one
decoder pass per step dominate. The KL is O(1).
Both are network-bound, but DDPM pays it T times
per sample at generation while the VAE pays once.

## L01

Forward process: a fixed Markov chain adding
Gaussian noise per a schedule. Toy: x_50 =
1.99917538. Derivation: the two-step unrolling in
E03, then induction. Implement: q_sample as in the
lesson. Compare: direct noising is O(1) and exact. Step simulation is O(T) for the same distribution.
Debug: exploding variance means the sqrt(alpha_t)
scaling is absent. Critique: Gaussian noise is
load-bearing for every closed form below. It is
restrictive but the restriction buys tractability.
Design: bisect the SNR curve. The crossing sits
where alpha_bar_t = 0.5.

## L02

Reverse posterior: q(x_{t-1} | x_t, x_0), Gaussian
by conjugacy once x_0 is known. Toy: (0.00960075,
0.03956209, 0.96013574). Derivation: multiply the
two Gaussians and complete the square. Implement:
posterior_coeffs and reverse_mean as in the
lesson. Compare: the true posterior needs the
unknown x_0. The learned reverse substitutes
x0_hat from eps_hat. Debug: upward drift means
eps_hat is biased high (or the coefficient
indexing is off by one). Check x0_hat against a
known pair. Critique: every reverse error is an
eps_hat error passed through fixed algebra, so
capacity belongs in the noise predictor, not the
sampling code. Design: run the loop with the three
sigma choices on fixed z draws and measure the
endpoint spread.

## L03

ELBO decomposition: the Markov structure splits the
bound into L_T + sum of middle KLs + L_0. Toy: L_T
= 0.7713 nats dominates. L_49 = 0.0002337 nats.
Derivation: write the joint, telescope the sum,
group by timestep. Implement: L_T_term and L_mid as
in the lesson. Compare: the full ELBO weights
timesteps by KL coefficients. The simplified loss
trains all t uniformly and samples better. Debug:
falling loss with worsening samples means the loss
weights the wrong timesteps for sample quality. Check per-t losses. Critique: the simplified loss
abandons the bound. Its justification is empirical
sample quality, not theory. Design: plot the KL
coefficient ratio curve across t.

## L04

Epsilon prediction: the network outputs the noise
added at step t. Toy: 0.1 noise error becomes
0.0535 x0 error. Derivation: x_t is linear in both
x_0 and eps, so the three targets are rotations of
each other. Implement: eps_to_x0 and x0_to_eps as
in the lesson. Compare: epsilon targets are
scale-free across t. X0 targets get harder at high
t. V is the middle ground. Debug: sampling inverts
the wrong parameterization when train and sample
code disagree. The round-trip test catches it.
Critique: "learning noise" versus "learning
structure" is wordplay. The network learns the
conditional expectation, whatever you call the
target. Design: train each parameterization on the
toy and compare per-t loss curves.

## L05

Sampling loop: from x_T ~ N(0, 1), iterate the
reverse Gaussian T times. Toy: 1.75690526,
1.74406522, 1.82846911. Derivation: each step
needs the network's eps_hat at the current x_t,
so steps are sequential. Implement: sample_loop as
in the lesson. Compare: DDPM needs T evaluations. DDIM (U08) reuses the network with fewer steps.
Debug: correlated samples mean reused z draws. Draw fresh noise per step per sample. Critique:
errors compound over T steps with no correction. The drift bound is an open empirical question.
Design: measure endpoint spread across sigma
choices and seeds.

## T1

The bug: t is 0-indexed (0..T-1) but net
expects 1-indexed (1..T). The noise level
is right: ab[t] for t in 0..T-1 reads
alpha_bar_1..alpha_bar_T, the correct
values. The label is wrong: net(xt, t)
conditions on timestep t while xt carries
noise level t+1, so every step applies the
wrong timestep's denoising rule (off by
one). Corrected: `t = int(rng.integers(1,
T + 1))` and `xt = np.sqrt(ab[t - 1]) * x0
+ np.sqrt(1 - ab[t - 1]) * eps`. With the
buggy code, timestep T never gets the
right label: t = T-1 labels noise level T
as T-1, and no call ever passes label T.

## S1

alpha_bar_T falls further (more product terms),
so the endpoint gets closer to pure noise: the
prior mismatch shrinks. Per-step cost is
unchanged. Total sampling cost rises 10x (1000
evaluations per sample). The SNR=1 crossing
moves later in t (more steps to reach the same
noise level) but earlier as a fraction of T.

## S2

Variance preservation assumed unit data variance:
Var(x_t) = alpha_t * 4 + beta_t != 1. The chain
endpoint is not N(0, 1). Fixes: (1) normalize the
data to unit variance first. (2) rescale the
schedule's variance target (variance-preserving
form with data variance s^2: x_t = sqrt(alpha_t)
x_{t-1} + sqrt(beta_t) s eps).

## R1

The claim drops the KL coefficients: the full ELBO
weights each timestep's squared error by 1/(2
sigma_t^2) times the coefficient ratio, while the
simplified loss weights them all 1. Uniform t
sampling then implies implicit weights proportional
to the sampling distribution, not the ELBO's.
Test: train three variants on the same data (full
ELBO, simplified, simplified with ELBO weights
restored) and compare held-out likelihood versus
sample quality metrics. If the reweighting helps
likelihood but hurts samples, the claim is
half-true at best.
