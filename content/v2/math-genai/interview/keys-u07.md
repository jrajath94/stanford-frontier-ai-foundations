# Answer keys, interview bank U07

Date: 2026-10-06. Ground truth: compute_run5a.py.
Interview provenance: role-derived practice, not employer
material. Format per answer: strong answer, red flags,
rubric, remediation.

## B1

q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1},
beta_t). q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0,
(1 - alpha_bar_t) I). alpha_bar_t is the fraction
of the original signal surviving at step t. Strong answer: both kernels plus the alpha_bar reading. Red flags: "alpha_bar is the noise". Rubric: 2/2 both kernels
plus reading. 1/2 one kernel. Remediation: U07-C01, C03.

## B2

beta_50 = 0.00994949. alpha_bar_50 = 0.77718008.
SNR_50 = 3.4879. Strong answer: all three numbers. Red flags: "SNR below 1 at t = 50". Rubric: 2/2 all numbers.
1/2 two. Remediation: U07-C02, C03.

## B3

q(x_49 | x_50, x_0) = N(0.03956209 x_0 +
0.96013574 x_50, 0.00960075). Strong answer: the two
coefficients plus the variance. Red flags: swapping the
coefficients. Rubric: 2/2 coefficients plus variance.
1/2 partial. Remediation: U07-C05.

## B4

x0_hat = (1.99917538 - 0.4720356 * 0.4) /
0.8815753 = 2.05354466. mu_rev = 0.03956209 *
2.05354466 + 0.96013574 * 1.99917538 =
2.00072225. Strong answer: both numbers with the
arithmetic. Red flags: forgetting the x0_hat step. Rubric:
2/2 both numbers. 1/2 one. Remediation: U07-C06.

## B5

L_T (prior match), L_{t-1} middle KLs (denoising
steps), L_0 (decoder). Toy L_T = 0.7713 nats =
1.1127 bits. Strong answer: the three terms plus the toy
number in both units. Red flags: "the ELBO is one term".
Rubric: 2/2 terms plus number. 1/2 terms only.
Remediation: U07-C07.

## B6

Sample x_0. Sample t uniform 1..T. Sample eps.
Form x_t by direct noising. Compute ||eps -
eps_hat(x_t, t)||^2 and backprop. Strong answer: the
five steps in order. Red flags: simulating the chain to
reach x_t. Rubric: 2/2 five steps. 1/2 three. Remediation:
U07-C09, C10.

## D1

D1.1. The forward process is a fixed Markov chain
adding small Gaussian noise per a schedule. Strong answer: fixed, Markov, and Gaussian in one sentence. Red flags: "the forward process is learned". Rubric: 2/2 all
three properties. 1/2 two. Remediation: U07-C01.
D1.2. 0.8815753 * 2.0 = 1.7631506. 0.4720356 * 0.5 = 0.2360178. Sum 1.9991684
(audit 1.99917538). Strong answer: the two products, the
sum, and the audit note. Red flags: "the sum is exactly
2". Rubric: 2/2 products plus sum. 1/2 partial.
Remediation: U07-C04.
D1.3. See lesson E03: merge the two noise terms
into variance 1 - alpha_2 alpha_1, induct. Strong answer: the merge plus the induction. Red flags: "the
variances add to 1". Rubric: 2/2 merge plus induction.
1/2 merge only. Remediation: U07-C03.
D1.4. Missing: the sqrt(alpha_t) mean scaling.
Without it Var(x_t) = Var(x_{t-1}) + beta_t
grows linearly. Strong answer: the missing scaling plus
the growth consequence. Red flags: "variance stays 1
anyway". Rubric: 2/2 scaling plus consequence. 1/2
scaling only. Remediation: U07-C01, C11.
D1.5. alpha_bar_10 = 0.5^10 = 0.00098. The
signal dies in 10 steps. 90 steps waste compute
and the reverse has nothing gradual to invert. Strong answer: the number plus both consequences. Red flags:
"more steps are always better". Rubric: 2/2 number plus
consequences. 1/2 number only. Remediation: U07-C02, C12.

## D2

D2.1. Minimize ||eps - eps_hat(x_t, t)||^2 over
uniform t: predict the added noise at every noise
level. Strong answer: the objective plus the plain
reading. Red flags: "predict x_0 directly". Rubric: 2/2
objective plus reading. 1/2 objective only. Remediation:
U07-C08, C10.
D2.2. 0.01, 0.09, 0.0025, 0.04, 0.01. Mean
0.0305. Strong answer: all five values plus the mean.
Red flags: "the mean is the loss". Rubric: 2/2 values
plus mean. 1/2 partial. Remediation: U07-C10.
D2.3. Both Gaussians share variance beta_tilde_t,
so KL = 0.5 (mu_q - mu_p)^2 / beta_tilde_t: a
squared mean difference. Strong answer: the shared
variance plus the reduced form. Red flags: keeping the
full KL formula. Rubric: 2/2 variance plus form. 1/2
form only. Remediation: U07-C06, C07.
D2.4. The simplified loss weights all t
uniformly while sample quality needs the right
per-t emphasis (or vice versa). Diagnostic:
per-t loss curves and sample metrics, not the
mean loss. Strong answer: the mismatch plus the
diagnostic. Red flags: "the mean loss is enough". Rubric:
2/2 mismatch plus diagnostic. 1/2 mismatch only.
Remediation: U07-C10.
D2.5. The claim drops the KL coefficients: the
ELBO weights each squared error by 1/(2
sigma_t^2) times the coefficient ratio. The
simplified loss weights them 1. Uniform t sampling
adds its own implicit weights. Test: train full
ELBO vs simplified vs ELBO-reweighted simplified
and compare likelihood against sample quality. Strong answer: the dropped coefficients plus the three-way
test. Red flags: "uniform t is neutral". Rubric: 2/2
coefficients plus test. 1/2 coefficients only.
Remediation: U07-C07, C09, C10.

## Q1

Ratio: 0.7713 / 0.0629 = 12.3. The toy ELBO's
mass sits in the prior term: the noisy schedule
endpoint dominates, not the denoising steps. Strong answer: the ratio plus the mass reading. Red flags: "the
denoising steps dominate". Rubric: 2/2 ratio plus
reading. 1/2 ratio only. Remediation: U07-C07.

## Q2

x0 error = eps error * sqrt(1 - alpha_bar_t) /
sqrt(alpha_bar_t) = 0.1 * 0.4720356 / 0.8815753
= 0.0535. Strong answer: the formula plus the number.
Red flags: "error is constant in t". Rubric: 2/2 formula
plus number. 1/2 number only. Remediation: U07-C04.

## T1

t is 0-indexed (0..T-1) but net expects 1..T.
The noise level is right (ab[t] reads
alpha_bar_1..alpha_bar_T). The label is wrong:
net applies the step-t rule to noise level t+1.
Fix: sample t in 1..T, index ab[t-1]. Strong answer: the
indexing mismatch plus the fix. Red flags: "the noise
levels are wrong". Rubric: 2/2 mismatch plus fix. 1/2
mismatch only. Remediation: U07-C09, C12.

## S1

alpha_bar_T falls further (more product terms):
closer to pure noise, smaller prior mismatch.
Serving cost rises 10x (1000 evaluations per
sample). The SNR = 1 crossing moves later in t
but earlier as a fraction of T. Strong answer: the
benefit, the cost, and the crossing shift. Red flags:
"T = 1000 is free". Rubric: 2/2 benefit plus cost. 1/2
one. Remediation: U07-C02, C12.

## S2

Var(x_t) = alpha_t * 4 + beta_t != 1 breaks the
unit-variance chain. The endpoint is not N(0, 1).
Fixes: normalize data to unit variance, or use
the variance-preserving form with data variance
s^2 in the noise scale. Strong answer: the break plus
both fixes. Red flags: "the endpoint is still N(0,1)".
Rubric: 2/2 break plus fixes. 1/2 break only.
Remediation: U07-C01, C11.

## R1

Defend with a caveat: the ELBO decomposition adds
the prior-matching term L_T and the telescoping
justification that the per-noise-level losses sum
to a bound on log p(x_0). Stacked denoising
autoencoders have no such joint bound. Separating
experiment: train the same per-t denoisers with
and without the L_T term and the ELBO weighting. If likelihood and prior-mismatch metrics move
while per-t denoising error stays fixed, the ELBO
structure adds something real. Strong answer: the
defense, the caveat, and the separating experiment. Red flags: "SDAEs have the same bound". Rubric: 2/2 defense
plus experiment. 1/2 defense only. Remediation: U07-C07,
C12.
