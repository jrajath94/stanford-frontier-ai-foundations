# Answer keys, interview bank U05

Date: 2026-10-06. Ground truth: compute_run5a.py.
Interview provenance: role-derived practice, not employer
material. Format per answer: strong answer, red flags,
rubric, remediation.

## B1

Prior p(z): input z, output density. Encoder q(z |
x): input x (2,), output mu, logvar (scalars).
Decoder p(x | z): input z, output xhat (2,). Strong answer: all three components with shapes. Red flags:
"the encoder outputs x". Rubric: 2/2 all three with
shapes. 1/2 two. Remediation: U05-C01.

## B2

log p(x) = log int p(x|z)p(z)dz = log
E_q[p(x|z)p(z)/q(z|x)] >= E_q[log p(x|z)] -
KL(q||p). Jensen at the log-E to E-log step. Strong answer: the chain plus where Jensen enters. Red flags:
flipping the inequality. Rubric: 2/2 chain plus Jensen
step. 1/2 chain only. Remediation: U05-C02, C10.

## B3

0.5(0.16 + 0.5 - 1 + 0.6931) = 0.1766 nats =
0.2547 bits. Strong answer: both numbers with units.
Red flags: dropping the bits conversion. Rubric: 2/2
both numbers. 1/2 one. Remediation: U05-C05.

## B4

z = mu + sigma eps is differentiable in mu and
sigma with eps as a constant input. Autodiff
differentiates the formula. A black-box sampler
hides the parameter dependence. Strong answer: the
formula plus why autodiff sees through it. Red flags:
"the sampler is differentiable". Rubric: 2/2 formula
plus reason. 1/2 formula only. Remediation: U05-C04.

## B5

Healthy: recon -0.7342, KL 0.1766, ELBO -0.9108
nats. Collapsed: recon -0.4516, KL 0, ELBO -0.4516
nats. Collapsed wins by 0.4592 nats. Strong answer: all
six numbers plus the margin. Red flags: "higher ELBO
means a better model here". Rubric: 2/2 numbers plus
margin. 1/2 partial. Remediation: U05-C02, C07.

## B6

Amortized inference: one trained encoder answers
every new x in a single forward pass. EM E-step:
per-point iterative optimization of the
responsibilities, exact but not amortized. Strong answer: the contrast in one clean pair. Red flags: "EM
is amortized". Rubric: 2/2 both sides right. 1/2 one.
Remediation: U05-C11.

## D1

D1.1. The ELBO is a tractable lower bound on the
log-likelihood: expected reconstruction minus the
KL to the prior. Strong answer: the bound reading with
both terms. Red flags: "the ELBO equals the likelihood".
Rubric: 2/2 bound plus terms. 1/2 one. Remediation:
U05-C02, C10.
D1.2. -0.7342 - 0.1766 = -0.9108 nats. Strong answer:
the arithmetic with units. Red flags: adding instead of
subtracting. Rubric: 2/2 number plus arithmetic. 1/2
number only. Remediation: U05-C02.
D1.3. See B2. Strong answer: the Jensen chain with the
log-E to E-log step named. Red flags: flipping the
inequality. Rubric: 2/2 chain plus step. 1/2 chain only.
Remediation: U05-C02, C10.
D1.4. No. The bound is in expectation. One draw
can exceed the marginal. Average over many draws
before comparing. Strong answer: the expectation point
plus the averaging rule. Red flags: "one draw refutes
the bound". Rubric: 2/2 point plus rule. 1/2 point only.
Remediation: U05-C10.
D1.5. The Gaussian KL closed form breaks. It
becomes a Monte Carlo estimate E_q[log q(z|x) -
log sum_k w_k N(z|m_k,v_k)]. Strong answer: what breaks
and the replacement estimator. Red flags: "the closed
form still works". Rubric: 2/2 break plus replacement.
1/2 one. Remediation: U05-C03, C05.

## D2

D2.1. Posterior collapse: q(z|x) = p(z) for all x. The
model ignores the latent code. Strong answer: the
equation plus the plain reading. Red flags: "the decoder
is broken". Rubric: 2/2 equation plus reading. 1/2
equation only. Remediation: U05-C07.
D2.2. Healthy: -0.7342, 0.1766. Collapsed:
-0.4516, 0.0000. ELBOs -0.9108 vs -0.4516 nats. Strong answer: all six numbers. Red flags: "collapsed loses".
Rubric: 2/2 all numbers. 1/2 partial. Remediation:
U05-C02, C07.
D2.3. With a perfect decoder ELBO = const - beta
KL, maximized at KL = 0, i.e. q = prior. Strong answer:
the reduced objective plus the maximizer. Red flags:
"the optimizer still uses the latent". Rubric: 2/2
objective plus maximizer. 1/2 objective only.
Remediation: U05-C07, C08.
D2.4. Collapse. Confirm with a latent traversal:
constant decoded output along every dim proves
it. Strong answer: the diagnosis plus the traversal
test. Red flags: "low KL proves the model is good".
Rubric: 2/2 diagnosis plus test. 1/2 diagnosis only.
Remediation: U05-C07, C09.
D2.5. The ledger refutes it: collapse scores
higher with zero latent use. The exposing
measurement is KL per dim plus traversal. Strong answer:
the ledger numbers plus the exposing measurement. Red flags: "ELBO rank equals model rank". Rubric: 2/2 numbers
plus measurement. 1/2 numbers only. Remediation: U05-C07,
C12.

## Q1

Healthy at beta = 4: -0.7342 - 0.7063 = -1.4405
nats. Collapsed: -0.4516 nats. Collapsed wins. The
healthy model does not survive. Strong answer: both
numbers plus the survival verdict. Red flags: "beta = 4
saves the latent". Rubric: 2/2 numbers plus verdict. 1/2
numbers only. Remediation: U05-C07, C08.

## Q2

-0.4516 = -0.7342 - beta* 0.1766. beta* =
0.2826/0.1766 = 1.6007. Strong answer: the equation plus
the solved value. Red flags: sign errors in the solve.
Rubric: 2/2 equation plus value. 1/2 equation only.
Remediation: U05-C02, C08.

## T1

The encoder's logvar head is disconnected or
constant (or mu is detached), so the KL formula
sees mu = 0, logvar = 0. Fix: connect the head to
the encoder trunk (remove the detach). Strong answer:
the cause plus the exact fix. Red flags: "retrain with
more data". Rubric: 2/2 cause plus fix. 1/2 cause only.
Remediation: U05-C01, C07.

## S1

Variance gaming: s -> 0 inflates the likelihood
without improving xhat. Fix: a floor s >= s_min
or a penalty on small s. Strong answer: the mechanism
plus the fix. Red flags: "small s means a good fit".
Rubric: 2/2 mechanism plus fix. 1/2 mechanism only.
Remediation: U05-C06, C12.

## S2

Collapse pressure rises: 64 dims each pay KL, and
the decoder can more easily reconstruct without
them. Watch KL per dim (active count) and
traversals. Strong answer: the pressure argument plus
the two monitors. Red flags: "more dims fix collapse".
Rubric: 2/2 argument plus monitors. 1/2 argument only.
Remediation: U05-C07, C09.

## R1

It fails for discrete latents: no differentiable
path exists. The toy proves only that on this
Gaussian toy at n = 64 the reparameterized SE is
~2x smaller, both unbiased. The mind-changing
experiment: a latent family where the score
estimator with a control variate beats
reparameterization on wall-clock gradient noise. Then
the claim's "always" falls. Strong answer: the scope
limit, the toy's exact claim, and the killing experiment.
Red flags: "reparameterization always wins". Rubric: 2/2
limit plus experiment. 1/2 limit only. Remediation:
U05-C04, C12.
