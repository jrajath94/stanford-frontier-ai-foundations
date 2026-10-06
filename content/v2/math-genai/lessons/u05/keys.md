# Answer keys, lesson 05 (variational autoencoders)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run5a.py.

## E01

Prior p(z): input z (scalar), output a density value.
Encoder q(z | x): input x shape (2,), output mu and
logvar, scalars. Decoder p(x | z): input z scalar,
output xhat shape (2,).

## E02

log p(x) = log int p(x|z) p(z) dz. Insert q/q: = log
E_q[p(x|z) p(z)/q(z|x)]. Jensen (log concave): >=
E_q[log p(x|z) + log p(z) - log q(z|x)]. The last two
terms form -KL(q||p). Jensen is used at the log-E to
E-log step. The KL appears when grouping E_q[log p(z)]
- E_q[log q].

## E03

KL = 0.5 (0.16 + 0.5 - 1 - ln 0.5) = 0.5 (0.353147) =
0.1766 nats. In bits: 0.1766 / 0.6931 = 0.2547 bits.

## E04

The bound is E_q[1-sample ELBO] <= log p(x), an
expectation statement. The 1-sample value is one draw
of a random estimator. It can land above the marginal.
The expected ELBO is -3.2382 nats, below -1.9341 nats.
No violation.

## E05

z = mu + sigma eps is an ordinary differentiable
expression in mu and sigma with eps treated as a
constant input. Autodiff differentiates the formula.
A black-box sampler hides the dependence of the draw
on the parameters, so no gradient path exists.

## E06

The reparameterized estimator: SE 0.1345 versus 0.2622,
about half the noise at the same 64 samples. Both means
are within 3 SE of 0.8, but the reparameterized one
pins the value down twice as tightly.

## E07

Squared error: (1.2-0.824264)^2 + (0.4-0.412132)^2 =
0.141163 + 0.000147 = 0.141310. Term 1: -0.5 * 0.141310
/ 0.25 = -0.282620. Term 2: -2 ln 0.5 = 1.386294. Term
3: -ln(2 pi) = -1.837877. Sum: -0.7342 nats.

## E08

The reconstruction term divides by s^2. Shrinking s
from 0.5 to 0.05 multiplies the -0.5||x-xhat||^2/s^2
penalty by 100 for the same error, while the -d ln s
bonus grows only logarithmically. The model buys ELBO
by shrinking s, not by improving xhat.

## E09

Collapsed ELBO -0.4516 nats beats healthy -0.9108 nats
by 0.4592 nats. The winner uses no latent information:
q = prior, KL = 0. The objective prefers ignoring z.

## E10

Healthy at beta = 4: -0.7342 - 4(0.1766) = -1.4405
nats. Collapsed: -0.4516 nats. Collapsed still wins.
The healthy model does not survive at beta = 4 on
this toy.

## E11

[-2, -1], [-1, -0.5], [0, 0], [1, 0.5], [2, 1].
Adjacent distance: ||[1, 0.5]|| = sqrt(1.25) =
1.1180.

## E12

mu = 0.3 (0.3 + -0.7) = -0.12. Inference answers one
x with one forward pass because the encoder is already
fitted. Training must average the noisy ELBO over the
dataset and move the parameters. One pass gives no
gradient signal.

## E13

Test 4 (collapse diagnostic: KL per dim and the
traversal) catches it. Tests 1-3 and 5 check algebra
and code paths that the collapsed model also satisfies.

## E14

N(0.3 | -1, 0.25) = 0.0270, weighted 0.6 * 0.0270 =
0.0162. N(0.3 | 1, 0.25) = 0.2995, weighted 0.4 *
0.2995 = 0.1198. gamma_2 = 0.1198 / (0.0162 + 0.1198)
= 0.8802.

## E15

Guarantee: the log-likelihood never decreases across
an EM step. Limit: convergence is to a local optimum,
not the global one.

## E16

The posterior must be a location-scale family (a
differentiable transform of fixed noise). Discrete
latents break it: no differentiable path exists from
a parameter to a discrete outcome.

## E17

No. The MC standard error at 200000 samples is about
sigma/sqrt(N). The observed 0.0006 nats is well
within a few SE of the analytical 0.1766. It is
sampling noise, not a bug.

## E18

Posterior collapse: the encoder ignores the input on
all 32 dims. Confirm with a latent traversal: if the
decoded output does not change along any dim, the
diagnosis holds.

## E19

s-collapse (variance gaming): the likelihood term is
gamed by shrinking s. The toy fixes s = 0.5. The
general fix is a lower bound on s or a separate
validation of sample quality.

## E20

VAE: approximate likelihood via the ELBO, free
architecture. GAN: no likelihood at all, samples via
the game. Flows: exact likelihood, restricted to
invertible architectures.

## L01

ELBO: E_q[log p(x|z)] - KL(q||p), a lower bound on
log p(x). Toy: -0.7342 - 0.1766 = -0.9108 nats.
Derivation: the five-line proof in C02. Implement:
elbo_1sample as in the lesson. Compare: 1-sample
-0.9108 is one draw. Expected -3.2382. Marginal
-1.9341. The ordering expected <= marginal holds.
Debug: the classmate compares one draw against the
marginal. The bound is in expectation. Average over
many draws. Critique: the ELBO confounds bound
tightness with model quality. A collapsed model can
score higher. Design: fix the model, vary the q
family richness, and measure the gap change. Then
fix the family, vary the model, and measure the
marginal change.

## L02

Reparameterization: write the sample as a
differentiable function of the parameters plus
fixed noise. Toy: d/dmu E[z^2] = 2 mu = 0.8.
Derivation: grad E_q[f] = E_eps[grad f(mu + sigma
eps)] by the change of variables. Implement:
grad_estimators as in the lesson. Compare: SE
0.1345 vs 0.2622 at n = 64. Debug: gradients None
means the sample came from a black-box sampler. Route it through z = mu + sigma eps. Critique:
lower variance per sample does not guarantee
faster wall-clock training. The score estimator
can win if reparameterization is unavailable.
Design: sweep sigma^2 in {0.5, 0.1, 0.02} and
measure the SE ratio. Predict it grows as mass
concentrates.

## L03

Posterior collapse: q(z|x) = p(z) for all x. The
model ignores the latent code. Toy ledger: healthy
-0.9108 nats, collapsed -0.4516 nats. Collapse
wins by 0.4592. Derivation: with a perfect decoder,
ELBO = const - beta KL, maximized at KL = 0.
Implement: collapse_diagnostic as in the lesson.
Compare: the collapsed row has KL 0 and better
ELBO but zero latent use. Debug: KL per dim 0
with diverse samples means the decoder generates
diversity from noise or autoregression, not from
z. Check traversal. Critique: collapse is not
always bad. If the task needs no latent
representation, the collapsed model is the honest
one. Design: the rank experiment from mechanism C
shell 9.

## L04

Beta-VAE: E_q[log p(x|z)] - beta KL. Toy at beta
= 2: -0.7342 - 0.3531 = -1.0874 nats. At beta = 8:
-0.7342 - 1.4126 = -2.1468 nats. Derivation of the
boundary: collapse wins while recon_collapse >
recon_healthy - beta KL_healthy. Implement:
beta_elbo as in the lesson. Compare: beta = 1
keeps the code. Beta = 4 already loses it on this
toy. Debug: beta = 8, great samples, bad
reconstructions: the KL tax erased the codes. Lower beta or anneal. Critique: with beta != 1
the objective is not a bound on log p(x). It is a
Lagrangian trading reconstruction against rate.
Design: solve the boundary inequality for beta*
on the toy and verify by sweeping beta in {0.5,
1, 2, 4, 8}.

## L05

Amortized inference: one trained encoder answers
every new x in a single forward pass. Toy: mus
0.48, 0.15, -0.12, 0.54 for the four points.
Derivation: the encoder approximates the argmax
over q that per-point VI would compute. Training
fits this mapping once. Implement: infer vs
train_step as in the lesson. Compare: EM refits
per point exactly. The VAE answers instantly but
approximately. Debug: poor encoding far from
training data is amortization error. The fix is
coverage in the training distribution, not a
deeper encoder. Critique: amortization is wrong
when per-point exactness matters more than speed
(e.g. small-data science). Design: measure
traversal smoothness versus KL per dim across
dims and predict the correlation sign.

## T1

Likeliest cause: the encoder's logvar head
outputs a constant (e.g. the head is
disconnected or initialized to produce the
prior), or mu is detached from the graph. The
KL formula then sees mu = 0, logvar = 0 and
reports 0. One-line fix: connect the logvar
head to the encoder trunk (or remove a
detach() on the encoder output). The dropped
constant terms (-d ln s - (d/2) ln(2 pi))
shift every reported ELBO by the same
constant: comparisons across runs stay valid
but the number is not a bound on log p(x).

## S1

The closed form KL breaks: KL(q || mixture
prior) has no elementary form. The recon
term is unchanged. The KL becomes a Monte
Carlo estimate E_q[log q(z|x) - log
sum_k w_k N(z|m_k, v_k)], or a bound on it.
All lesson numbers that quote 0.1766 nats
change.

## S2

Failure: variance gaming, s -> 0 to inflate
the likelihood. Constraint: a floor s >=
s_min, or learn log s with a penalty. The
reconstruction becomes -0.5 sum_d (x_d -
xhat_d)^2 / s_d^2 - sum_d ln s_d - (d/2)
ln(2 pi).

## R1

The collapse ledger refutes it: the collapsed
model scores -0.4516 nats versus -0.9108 for
the healthy one, yet uses no latent code.
The exposing measurement is KL per dim plus
a latent traversal. The separating design:
fix the q family and vary the model while
measuring the numerical marginal (model
quality), then fix the model and vary the q
family while measuring the gap (bound
quality).
