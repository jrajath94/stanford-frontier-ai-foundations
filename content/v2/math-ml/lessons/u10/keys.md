# Keys, lesson 10 generative bridge and synthesis

Date: 2026-10-06. All numbers computed 2026-10-06 unless marked
as hand arithmetic (verifiable by hand).

## E01

Accept the loop. at n = 1000, seed 7, the z=1 share is
within 0.03 of 0.5 (Monte Carlo noise ~0.016).

## E02

0.5(0.8413) + 0.5(1 - 0.8413) = 0.5. Confirmed.

## E03

sqrt(0.5 * 0.5 / 10) = 0.1581. The observed 0.3 is
about 1.3 standard errors below 0.5: unremarkable.

## E04

p(x) = sum_z p(z) p(x|z) is symmetric in the label
names. It proves the latent labels are unidentifiable
from x alone.

## E05

KL = E_q[(z-mu)^2/(2 sigma^2)... ] Accept the standard
derivation ending at 0.5(mu^2 + sigma^2 - 1 - log
sigma^2).

## E06

KL = 0.5(0 + 4 - 1 - log 4) = 0.5(3 - 1.3863) =
0.8069. The encoder claims twice the prior spread:
diffuse, not collapsed.

## E07

-0.5(0.0225) - 0.9189 = -0.01125 - 0.9189 =
-0.9302. Confirmed.

## E08

One eps draw decides the recon term. its variance is
the full latent variance. Standard fix: more Monte
Carlo samples per step, or the analytic KL (already
used) plus a low-variance gradient estimator.

## E09

V = log 0.8 + log 0.1 = -2.5257. Better for G (lower
V): D is nearly fooled.

## E10

Accept pseudocode with D ascending V and G
descending E[log(1 - D(G(z)))].

## E11

If G(z) = 1.0 always and D(1.0) is high, V is
constant in G's parameters: no gradient, no
learning. D sees one repeated point and cannot
demand variety.

## E12

Identical: JSD 0. Disjoint: m = [0.5, 0.5],
JSD = 0.5 log 2 + 0.5 log 2 = log 2 = 0.6931.

## E13

Accept code with assert abs(jsd(p,q) -
jsd(q,p)) < 1e-12.

## E14

0.1438/0.7340 = 19.6%. Not tight: a fifth of the
log-likelihood is gap.

## E15

Posterior [0.25, 0.75]. KL = 0.5 log(0.5/0.25) +
0.5 log(0.5/0.75) = 0.3466 - 0.2027 = 0.1438.
Confirmed.

## E16

ELBO = E_post[log p(x,z)] + H(post). E =
0.25(-2.1203) + 0.75(-1.0217) = -1.2963. H =
0.5623. ELBO = -0.7340 = log p(x). Gap 0.

## E17

The term 0 * log(0.36/0): 0 * (-infinity) is
undefined/NaN in code. The ELBO sum breaks:
support mismatch made concrete.

## E18

The six val residuals are order 0.1-0.2, so the
MSE standard error is roughly 0.01-0.02,
comparable to the 0.0066 rise. The rise is
within noise: no evidence of harm.

## E19

"Bagging usually cuts validation error up to the
rho floor. on tiny validation sets the curve is
noisy (f04: 0.0278 at B=5, 0.0344 at B=25 on 6
points)."

## E20

|4.2887 - 4.0|/4.0 = 7.2%. Vindicated within
Monte Carlo noise at 4000 trials.

## E21

Unanswerable without running: the ratio is a
random variable over seeds. Any directional
prediction is a guess. the honest answer runs
it.

## E22

Total = -0.9302 + 0.5(0.3981) = -1.1293. A
pure-reconstruction fan picks beta = 0.

## E23

Adam with L2 adds the decay inside the adaptive
scaling. AdamW applies the decay outside it, as
a plain step. AdamW is the decoupled one.

## E24

The ELBO can rise while the gap widens: the
bound improves but the model stays far from
the likelihood. T1: with beta = 10 the
informative q scores -5.07 vs collapsed -1.67,
so "better objective" picks the dead latent.

## E25

Accept a fair two-sentence steelman, e.g.:
"Attention trains in parallel and links distant
positions directly. on fixed-length tasks it
beats recurrence on both speed and accuracy."

## E26

Accept bullets covering: log-sum-exp trick,
Jensen, the four assumptions, the gap as KL.

## E27

The honest route is a captions API or a
transcript endpoint reachable from an
unfiltered network, or the official NPTEL
transcript PDFs if the course page exposes
them under JS rendering.

## E28

TAUGHT+ASSESSED records artifact existence:
lesson plus keys exist for every row. SOURCE
ATTRIBUTION PENDING records that no instructor
artifact confirmed the leaf. Both are true at
once.

## E29

Accept code with assert elbo <= logpx and
abs((logpx - elbo) - 0.1438) < 1e-4.

## E30

Accept any hypothesis of the form: claim,
baseline, metric, falsification condition.

## Ladders

L01. (1) Draw z, then x|z. (2) The ten draws.
(3) Marginalization over z. (4) Accept the
sampler. (5) z = rng.standard_normal(n). the
rest is unchanged.

L02. (1) E_q[log p(x|z)] - KL(q||p). (2)
-1.3283, KL 0.3981. (3) Jensen on log sum_z
p(x,z). (4) Accept z = mu + sigma*eps. (5)
Collapsed: KL -> 0, latent dead. executed:
beta=10 totals -5.07 vs -1.67.

L03. (1) E[log D(real)] + E[log(1-D(fake))].
(2) -0.5798 to -0.9163. (3) D ascends V. G
descends the fake term. (4) Accept the loop.
(5) See interview keys-u10 T1.

L04. (1) 0.5KL(p||m)+0.5KL(q||m). (2) 0.0462.
(3) Symmetry from the m construction. max at
disjoint supports. (4) Accept the code. (5)
When the goal is sample quality: likelihood
rewards covering all modes, painting mass
off-data.

L05. (1) Fraction of samples <= x. (2) 0.3 vs
0.5. (3) Each sample contributes one jump of
1/n. (4) Accept the plot. (5) The staircase
has 1e5 tiny steps: visually smooth, still
stepwise.

L06. (1) KL: encoder honesty. Weight decay:
weight size. Adam: step choice. (2) -0.9302,
-1.3283, -5.0713. (3) beta -> infinity forces
q = p. (4) Accept the sweep. (5) "The
optimizer regularizes": false. Adam changes
steps, not the loss surface's preferences.

L07. (1) E[phi(X)] >= phi(E[X]) for concave
phi... (direction: phi(E[X]) >= E[phi(X)]).
(2) Valid q, expectation over q, support,
concavity of log. (3) Accept the derivation
with the KL gap. (4) Accept the code. (5)
The log term: 0 * log(0) -> NaN.

L08. (1)-(3) The three claims with killers and
qualifiers. (4) Accept any honest fourth kill.
(5) Accept a concrete audit proposal.

L09. (1) Predicted from theory, measured from
code. (2) 4.0 vs 4.2887. (3) Var(mean) =
sigma^2/B. (4) Accept the seeded loop. (5)
Missing: seed, trial count, and the predicted
value. Without them the number is decoration.

L10. Accept the timed delivery. score with the
rubric: assumptions 2, build numbers 2,
failure mode 2, regime qualifier 2,
counterargument 2. Name the weakest axis
honestly.
