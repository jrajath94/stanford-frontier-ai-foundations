# Keys, interview U10 generative bridge and synthesis

Date: 2026-10-06. Each answer: minimum sufficient
explanation, strong answer, red flags, rubric, remediation.

## B1

Minimum: draw z ~ p(z), then x ~ p(x|z). Toy:
z in {0,1} fair coin, then x ~ N(-1,1) or
N(1,1).
Strong: adds the code order matters: latent
first.
Red flags: "draw x then z".
Rubric: 2. Remediation: C01.

## B2

Minimum: ELBO = E_q[log p(x|z)] - KL(q||p).
Recon term wants faithful decoding. KL wants
the encoder near the prior.
Strong: notes the tension between the terms.
Red flags: "the KL wants good reconstructions".
Rubric: 2. Remediation: C02.

## B3

Minimum: V = E[log D(real)] + E[log(1 -
D(fake))]. D maximizes, G minimizes.
Strong: states the alternating updates.
Red flags: swapped players.
Rubric: 2. Remediation: C03.

## B4

Minimum: JSD = 0.5 KL(p||m) + 0.5 KL(q||m),
m = (p+q)/2. Symmetric, max log 2.
Strong: gives the nats value 0.6931.
Red flags: "JSD is asymmetric like KL".
Rubric: 2. Remediation: C04.

## B5

Minimum: -0.7340, -0.8779, gap 0.1438 =
KL(q||posterior).
Strong: shows the posterior [0.25, 0.75]
computation.
Red flags: "the gap is zero".
Rubric: 1 per number. Remediation: C07.

## B6

Minimum: KL regularizes the encoder toward the
prior. weight decay regularizes weights
toward small values. Adam optimizes (chooses
steps, changes no loss).
Strong: adds AdamW vs Adam-with-L2.
Red flags: "Adam regularizes".
Rubric: 1 per job. Remediation: C06.

## D1

D1.1. A lower bound on log p(x) from Jensen on
the latent expectation.
D1.2. -0.7340, -0.8779, 0.1438.
D1.3. log p(x) = log E_q[p(x,z)/q(z)] >=
E_q[log(p(x,z)/q(z))] = ELBO. gap =
KL(q||p(z|x)) >= 0.
D1.4. See T1.
D1.5. Gap 0. ELBO = log p(x) (E16:
-0.7340).
Rubric: 2 per follow-up. Remediation: C07.

## D2

D2.1. G outputs one (or few) points always.
the loss stops moving.
D2.2. V start -0.5798, after -0.9163. lower
is better for G.
D2.3. The ELBO is the likelihood minus the KL
gap. maximizing it tightens the bound, and
only equals likelihood maximization when q is
exact.
D2.4. VAE: stable, blurry. GAN: sharp,
unstable. Pick VAE for latent use and stable
training. GAN for raw sample quality.
D2.5. Attack: the ELBO can rise while the gap
widens. T1: at beta = 10 the informative q
scores -5.07 vs collapsed -1.67, so "better
objective" picks the dead latent.
Rubric: 2 per follow-up. Remediation:
C02-C04, C06.

## Q1

p = 0.45, log p = -0.7985. ELBO: 0.4 ln 0.05
+ 0.6 ln 0.4 + H(0.4,0.6) = -1.1983 - 0.5497
+ 0.6730 = -1.0751. Gap 0.2765. Posterior
[0.1111, 0.8889]. KL = 0.2765 (matches).

## Q2

Totals: -0.9302, -1.3283, -5.0713. At beta =
10 the collapsed q wins (-1.6695 >
-5.0713). The latent carries nothing: q = p,
no information about x.

## T1

(a) Informative: -0.9302 + 10(0.3981) =
-5.0713. Collapsed: -1.6695 + 0 = -1.6695.
(Executed 2026-10-06.)
(b) Diagnosis: at beta = 10 the KL term
dominates, so training prefers q = p (KL 0)
over the informative q. the decoder learns to
ignore z and the latent dies (KL collapse).
(c) Fix: anneal beta from 0 upward, or lower
beta to ~1. Verify: latent traversals change
the output AND the KL stays clearly above 0
on held-out data.
Strong: adds monitoring mutual information or
the KL per dimension.
Red flags: "train longer".
Rubric: reproduce 2, diagnose 2, fix 1.
Remediation: C02, C06, lab Task 3.

## S1

Minimum: VAE failure: blurry/off-manifold
samples teach the classifier wrong boundaries.
GAN failure: mode collapse drops whole
subgroups, biasing the classifier silently.
Measure: downstream classifier accuracy on
real held-out data plus subgroup recall, and
sample diversity stats. Rollback: subgroup
recall drops or diversity collapses.
Strong: names a concrete diversity metric and
a threshold.
Red flags: picking on sample sharpness alone.
Rubric: failures 2, measures 2, rollback 1.
Remediation: C03, C04.

## S2

Minimum: predict the 1/B gradient-variance
ratio (4.0 for B=4 vs 16). measure with fixed
seeds and 4000 trials. controls: same data,
same code path, only B changes. falsified if
the measured ratio is far from 4.0 outside
Monte Carlo noise, which signals a broken
sampler or gradient bug.
Strong: adds reporting seed, trial count, and
both numbers.
Red flags: "the loss went down" as the
evidence.
Rubric: predict 1, measure 1, controls 1,
falsify 1. Remediation: C09.

## R1

Minimum: pick one claim and kill it with two
build numbers, e.g. trees: train 0.125->0.0
but val 0.25->0.25 (C04). honest version:
"depth helps until validation stalls."
Strong: the steelman-then-strike structure.
Red flags: attacking without numbers.
Rubric: two numbers 2, honest version 2.
Remediation: C08, C10.
