# Interview bank, U10 generative bridge and synthesis

Date: 2026-10-06. Questions only. Keys in interview/keys-u10.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Define ancestral sampling in one sentence and give the
two steps for the lesson's mixture toy.
B2. Write the ELBO. Name its two terms and what each one
wants.
B3. Write the GAN value V. Which player maximizes it and
which minimizes it?
B4. Define the Jensen-Shannon divergence. What are its
symmetry and its maximum?
B5. The C07 toy: give log p(x), the ELBO, and the gap.
What does the gap equal?
B6. Name the three jobs (KL term, weight decay, Adam) and
state in one sentence each what job each one does.

## Deep ladder D1, the bound (5 follow-ups)

D1.1. Define the ELBO in one sentence.
D1.2. Toy: the C07 numbers. Give log p(x) = -0.7340,
ELBO = -0.8779, gap 0.1438.
D1.3. Derive or justify: log p(x) >= ELBO, and identify
the gap.
D1.4. Implement/debug: see T1 below.
D1.5. Changed constraint: q equals the true posterior.
What is the gap, and what does the ELBO equal?

## Deep ladder D2, objectives and evidence (5 follow-ups)

D2.1. Define mode collapse in one sentence.
D2.2. Toy: the C03 numbers. Give V at the start and after
the generator step, and say which direction is better
for G.
D2.3. Derive or justify: why does maximizing the ELBO not
maximize the likelihood?
D2.4. Compare: VAE vs GAN on training stability and on
sample sharpness. When do you pick each?
D2.5. Research critique: "Our VAE's ELBO improved, so the
model is better." Attack with the gap and the collapse
numbers.

## Analytical/quantitative (2)

Q1. Discrete toy: z in {0,1}, p(z) = [0.5, 0.5],
p(x=1|z=0) = 0.1, p(x=1|z=1) = 0.8, observe x = 1,
q = [0.4, 0.6]. Without a computer: p(x=1), log p(x)
(use ln 0.45 = -0.7985), the ELBO, the gap, the true
posterior, and KL(q||posterior).
Q2. Beta sweep: one-sample recon -0.9302, KL 0.3981.
Without a computer: totals at beta 0, 1, 10. The
collapsed q has KL 0 and recon -1.6695: at beta = 10,
which q wins, and what does the latent carry?

## Implementation/debug (1)

T1. A teammate trains a beta-VAE with beta = 10. The
regularized objective keeps falling and reconstructions
look fine, but latent traversals change nothing: every
z decodes to the same face. (a) Reproduce the mechanism
with the lesson's numbers: compute the beta = 10 totals
for the informative q (KL 0.3981, recon -0.9302) and the
collapsed q (KL 0, recon -1.6695). (b) Diagnose in two
sentences. (c) State the fix and how you would verify
it worked.

## Scenarios (2)

S1. Your team must pick a generative model for tabular
data augmentation where downstream users will train
classifiers on the synthetic data. Walk through the
choice between a VAE and a GAN: which failure mode of
each threatens the downstream task most, what do you
measure before deciding, and what is your rollback
signal after deployment?
S2. A reviewer asks for "predicted vs measured" evidence
that your training pipeline is sound. Design the
smallest experiment that answers: which quantity do you
predict, what do you measure, what are the controls,
and what would falsify the pipeline?

## Research critique (1)

R1. "Deeper trees always help. attention made recurrence
obsolete. the ELBO is the true objective." Pick one of
the three claims and demolish it with two numbers from
this build, then state the honest version of the claim.
