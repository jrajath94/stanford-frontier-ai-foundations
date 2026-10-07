# Interview bank: U05 (questions)

Unit: math-genmodels-U05. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u05.md.

## Breadth (6)

Q1. Why does a GAN have no likelihood function? What would you
need to compute one?
Q2. What does the optimal discriminator compute, and why is it a
function of the density ratio?
Q3. Derive the JSD identity for the GAN game in two lines. Where
does -log 4 come from?
Q4. Why does JSD saturate on disjoint laws while W1 does not?
Q5. What does the 1-Lipschitz constraint buy the WGAN critic, and
what breaks without it?
Q6. Name two ways the gradient penalty differs from weight
clipping, and when you would pick each.

## Deep ladders (2 x 5)

L1 (the game, end to end):
1. Define the minimax objective V(D, G).
2. Derive the pointwise optimal discriminator.
3. Rewrite D* through the density ratio.
4. Plug D* into V and show the JSD identity with the toy
   numbers.
5. The discriminator reaches 100 percent held-out accuracy.
   Explain why the generator then learns nothing.

L2 (pathologies):
1. Define mode collapse with the two-mode toy.
2. Show precision/recall split on the collapsed model.
3. Explain the bilinear spiral with the eigenvalue argument.
4. State why a falling generator loss is not progress.
5. Design the evaluation protocol you would trust instead of
   the loss.

## Analytical exercises (2)

A1. Prove max_D V = -log 4 + 2 JSD(p, q) starting from D*,
showing every algebra step.
A2. For 1-D Gaussians with equal variance, prove W1 = |mu1 -
mu2| via the CDF integral, and show a 1-Lipschitz critic
attains it.

## Implementation and debug (1)

D1. The WGAN-GP loss oscillates and samples degrade after looking
good at step 10k. The penalty term reads 0.0 throughout. Name
three candidate causes in order of likelihood, with one test for
each.

## Changed-constraint scenarios (2)

T1. You must deploy the discriminator on 8-bit fixed point with
no division. Redesign the score computation and state the cost
to the ratio estimate.
T2. Data arrives as a stream with no revisits. Redesign training
and evaluation, and name the failure mode that gets worse.

## Research critique (1)

R1. "Our GAN beats the VAE on FID, so adversarial training is the
better density estimator." Attack this claim with two arguments
from this unit and propose the fair experiment.
