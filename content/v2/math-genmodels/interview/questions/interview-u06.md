# Interview bank: U06 (questions)

Unit: math-genmodels-U06. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u06.md.

## Breadth (6)

Q1. Why does the score not need the partition function? What
does that buy?
Q2. Derive the explicit score-matching objective in three lines.
Where does the trace come from?
Q3. State Tweedie's formula. Why does it give the posterior mean
and not a sample?
Q4. Why does DDPM training sample t uniformly instead of
simulating the chain?
Q5. What does eta control in DDIM? What breaks when you stride
too aggressively?
Q6. Why does classifier-free guidance double the sampling cost?
When is w = 3 too large?

## Deep ladders (2 x 5)

L1 (from noise to data):
1. Define the score and compute it on N(0,1).
2. Write the forward chain closed form and evaluate x_4.
3. Derive one reverse kernel mean.
4. Contrast the DDIM and DDPM updates on the same step.
5. A sampler with 4 DDIM steps looks washed out. Diagnose and
   propose the fix.

L2 (theory and failure):
1. State the SDE limit of the chain.
2. State the probability flow ODE and its marginal claim.
3. Name the assumption that fails on manifold data.
4. Compute the boundary score for the smoothed uniform.
5. Explain why the ODE likelihood needs a named noise floor.

## Analytical exercises (2)

A1. Prove Tweedie's formula for Gaussian noise: E[x_0 | x] = x
+ sigma^2 grad_x log p(x).
A2. Show the DDIM eta = 0 update preserves the DDPM marginals
by induction on t.

## Implementation and debug (1)

D1. Training loss is 0.005 but samples are gray mush. The time
embedding is present. Name three candidate causes with one test
each, in order.

## Changed-constraint scenarios (2)

T1. Sampling budget is 4 network calls. Redesign the sampler,
state the quality cost, and say how you would measure it.
T2. Data are discrete tokens. Redesign the diffusion process and
name which unit concepts break.

## Research critique (1)

R1. "Our model has the best ODE likelihood, so it is the best
generative model." Attack with two arguments from this unit and
propose the fair experiment.
