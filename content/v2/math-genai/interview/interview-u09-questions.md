# Interview bank, U09 score-based models and autoregressive LMs

Date: 2026-10-06. Questions only. Keys in
interview/keys-u09.md. Closed-book. Do not read the keys
first. Toys: the two-mode mixture, the bigram P, the tiny
transformer (d = 4).

## Breadth (6)

B1. Define the score. Why does it need no
normalizer?
B2. Compute the DSM target for x = 1.5, x_tilde =
1.7, sigma^2 = 0.04.
B3. Write one Langevin step. What are the roles of
the drift and the noise?
B4. Write the AR chain rule for [a, b, c]. Give
the toy NLL in bits.
B5. Write the 3x3 causal mask. What does row 1
allow?
B6. Name the four attention memory numbers: n =
1024 and 4096 bytes for 12 heads, 12 layers, fp32.

## Deep ladder D1, score to samples (5 follow-ups)

D1.1. Define the score in one sentence.
D1.2. Toy: compute s(1.0) = 1.999926. Show the
weight step.
D1.3. Derive the DSM target from the Gaussian
perturbation.
D1.4. Implement/debug: 20000 Langevin steps, mean
0.161. Is the sampler broken? Use the crossing
count.
D1.5. Changed constraint: a third mode appears at
x = 0. Predict the effect on crossings.

## Deep ladder D2, AR mechanics (5 follow-ups)

D2.1. Define teacher forcing in one sentence.
D2.2. Toy: compute the exposure gap 0.7653
bits/token. Show both terms.
D2.3. Derive why the causal mask makes one forward
pass train all positions.
D2.4. Implement/debug: row 0 of A is [0.5, 0.5,
0]. Name the bug and the fix.
D2.5. Research critique: "teacher forcing is
obsolete, always train free-run." Attack it.

## Analytical/quantitative (2)

A1. Perplexity 1.82574 on vocab 4. A second model
reports 15.0 on vocab 50000. Can you rank them?
Compute what is comparable.
A2. n = 8192, 12 heads, 12 layers, fp32: compute
the attention bytes. Propose a concrete config
under 12 GB and compute its bytes.

## Implementation/debug (1)

I1. A colleague's transformer trains to near-zero
loss but generates gibberish. Their row 0 is
[0.5, 0.5, 0]. Diagnose fully: the bug, why the
loss lies, the fix, and the test that prevents
recurrence.

## Changed-constraint scenarios (2)

S1. The score net must run at 10x fewer evals.
Name two levers (ODE solver, distillation) and
the tradeoff each buys.
S2. The vocab grows 4 -> 40000. Which toy numbers
change (softmax cost, perplexity bounds) and
which do not (chain rule, mask, exposure gap
form)?

## Research-critique (1)

R1. "Exact likelihood makes AR strictly better
than score-based models for science." Attack the
claim and design the deciding experiment.
