# Interview bank, U03 GAN foundations and applications

Date: 2026-10-06. Questions only. Keys in
interview/keys-u03.md. Closed-book. Do not read the keys
first. The running toy: p_data = N(0,1), p_g = N(1,0.5),
D*(0) = 0.7870, max_D V = -1.3537 = 2 JS - 2.

## Breadth (6)

B1. Write the minimax objective. Which player does what?
B2. State the optimal discriminator and the one-line
derivation.
B3. What does the game minimize when D is optimal? State
the identity and its three assumptions.
B4. At D = 0.001, compare the minimax and non-saturating
generator gradients. Which pathology does the fix
address, and which does it not?
B5. Define mode collapse. Name the toy number that
diagnoses it and the number that hides it.
B6. Name the three gradient pathologies. For each, give
the standard guard.

## Deep ladder D1, the game to the JS identity (5 follow-ups)

D1.1. Define the minimax game in one sentence, no jargon.
D1.2. Toy: compute D*(1) on the running toy by hand.
Show a and b.
D1.3. Derive max_D V = 2 JS - 2 in four lines from D*.
D1.4. Implement/debug: a colleague's V(D*, toy) comes
out -inf. Name the likeliest cause and the one-line fix.
D1.5. Changed constraint: the supports are disjoint.
What is JS, what is its gradient in the generator
parameters, and what does the game do now?

## Deep ladder D2, saturation and loss surgery (5 follow-ups)

D2.1. Define saturation in one sentence.
D2.2. Toy: at D = 0.001 compute both generator
gradients in t. Show the formulas.
D2.3. Derive the non-saturating identity: E_{p_g}[-log2
D*] = KL(p_g || p_data) - KL(p_g || m) + 1.
D2.4. Implement/debug: the G loss prints +inf after 20
steps. Diagnose with the C07 pathology list.
D2.5. Research critique: "The non-saturating loss
dominates the minimax loss, so the minimax form should
be retired." Attack the claim: name what the fix
changes about the objective and one setting where the
original is the right tool.

## Analytical/quantitative (2)

Q1. On the running toy, verify the identity numerically:
state V, JS, and the identity error from the lesson, and
explain why the clip at 1e-15 is load-bearing.
Q2. A batch of 64 has one fake with D = 1e-9 and the rest
at D = 0.5. Compute that sample's loss contribution in
bits and its gradient scale d/dD log2 D. Then state what
the eps = 1e-7 clip changes.

## Implementation/debug (1)

T1. This code intends the GAN value in bits:

```python
import numpy as np
def gan_value(D, xd, xf):
    return np.log(D(xd)).mean() + np.log(1 - D(xf)).mean()
```

It runs without error but the numbers disagree with the
lesson by a constant factor, and once it returned -inf
with no explanation. Find the two bugs (natural log
instead of log2. No (0,1) assert before the logs), fix
them, and state the general rule each fix enforces.

## Changed-constraint scenarios (2)

S1. D must output hard {0, 1} decisions. Which parts of
U03 survive, what breaks first, and what is the smallest
repair?
S2. You must ship a conditional generator for 1000
classes with little data per class. Compare one
conditional GAN against 1000 separate GANs on three
axes: parameter sharing, rare-class behavior, and
evaluation cost.

## Research-critique (1)

R1. A paper reports "our GAN reaches JS = 0.001" from the
empirical game value with a 2-layer critic and N = 500
samples, and claims the generator matches the data.
Dismantle the claim in three sentences, one per flaw,
each naming the U03 section that exposes it.
