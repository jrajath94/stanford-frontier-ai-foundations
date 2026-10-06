# Interview bank, U04 Wasserstein and improved adversarial training

Date: 2026-10-06. Questions only. Keys in
interview/keys-u04.md. Closed-book. Do not read the keys
first. The running toy: P0 = delta_0, P_theta =
delta_theta, theta = 1, W1 = 1.0, JS = 1 bit with zero
gradient in theta.

## Breadth (6)

B1. Define a coupling and write its two marginal
constraints. Why must W1 minimize over couplings?
B2. State the Kantorovich-Rubinstein dual with its
three assumptions. What breaks first when the
Lipschitz assumption fails?
B3. Bound a critic's Lipschitz constant from its
weight matrices. Why is the product bound not an
equality?
B4. Weight clipping at c = 0.01: compute the toy gap
and name the two costs beyond the biased number.
B5. Write the gradient-penalty term. Compare the
two-sided and one-sided forms at ||grad|| = 0.3.
Name the penalty's blind spot.
B6. Define the calibration ratio. What does a ratio
of 0.5 tell you? What does 1.4 tell you?

## Deep ladder D1, coupling to the dual (5 follow-ups)

D1.1. Define W1 in one sentence, no jargon.
D1.2. Toy: compute both coupling costs on the 2x2
toy by hand. Show the cost matrix.
D1.3. Derive W1 = |theta| on point masses from the
coupling definition.
D1.4. Implement/debug: a colleague's coupling cost
comes out 2.0 on the toy where you expect 1.0. Name
the likeliest cause and the one-line fix.
D1.5. Changed constraint: the ground cost is 0 for
all pairs. What is W1 now, and what does the dual
sup give?

## Deep ladder D2, enforcing Lipschitz (5 follow-ups)

D2.1. Define the Lipschitz constant in one
sentence.
D2.2. Toy: at |w| = 0.3 and 2.5 compute both
penalty forms with lambda = 10. Show the formulas.
D2.3. Derive why 90 percent of clipped weights
land on the boundary, from U[-0.1, 0.1] to
[-0.01, 0.01].
D2.4. Implement/debug: the penalty reads 0 but the
critic is visibly spiky. Diagnose with the C06
blind spot.
D2.5. Research critique: "Weight clipping is
sufficient because it guarantees a Lipschitz
bound." Attack the claim: name what the guarantee
omits and the measurement that exposes it.

## Analytical/quantitative (2)

Q1. On the running toy, a critic reports gap 0.02
while the exact W1 on the mode-drop pair is 2.0.
Compute the calibration ratio and state in one
sentence what training optimizes versus what you
want.
Q2. A 5-layer critic has every weight at +-0.01.
Compute the output scale factor. Then state what
this does to the generator's gradient through the
critic.

## Implementation/debug (1)

T1. This code intends the WGAN critic loss:

```python
import numpy as np
def critic_loss(f, xp, xq):
    return float(f(xq).mean() - f(xp).mean())
```

It trains, the loss falls, but samples never
improve and the "W1" reads 3.0 on the theta = 1
toy. Find the two bugs (sign: the critic must
maximize E_p - E_q, so the loss to minimize is
E_q - E_p negated correctly. No Lipschitz
enforcement: nothing bounds f, so 3.0 exceeds
the true W1 = 1.0), fix them, and state the
general rule each fix enforces.

## Changed-constraint scenarios (2)

S1. The critic must be 2-Lipschitz, not
1-Lipschitz. How does the reported gap change on
the toy, and which penalty target changes?
S2. You must ship a domain-adversarial feature
extractor where the domain label correlates with
the task label. What breaks, and what is the
smallest repair to the objective?

## Research-critique (1)

R1. A paper reports "our WGAN reaches W1 = 0.001"
from the clipped critic's gap at c = 0.01 with
n_critic = 1 on one seed, claiming the generator
matches the data. Dismantle the claim in four
sentences, one per flaw, each naming the U04
section that exposes it.
