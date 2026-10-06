# Interview bank, U02 variational divergence minimization

Date: 2026-10-06. Questions only. Keys in interview/keys-u02.md.
Closed-book. Do not read the keys first. The running toy: p =
Bernoulli(0.7), q = Bernoulli(0.4), ratios 0.5 and 1.75.

## Breadth (6)

B1. State Jensen inequality with its two conditions. When does
the direction flip?
B2. What is a variational family, and what does it cost?
B3. Write the f-divergence definition. Name the knob and its
two requirements.
B4. State the variational dual of an f-divergence. What does
each expectation need?
B5. Why does E_q[p/q] = 1 always hold, and what breaks it?
B6. Name the three gaps between a reported dual value and the
true divergence.

## Deep ladder D1, Jensen to lower bound (5 follow-ups)

D1.1. Define convex in one sentence, no jargon.
D1.2. Toy: f(x) = x^2, X in {0.4, 0.7} with equal weight.
Compute the Jensen gap.
D1.3. Derive the C03 lower bound on log p(x) in three lines
from Jensen.
D1.4. Implement/debug: a colleague codes the bound and it
exceeds log p(x) on a test case. Name the two likeliest
causes and how to tell them apart.
D1.5. Changed constraint: z is continuous. Which steps of the
derivation survive, and what replaces the finite sum?

## Deep ladder D2, f-divergence dual (5 follow-ups)

D2.1. Define the convex conjugate f*.
D2.2. Toy: for KL in nats, write T*(x) on the Bernoulli pair
and compute the dual value.
D2.3. Derive: why does sup_T E_p[T] - E_q[f*(T)] equal
D_f(p || q)?
D2.4. Implement/debug: a teammate restricts T to constants
and reports "divergence 0, models match". Diagnose with the
gap ledger.
D2.5. Research critique: "A bigger witness class always gives
a better divergence estimate." Attack the claim: name what
else must grow and what breaks if it does not.

## Analytical/quantitative (2)

Q1. On the toy, compute reverse KL from the generator -log2 t
by hand. Show both terms and verify f(1) = 0.
Q2. Predict the Monte Carlo standard error of the E_q[r]
estimate at N = 1600 before computing. Then state the
measured error scale from the lesson at N = 1000 and N =
10000 and judge whether the prediction law holds.

## Implementation/debug (1)

T1. This code intends D_KL(p || q) in bits:

```python
import numpy as np
p = np.array([0.3, 0.7])
q = np.array([0.6, 0.4])
kl = np.sum(np.log2(p / q))
```

It returns 0.5850, but the true value is 0.2651. Find the bug
(the missing p-weighting), fix it, and state the general rule
the bug violates.

## Changed-constraint scenarios (2)

S1. Your outcomes are images in R^d, not two coins. Which U02
facts survive unchanged, and which formula must change first?
S2. You must report a divergence number to a stakeholder who
will ship on it. List the four items in the shippable report
and say which U02 section each comes from.

## Research-critique (1)

R1. A paper claims a new divergence D_new with generator
f(t) = sqrt(t), reports D_new = -0.045 on a benchmark, and
concludes the model beats the baseline. Dismantle the claim
in three sentences.
