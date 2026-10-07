# Interview bank U03: Transformer accounting and speculative inference

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Write the full training FLOP formula with the
attention term. When does the attention term dominate?

B2. State per-token decode FLOPs. Why is decode
bandwidth-bound at batch 1 despite low FLOPs?

B3. Write the KV-cache byte formula. Compute it for B=1,
T=32768, L=32, n=4096, fp16.

B4. Define prefill. Why is time-to-first-token dominated
by prefill at long prompts?

B5. Give prefill and decode arithmetic intensities. Name
the bound for each at the toy ridge of 150 FLOP/byte.

B6. Define the speculative decoding speedup formula with
all symbols.

B7. Write the token acceptance rule. What distribution
does a rejected token get resampled from?

B8. State the two exactness assumptions of speculative
decoding.

## Deep ladder 1: speculative decoding economics

L1a. Define: what are c, gamma, and a?
L1b. Toy: c=0.05, a=0.8, gamma=5. Compute E[k] and the
speedup.
L1c. Derive: show E[k] = (1-a^{gamma+1})/(1-a) from the
per-position acceptance model.
L1d. Implement and complexity: write `speedup(a, gamma,
c)` and state the cost of the sweep.
L1e. Compare: (a=0.8, c=0.05) vs (a=0.3, c=0.5) at
gamma=5. Which wins and by how much?
L1f. Debug: formula predicts 2.9x, measured 1.2x. Name
two causes.
L1g. Critique: state the constant-a assumption and how
real drafts violate it.
L1h. Design: propose the gamma sweep experiment on a
real pair. Name the controls and the success criterion.

## Deep ladder 2: exactness

L2a. Define: what does "exact" mean for speculative
decoding output?
L2b. Toy: p=[0.2,0.8], q=[0.7,0.3]. Compute the accept
probability for token 0.
L2c. Derive: show accepted mass + resample mass = p(x)
for one position.
L2d. Implement and complexity: write the resample step
and state its cost relative to model FLOPs.
L2e. Compare: speculative sampling vs greedy draft
acceptance. When is each valid?
L2f. Debug: output histogram drifts from p over 50K
rounds. Name the bug class.
L2g. Critique: state the shared-tokenizer assumption
and construct the failure.
L2h. Design: propose the histogram exactness test. Name
N, the tolerance, and what failure would look like.

## Analytical exercises

A1. P=7e9, D=1e12, L=32, n=4096. Compute the dense term
6PD and the attention term 12LnTD at T=2048 and at
T=32768. Give the attention share in each case.

A2. A=0.9, gamma=7, c=0.1. Compute E[k] and the speedup.
Then find the break-even a at gamma=7, c=0.1 (solve
E[k] = gamma*c+1 numerically to 2 decimals).

## Implementation / debug task

D1. The acceptance loop below is meant to implement
speculative sampling, but the output distribution is
wrong. Find the bug, fix it, and state the invariant.

```python
import numpy as np
def accept(p, q, drafts, rng):
    out = []
    for x in drafts:
        if rng.random() < p[x] / q[x]:  # bug is here
            out.append(x)
        else:
            break  # bug: no resample
    return out
```

## Changed-constraint scenarios

S1. Constraint change: the draft model is free (c=0)
but verification costs 1+slope*gamma steps with slope
0.1. Derive the new speedup formula and find the best
gamma at a=0.8.

S2. Constraint change: batch is 256 (compute-bound
decode). Does speculative decoding still help? Argue
from the intensity picture and name what changes in
the cost model.

## Research critique

R1. A paper claims "3x faster generation with no
quality loss" for a new draft architecture. List five
audit questions, and for each state the answer that
would invalidate the claim.
