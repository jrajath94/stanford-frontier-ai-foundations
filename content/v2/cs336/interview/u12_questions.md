# U12 interview bank , questions

Closed-book. Keys in `u12_key.md`. All numbers are synthetic toys
from `visuals/compute_u12.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Convert loss 2.3 to perplexity and state its meaning.
B2. What are the two contamination detection tools?
B3. Why is bits per byte fairer than per-token loss across
tokenizers?
B4. State the internal vs external evaluation rule.
B5. What does Cohen's kappa correct for?
B6. Write the cost-per-correct-task formula.

## Deep ladders (2 x 5)

L1. Uncertainty.
- L1.1 Define SE for accuracy.
- L1.2 Toy: n=500, p=0.72 gives [0.681, 0.759].
- L1.3 Derive the 1.96 factor's origin.
- L1.4 Implement the CI, state the 5-seed consistency check.
- L1.5 Compare with bootstrap, debug the correlated-seed case,
  critique the normal approximation at small n, propose the
  20-seed experiment.

L2. Contamination.
- L2.1 Define contamination and the canary.
- L2.2 Toy: 100% overlap on planted items, canary recall 1.00.
- L2.3 Justify n-gram matching as detection.
- L2.4 Sketch the overlap check, state the planted-item check.
- L2.5 Compare exact with fuzzy matching, debug the paraphrase
  case, critique the threshold, propose the paraphrase test.

## Analytical exercises (2)

E1. Slice A: n=40, acc=0.80. Slice B: n=60, acc=0.65. (a) Compute
both 95% half-widths. (b) Is the gap significant? Show the math.
(c) How many items per slice to halve the half-widths?
E2. Model X: 800 tokens/task, acc 0.72, $0.002/1k tokens. Model Y:
2400 tokens/task, acc 0.85. (a) Compute cost per correct task for
both. (b) A penalty of $0.01 per wrong task is added. Recompute
and re-rank. (c) What does the flip teach?

## Implementation/debug task (1)

D1. This kappa implementation is wrong. Find the bug.

```
def kappa(a, b):
    agree = (a == b).mean()
    return agree  # bug: no chance correction
```

## Changed-constraint scenarios (2)

S1. The eval set is released after training started. List the
containment steps in order.
S2. The judge's kappa is 0.4 on the calibration set. The team
wants to rank 5 models with it. Decide and defend.

## Research critique (1)

R1. A paper reports a 2-point benchmark win with no CIs, no seed
information, and no decoding settings. List three gaps and the
fix for each.
