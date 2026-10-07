# U14 interview bank , questions

Closed-book. Keys in `u14_key.md`. All numbers are synthetic toys
from `visuals/compute_u14.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Why dedup before filtering?
B2. State the Bloom FPR formula and the toy optimum.
B3. What three stages make near-dup detection?
B4. Why is document-level dedup not enough for train-test leaks?
B5. Write the reweighting multiplier rule.
B6. State the memorization/repetition tradeoff.

## Deep ladders (2 x 5)

L1. Bloom filters.
- L1.1 Define the structure and the guarantee.
- L1.2 Toy: k=6, FPR 0.0216 at 8 bits/item.
- L1.3 Derive (1-e^{-kn/m})^k.
- L1.4 Implement add/contains, state the no-false-negative
  check.
- L1.5 Compare with a hash set, debug the adversarial-input
  case, critique uniformity, propose the empirical FPR test.

L2. MinHash+LSH.
- L2.1 Define shingles, signatures, bands.
- L2.2 Toy: Jaccard 0.113 vs 0.125. S-curve points.
- L2.3 Derive 1-(1-s^r)^b.
- L2.4 Sketch the pipeline, state the S-curve midpoint check.
- L2.5 Compare with exact all-pairs, debug the boilerplate
  case, critique the synthetic sets, propose the signature-
  length sweep.

## Analytical exercises (2)

E1. n=1e9 docs, budget 1 GB for the Bloom filter. (a) Compute
bits per item. (b) Choose k and compute the FPR. (c) The team
wants FPR < 0.01. What changes?
E2. Mixture toy: L(w)=w*2.5+(1-w)*3.0+0.4*w*(1-w). (a) Find the
argmin on [0,1]. (b) The interaction coefficient rises to 1.0.
Recompute. (c) Interpret the shift.

## Implementation/debug task (1)

D1. This dedup keeps the wrong copy. Find the bug.

```
seen = {}
for doc in docs:
    h = hash(doc.text)
    if h not in seen:
        seen[h] = doc
    # keeps first, but later code uses seen[h] after overwriting below
    seen[h] = doc  # bug: overwrites, keeps last
```

## Changed-constraint scenarios (2)

S1. A 100x-repeated domain must stay in the mix for product
reasons. Set the repetition policy and the monitoring.
S2. The eval set is paraphrased in training data. The n-gram
check reads clean. What now?

## Research critique (1)

R1. A paper claims dedup gains with no before/after token
counts, no near-dup method named, and no train-test leak check.
List three gaps and the measurement for each.
