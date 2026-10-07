# U18 interview bank , questions

Closed-book. Keys in `u18_key.md`. All numbers are synthetic toys
from `visuals/compute_u18.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. What is known and unknown about the two guest lectures?
B2. State the anti-inference rule in one sentence.
B3. What makes a source "obtained"?
B4. Name the five fields of a failure-diary entry.
B5. Write the replication pass rule.
B6. Recite the 8-step oral defense ladder.

## Deep ladders (2 x 5)

L1. Source honesty.
- L1.1 Define the honesty protocol's three sentences.
- L1.2 Toy: G1/G2 records with 0 artifacts inspected.
- L1.3 Justify separating reported from inspected facts.
- L1.4 Write a source record, state the reconciliation check.
- L1.5 Compare with reputation-based guessing, handle a
  surfacing artifact, critique the schedule's authority, propose
  the quarterly re-run.

L2. Ablation design.
- L2.1 Define the three designs.
- L2.2 Toy: 16/8/5 runs for 4 factors.
- L2.3 Derive 2^4, 2^{4-1}, 1+4.
- L2.4 Implement `design_runs`, state the aliasing note.
- L2.5 Compare the designs, debug the aliased interaction,
  critique the toy, propose the interaction simulation.

## Analytical exercises (2)

E1. 5-seed losses: [2.100, 2.109, 2.092, 2.073, 2.086]. (a)
Compute mean and 1.96*SE. (b) A rival reports 2.085 on one seed.
Significant? (c) How many seeds to halve the band?
E2. Ledger: 480 GPU-hr at $2.50, 1200 GB storage. (a) Total
compute cost. (b) Failures were 120 GPU-hr: what fraction of the
budget? (c) The next project is 10x with the same failure rate.
Budget it.

## Implementation/debug task (1)

D1. This reconciliation misses a hidden session. Find the bug.

```
def reconcile(counts):
    return sum(counts.values()) == 19  # bug: count-only, no mapping check
```

## Changed-constraint scenarios (2)

S1. Slides for G1 surface but no recording. Update the records
and state what can now be taught.
S2. Two of three replication metrics pass. The team wants to
claim replication. Decide.

## Research critique (1)

R1. A capstone report shows a trained model with no manifest, no
ablations, no failure diary, and no scale labels. List four gaps
(the quota is 1 critique, this one needs four) and the artifact
for each.
