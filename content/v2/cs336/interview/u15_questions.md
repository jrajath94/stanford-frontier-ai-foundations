# U15 interview bank , questions

Closed-book. Keys in `u15_key.md`. All numbers are synthetic toys
from `visuals/compute_u15.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Name the three training stages and what defines each.
B2. What does raising RoPE theta do, and what else is needed?
B3. Why must train and serve templates match byte-exactly?
B4. Write the SFT loss mask rule.
B5. State the LoRA parameter formula.
B6. What four items make a full checkpoint state?

## Deep ladders (2 x 5)

L1. Packing.
- L1.1 Define padding waste.
- L1.2 Toy: 47.6% pad vs 22.0% packed, 344 bins.
- L1.3 Justify the boundary attention mask.
- L1.4 Implement greedy `pack`, state the no-overflow check.
- L1.5 Compare with padding, debug the mask bug, critique
  greedy, propose the canary test.

L2. Forgetting.
- L2.1 Define the measurement protocol.
- L2.2 Toy: mean +0.15 nats over 6 checkpoints.
- L2.3 Justify per-checkpoint base evals.
- L2.4 Implement the delta measure, state the replay check.
- L2.5 Compare the mitigations, debug the silent-capability
  case, critique the toy, propose the replay-fraction sweep.

## Analytical exercises (2)

E1. LoRA r=16, d=4096, adapting q,v in 32 layers. (a) Compute
trainable params. (b) Full fine-tuning of q,v: how many? (c) The
task needs a new language. Recommend and defend.
E2. 250k SFT pairs, 3 epochs, 1% bad. (a) Compute bad exposures.
(b) The bad rate is cut to 0.1% at 3x the curation cost. Worth
it? Show the tradeoff.

## Implementation/debug task (1)

D1. This mask leaks the system prompt. Find the bug.

```
mask = [1 if role in ("assistant", "system") else 0 for role in roles]
# bug: system tokens carry loss
```

## Changed-constraint scenarios (2)

S1. One 100k-token document sits among 1k-token ones. Pack or
isolate? Decide with the waste math.
S2. The base model moves from v1 to v2. 10 LoRA adapters were
trained on v1. State the compatibility verdict and the check.

## Research critique (1)

R1. An SFT paper reports wins with no data card, no mask
description, and no base-capability regression check. List three
gaps and the disclosure for each.
