# U17 interview bank , questions

Closed-book. Keys in `u17_key.md`. All numbers are synthetic toys
from `visuals/compute_u17.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. How does an image become tokens?
B2. What is an interleaved stream?
B3. Name the three system interfaces and their units.
B4. Why separate rollout and training fleets?
B5. Write the artifact lineage chain rule.
B6. State Amdahl's bound in one sentence.

## Deep ladders (2 x 5)

L1. Async pipelines.
- L1.1 Define utilization and backlog.
- L1.2 Toy: 120/100 per min gives 83.3%, 1200/hr backlog.
- L1.3 Derive backlog=(r-c)*t.
- L1.4 Implement `backlog`, state the r=c check.
- L1.5 Compare with lockstep, debug the trainer-stall case,
  critique constant rates, propose the stall simulation.

L2. Bottlenecks.
- L2.1 Define the stage table.
- L2.2 Toy: 143 ms step, backward 38.5%, 2x gives 19.2%.
- L2.3 Derive new total = total - share + share/speedup.
- L2.4 Implement `amdahl`, state the infinite-speedup check.
- L2.5 Compare profile-driven with gut-feel, debug the moving
  bottleneck, critique the toy, propose the sequential-speedup
  test.

## Analytical exercises (2)

E1. Stage times: data 12, forward 40, backward 55, all-reduce 25,
optimizer 8, checkpoint 3 ms. (a) Identify the bottleneck and its
share. (b) Overlap halves the all-reduce's exposed time. New step
time? (c) After that fix, what is the next bottleneck?
E2. 8 images at 256 tokens each + 512 text. (a) Total tokens and
image share. (b) Context limit 2048: how many images fit with 512
text reserved? (c) Name two ways to fit more.

## Implementation/debug task (1)

D1. This lineage chain silently accepts a double-run. Find the
bug.

```
def record(stage, inputs):
    art_id = sha256(str(inputs))  # bug: no parent chain, no run id
    return art_id
```

## Changed-constraint scenarios (2)

S1. The trainer stalls for an hour, the queue is unbounded.
Predict the outcome and set the policy.
S2. All evidence comes from CPU toys. The team must decide a 70B
run. State what the toys do and do not support.

## Research critique (1)

R1. A systems paper reports step-time wins with no stage
breakdown, no hardware named, and no baseline config. List three
gaps and the disclosure for each.
