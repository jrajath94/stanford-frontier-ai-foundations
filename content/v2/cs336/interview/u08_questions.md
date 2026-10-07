# U08 interview bank , questions

Closed-book. Keys in `u08_key.md`. 7B reference: 84 GB total (14+14+
56). Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Why is the DP update an exact big-batch update?
B2. When does ring all-reduce beat tree?
B3. What does each ZeRO stage shard?
B4. What is loss scaling, and when is it unnecessary?
B5. What is the FSDP per-layer cycle?
B6. State the checkpoint-restart contract.

## Deep ladders (2 x 5)

L1. ZeRO memory.
- L1.1 Define the three state tensors and their bytes for 7B.
- L1.2 Toy: compute per-GPU bytes for stages 0-3 at N=8.
- L1.3 Derive the stage-3 communication cost.
- L1.4 Implement zero_bytes, state the monotonicity check.
- L1.5 Compare ZeRO-3 with tensor parallelism, debug the tiny-layer
  case, critique the even-shard assumption, propose the 13B check.

L2. Overlap.
- L2.1 Define exposed communication.
- L2.2 Toy: 8 layers, 12 ms compute, 3 ms comm. Compute both
  schedules.
- L2.3 Derive the bucket-sizing condition.
- L2.4 Implement the schedule simulator, state the bounds checks.
- L2.5 Compare with faster links, debug the serialization,
  critique the async assumption, propose the ratio sweep.

## Analytical exercises (2)

E1. 13B model, fp16 params/grads, fp32 Adam, 8x80GB GPUs. (a) Total
replicated bytes. (b) Per-GPU bytes at ZeRO-1/2/3. (c) Which stages
fit, and what is the catch at stage 3?
E2. A step takes 10 s with 3 s exposed communication. (a) What
fraction is exposed? (b) Bucketing halves the exposed part. New step
time and efficiency gain? (c) The exposed part will not shrink
further. Name two structural fixes.

## Implementation/debug task (1)

D1. This DP step trains but converges as if the batch were 1/N. Find
the bug and fix it.

```
def dp_step_bad(shard_grads):
    total = shard_grads[0]
    for g in shard_grads[1:]:
        total = total + g
    apply_update(total)   # <-- suspect
```

## Changed-constraint scenarios (2)

S1. The cluster changes from 8 to 4 GPUs weekly. Checkpoints are
per-rank shards. What must the pipeline do on every size change?
S2. One GPU runs 40 percent slower (thermal). The step is collective-
bound. What is the observed effect, and what are the two responses?

## Research critique (1)

R1. A paper reports "linear scaling to 1024 GPUs" with tokens/s only,
no efficiency numbers, no per-rank timings, and no baseline. List
three gaps and the measurement for each.
