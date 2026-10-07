# U09 interview bank , questions

Closed-book. Keys in `u09_key.md`. Reference: d=4096, L=32, 7B, bf16.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. How does tensor parallelism split an MLP?
B2. What is the pipeline bubble, and what shrinks it?
B3. What is 1F1B?
B4. State the hybrid ordering rules (TP/PP/DP).
B5. Why does inference prefer TP over PP?
B6. Name the five parallelism planning invariants.

## Deep ladders (2 x 5)

L1. Tensor parallelism.
- L1.1 Define the column/row split.
- L1.2 Toy: verify the split sums exactly, compute 67.1 MB/layer.
- L1.3 Derive why two all-reduces per layer are needed.
- L1.4 Implement the split simulator, state the exactness check.
- L1.5 Compare with ZeRO-3, debug cross-node TP, critique the
  reference shapes, propose the tp sweep.

L2. Pipeline.
- L2.1 Define stages and microbatches.
- L2.2 Toy: compute the 42.9 percent bubble.
- L2.3 Derive (p-1)/(m+p-1).
- L2.4 Implement the schedule simulator, state the total-time check.
- L2.5 Compare GPipe with 1F1B, debug the m<p case, critique
  uniformity, propose the m sweep.

## Analytical exercises (2)

E1. 70B model (140 GB fp16), 32 GPUs, node size 8. (a) Propose a
(tp,pp,dp) triple satisfying the invariants. (b) Compute per-GPU
fp16 params. (c) Compute the TP comm per step (d=8192, L=80,
microbatch b=1, T=4096, bf16).
E2. A pipeline has p=8 uneven stages (slowest 2x the mean). (a) What
sets the throughput? (b) The bubble formula says 8 percent at
m=80. Why is the real bubble worse? (c) Name two fixes.

## Implementation/debug task (1)

D1. This hybrid plan passes the product check but will fail on the
cluster. Find the bug and fix it.

```
plan = dict(tp=16, pp=2, dp=2, N=64, node=8, hbm=80)
assert plan["tp"] * plan["pp"] * plan["dp"] == plan["N"]  # passes
```

## Changed-constraint scenarios (2)

S1. The serving SLO is p99 latency, batch 1, 70B model. TP=8 is
proposed. An alternative proposes PP=8. Decide with the latency
model and name the risk of the loser.
S2. Memory allows only 2 in-flight microbatches with p=4. 1F1B wants
~4. What are the options, and what does each cost?

## Research critique (1)

R1. A paper reports hybrid-parallel scaling with speedup numbers but
no (tp,pp,dp) triple, no link tiers, and no per-axis breakdown.
List three gaps and the measurement for each.
