# keys-u09.md: interview bank U09 answers

Date: 2026-10-06. Closed-book reference answers with
rubrics. Keep separate from the questions file.

## Breadth answers

B1. Data: data split, model copied. Tensor: matmuls
split. Pipeline: layers split into stages. Expert:
experts split, tokens move.

B2. T = 2*(n-1)/n * S/BW. Reduce-scatter then
all-gather.

B3. Keep tensor-parallel groups inside the node. 
cross nodes with data or pipeline parallel.

B4. Step = max(worker times). Tax = (max -
median)/median.

B5. Bubble = (p-1)/(m+p-1).

B6. ZeRO-1: optimizer states. ZeRO-2: + gradients.
ZeRO-3: + parameters.

B7. Overhead = (N/k) * save_cost.

B8. Cluster MTBF = per-GPU MTBF / GPU count.

## Deep ladder 1 answers

L1a. Gradients (and, under sharding, parameters):
every step averages the copies.

L1b. 2*7/8 * 20/100 = 0.35 s.

L1c. Bytes: 2*(n-1)/n -> 2 as n grows (nearly
flat). Latency: 2*(n-1) hops grow with n.

L1d. Ring_ar as in the lab. O(S) bytes per GPU,
O(n) hop latency.

L1e. Ring for big S (bandwidth-optimal). Tree for
small S and many GPUs (latency-optimal).

L1f. Incast congestion, or a degraded link
throttling the ring.

L1g. Assumption: full duplex, no congestion.
Counterexample: all GPUs blast one peer: incast
collapse, multiples of the formula.

L1h. Measure time vs S. Expect slope 1.75/BW at
n=8.

Rubric: must separate byte scaling from latency
scaling. Red flag: "all-reduce is free at
scale." Fix: the hop count.

## Deep ladder 2 answers

L2a. Stages: layer groups. Microbatches: the
batch split into pipe units. Bubble: idle stage
time.

L2b. 3/11 = 27.3%.

L2c. Fill takes p-1 steps, drain p-1. Total
m+p-1, useful m.

L2d. Bubble() as in the lab. 1F1B keeps the same
bubble with less activation memory.

L2e. More microbatches when memory allows. 
interleaving when it does not.

L2f. Uneven stages: one stage (e.g. embeddings)
runs longer than the rest.

L2g. Assumption: equal stage times. Counterexample:
the embedding stage at 2x: the formula
understates idle time.

L2h. Measure idle fraction vs m. Expect the
(p-1)/(m+p-1) fit.

Rubric: must derive m+p-1. Red flag: "more
microbatches are always free." Fix: the memory
cost.

## Analytical answers

A1. Intra-node 0.058 s, split 0.70 s, ratio 12x.
Each TP sync costs 0.70 s: the job is dead. Move
TP inside the node.

A2. 256 GPUs: 102.7 h = 4.28 days, ~7 failures
per 30 days. 4096 GPUs: 6.4 h, ~112 failures per
30 days. At 4096, checkpoint/restart must be
fast, frequent, and automatic: it dominates the
system design.

## Debug answer

D1. Bug: forgot to divide by n (shards, not
copies). Fix: per_gpu = 16 * P / n. Invariant:
per-GPU bytes fall as 1/n under ZeRO-3.

## Changed-constraint answers

S1. Infinite bandwidth kills the collective tax
(TP all-reduce, DP sync, MoE dispatch). Bubbles,
stragglers, and memory caps remain. Best split:
TP everywhere (no comm tax) plus PP for memory.

S2. No failures: checkpointing becomes optional
(keep a final save only). Scheduling drops
preemption handling. Spot vs on-demand: spot
always wins (discount with zero rework).

## Research critique answer

R1. Five audits: (1) MFU definition? Invalidate:
counts recompute as useful FLOPs. (2) sustained?
Invalidate: a 10-minute burst, not a run. (3)
all ranks? Invalidate: stragglers excluded from
the average. (4) matched baseline? Invalidate: no
comparable system measured. (5) failures?
Invalidate: the window had zero faults by luck.
