# Interview bank U09: Parallelism, clusters, scheduling

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Name the four parallelism splits and what each
shards.

B2. Write the ring all-reduce formula. Name the two
passes.

B3. State the interconnect placement rule.

B4. Write the synchronous step-time rule and the
straggler tax.

B5. Write the pipeline bubble formula.

B6. Name the three ZeRO stages and what each shards.

B7. Write the checkpoint overhead formula.

B8. Write the cluster MTBF formula.

## Deep ladder 1: the DP tax

L1a. Define: what does data parallel synchronize?
L1b. Toy: S=20 GB, n=8, BW=100 GB/s. Compute the
ring all-reduce time.
L1c. Derive: show the time is nearly independent
of n in bytes but not in latency.
L1d. Implement and complexity: write ring_ar() and
state its scaling.
L1e. Compare: ring vs tree all-reduce. When does
each win?
L1f. Debug: sync takes 3x the formula. Name two
causes.
L1g. Critique: state the no-congestion assumption.
Construct the incast counterexample.
L1h. Design: propose the time-vs-S measurement.
Name the expected slope.

## Deep ladder 2: pipeline reality

L2a. Define: stages, microbatches, bubble.
L2b. Toy: p=4, m=8. Compute the bubble fraction.
L2c. Derive: total steps m+p-1 and the bubble
ratio.
L2d. Implement and complexity: write bubble() and
state what 1F1B changes.
L2e. Compare: more microbatches vs interleaved
pipeline.
L2f. Debug: measured bubble is 2x the formula.
Name the cause.
L2g. Critique: state the equal-stages assumption.
Construct the heavy-embedding counterexample.
L2h. Design: propose the idle-vs-m measurement.
Name the expected fit.

## Analytical exercises

A1. Placement: all-reduce 20 GB at n=8, intra-node
600 GB/s vs split 50 GB/s. Compute both times and
the ratio. A TP-8 job is split across the nodes:
what does each matmul sync cost, and what is the
verdict?

A2. Fleet: per-GPU MTBF 3 years, 256 GPUs vs 4096
GPUs. Compute both cluster MTBFs. A 30-day run on
each: expected failures? What does the 4096-GPU
number imply for checkpoint design?

## Implementation / debug task

D1. The sharding helper below is meant to be ZeRO-3
but a 10B-parameter run still OOMs on 40 GB GPUs.
Find the bug, fix it, and state the invariant.

```python
P, n = 10e9, 8
per_gpu = 16 * P  # bug: forgot the split
```

## Changed-constraint scenarios

S1. Constraint change: interconnect bandwidth is
infinite and free. Which parallelism taxes vanish,
which remain, and what is the best split now?

S2. Constraint change: GPUs never fail and
preemption is impossible. What changes in
checkpointing, scheduling, and the spot-vs-
on-demand choice?

## Research critique

R1. A paper claims "our scheduler achieves 95% MFU
on 4096 GPUs." List five audit questions, and for
each state the answer that would invalidate the
claim.
