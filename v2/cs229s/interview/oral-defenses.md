# Oral defenses: 10 deep ladders x 8 follow-ups

Date: 2026-10-06. Closed-book oral format. Keys in
`keys-oral.md`. Provenance: original practice. Each
ladder runs: define -> toy -> derive ->
implement/complexity -> compare -> debug -> critique
-> design.

## O1. From the attention bottleneck to FlashAttention

O1a. Define the attention bottleneck in one
sentence.
O1b. Toy: T=8192, d=128, one head. Compute score
FLOPs and score memory.
O1c. Derive why the naive implementation is
memory-bound, not compute-bound.
O1d. Implement online softmax in words and state
its memory complexity.
O1e. Compare FlashAttention to a sparse
approximation: when does exactness win?
O1f. Debug: your tiled attention is 2x slower
than naive at T=512. Why?
O1g. Critique: state the assumption that makes
tiling legal.
O1h. Design the experiment that proves the win on
your GPU.

## O2. From scaling laws to the emergence audit

O2a. Define a scaling law.
O2b. Toy: fit the two-point law and predict the
third scale.
O2c. Derive the decade slope from the two points.
O2d. Implement the fit and state what each point
costs.
O2e. Compare the two-point line to the floor form.
O2f. Debug: the big run beats prediction by 0.2.
Name two causes.
O2g. Critique: construct the data-mix
counterexample to the fixed-recipe assumption.
O2h. Design the emergence audit for a vendor's
"emergent ability" claim.

## O3. Continuous batching economics

O3a. Define iteration-level scheduling.
O3b. Toy: compute continuous vs static mean
latency on the 4-request trace.
O3c. Derive why the 0.5 s gap is wait time, not
speed.
O3d. Implement the scheduler comparison and state
the per-iteration cost.
O3e. Compare continuous vs static under bursty vs
simultaneous arrivals.
O3f. Debug: decodes stall on every long prompt.
Name the missing mechanism.
O3g. Critique: state the free-insertion
assumption and break it.
O3h. Design the capacity plan: how many GPUs for
a bursty trace at a p99 SLO?

## O4. MoE routing and balance

O4a. Define top-k expert routing.
O4b. Toy: route one token through the 8-expert
gate. Compute the weights.
O4c. Derive the capacity formula and the drop
count when expert 0 gets 40 tokens.
O4d. Implement capacity() and state what sets
buffer memory.
O4e. Compare token-choice vs expert-choice
routing.
O4f. Debug: one expert's loss will not fall.
Trace the mechanism.
O4g. Critique: the balance loss fixes skew, but
at what price?
O4h. Design the production MoE: pick k, factor,
and the drop policy for a latency SLO.

## O5. The SSM triple identity

O5a. Define the three views.
O5b. Toy: run the recurrence, unroll the kernel,
combine the tree. Give the numbers.
O5c. Derive K_k = C A^k B from the recurrence.
O5d. Implement the scan combine and state work
vs depth.
O5e. Compare convolution view vs scan: which
assumption does each need?
O5f. Debug: tree total disagrees with the loop.
What failed?
O5g. Critique: the FFT conv-view "always wins"
claim. Construct the materialization
counterexample.
O5h. Design the measurement that finds the true
crossover on your hardware.

## O6. Pipeline bubbles

O6a. Define stages, microbatches, bubble.
O6b. Toy: p=4, m=8. Compute the bubble.
O6c. Derive the m+p-1 total.
O6d. Implement bubble() and state what 1F1B
changes.
O6e. Compare more microbatches vs interleaved
schedules.
O6f. Debug: measured bubble is 2x the formula.
Name the cause.
O6g. Critique: the equal-stages assumption. 
construct the heavy-stage counterexample.
O6h. Design a pipeline for 8 uneven stages: how
do you split and verify?

## O7. ZeRO vs collectives

O7a. Define the three ZeRO stages.
O7b. Toy: 10B params, 8 GPUs. Compute ZeRO-3 vs
DP bytes per GPU.
O7c. Derive the 14-bytes-per-param figure.
O7d. Implement the per-GPU formula and state the
communication added at stage 3.
O7e. Compare ZeRO-3 to plain DP: memory vs step
time.
O7f. Debug: ZeRO-3 is slower than ZeRO-2 on
small layers. Why?
O7g. Critique: the even-shard assumption. When
does it hurt?
O7h. Design the memory plan for a 70B run on
your cluster.

## O8. Quantization granularity

O8a. Define s, z, and the quantization map.
O8b. Toy: quantize 0.5 with min=-1, max=1, 4
bits.
O8c. Derive the s/2 in-range error bound.
O8d. Implement quantize/dequantize and state the
one-time cost.
O8e. Compare per-tensor vs per-channel on mixed
ranges.
O8f. Debug: small values collapse to one level.
Name the cause and first fix.
O8g. Critique: the range-covers-data assumption. 
construct the outlier counterexample.
O8h. Design the ship/no-ship experiment for int8
serving.

## O9. The retrieval cascade

O9a. Define the cascade stages.
O9b. Toy: compute the cascade cost (ANN + 20x20
ms rerank).
O9c. Derive why the last stage dominates.
O9d. Implement the cost model and state the
recall risk of narrowing.
O9e. Compare cascade vs ANN-only.
O9f. Debug: quality drops though the reranker is
great. Where is the gold?
O9g. Critique: the funnel-keeps-gold assumption. 
construct the recall-miss counterexample.
O9h. Design the RAG serving stack under a 4 s
SLO: pick the funnel, the gates, and the
rollback.

## O10. Failure math at scale

O10a. Define cluster MTBF.
O10b. Toy: 256 GPUs, 3-year per-GPU MTBF.
Compute.
O10c. Derive the divide-by-n rule and its
assumption.
O10d. Implement the checkpoint overhead formula
and price k=100.
O10e. Compare checkpoint-restart vs spare GPUs.
O10f. Debug: failures cluster in time. What
broke?
O10g. Critique: the independence assumption. 
construct the correlated counterexample.
O10h. Design the resilience plan for a 4096-GPU
month-long run.
