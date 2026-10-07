# keys-u08.md: interview bank U08 answers

Date: 2026-10-06. Closed-book reference answers with
rubrics. Keep separate from the questions file.

## Breadth answers

B1. The batch re-forms every iteration, admitting
arrivals at once. It removes batch-formation
waiting.

B2. Admission picks the running set. Backpressure
tells upstream to slow down. With neither, the
queue grows until memory dies.

B3. Prefill: parallel, compute-bound. Decode:
serial, memory-bound. Chunked prefill splits big
prefills so decodes stop stalling.

B4. 2 * L * n * bytes per float. LRU evicts the
longest-unused entry.

B5. P99: latency 99% beat. Little: concurrency =
throughput * mean latency. Means hide the tail.

B6. G = softmax(W_g x), top-k renormalized,
output = sum w_i E_i(x).

B7. Capacity = (tokens*k/experts)*factor. Past
capacity, tokens drop (skip the expert).

B8. Ring all-reduce: 2*(n-1)/n * S per GPU.
All-to-all: ~S each way per GPU.

## Deep ladder 1 answers

L1a. Queue wait + prefill + decode tokens +
scheduling stalls.

L1b. Continuous mean latency 1.0 s, static 1.5 s.

L1c. Both finish the last request at 4.0 s at the
same token rate. The 0.5 s is pure formation
wait.

L1d. The comparison is arithmetic on arrival
times. Per-iteration cost: O(batch) to pick the
next set.

L1e. Bursty: continuous wins. Simultaneous: static
wins in the toy (no wait to remove).

L1f. Missing chunked prefill: whole prefills
stall the decodes.

L1g. Assumption: insertion is free. Counterexample:
a huge prefill stalls every running decode for
its full duration.

L1h. Replay bursty traces through both. Expect
the gap to grow with burstiness.

Rubric: must separate waiting from speed. Red
flag: "continuous batching is faster per token."
Fix: the last-completion tie.

## Deep ladder 2 answers

L2a. Routing picks experts per token. Capacity
caps tokens per expert. Imbalance is max/mean
load.

L2b. Experts 0 and 2, weights 0.646 and 0.354.

L2c. Capacity 20. Expert 0 drops 20 of 40 (50%
there, 15.6% global).

L2d. Capacity() as in the lab. Buffer memory
scales with the factor.

L2e. Token-choice drops overflow (needs balance
loss). Expert-choice picks top tokens per expert:
zero drops by construction, but unpopular tokens
may never train.

L2f. Heavy drops at that expert: it trains on a
biased subset. Check per-expert drop rates.

L2g. Assumption: the balance loss fixes skew.
Over-weighted: uniform routing, no
specialization. Under-weighted: collapse.

L2h. Sweep the factor. Plot drop rate and eval.
Expect eval flat until ~5% drops, then falling.

Rubric: must compute drops, not hand-wave. Red
flag: "MoE is free capacity." Fix: the dispatch
and balance taxes.

## Analytical answers

A1. Mean 1.2, p50 1.0, p99 3.0. Concurrency 60.
Tail fixed: mean 1.0, p50 1.0, p99 1.0,
concurrency 50.

A2. One way 1 MiB, round trip 2 MiB, 32 layers
64 MiB per step. At 50 steps/s: 3.2 GB/s
dispatch per GPU.

## Debug answer

D1. Bug class: missing load-balance term (gate
collapse). Fix: add an auxiliary balance loss
(experts * sum f_i p_i) to the training
objective. Invariant: every expert receives a
nontrivial token share each epoch.

## Changed-constraint answers

S1. With infinite bandwidth, dispatch is free but
MoE still pays: gate compute, balance loss,
capacity drops, and routing quality risk. Dense
wins on simplicity. MoE wins only if the extra
parameters buy quality per FLOP.

S2. Hard SLO: bounded queue with rejection past
the wait budget, prefill chunking mandatory,
priority to oldest-deadline, eviction by
earliest-deadline-first instead of LRU, shed
load early (fail fast beats SLO miss).

## Research critique answer

R1. Five audits: (1) matched quality? Invalidate:
the dense baseline is weaker. (2) dispatch
counted? Invalidate: serving cost omits the
all-to-all. (3) utilization? Invalidate: the MoE
ran at higher batch. (4) tail SLOs? Invalidate:
p99 is worse. (5) training cost? Invalidate:
"cheaper" counts inference only.
