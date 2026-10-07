# keys-u10.md: interview bank U10 answers

Date: 2026-10-06. Closed-book reference answers with
rubrics. Keep separate from the questions file.

## Breadth answers

B1. Amortized = B/N + q. Brute force wins below
break-even N.

B2. Dense: docs*dims*bytes. PQ: docs*code_bytes.

B3. Recall@k: fraction of true top-k returned. The
floor admits only systems at or above it.

B4. RRF = sum 1/(k+rank). Ranks need no
calibration across systems.

B5. Total = sum count*cost per stage. The last
(rerank) stage dominates.

B6. Measured, derived, stated, opinion.

B7. Question, hypothesis, baselines, budget,
failure criteria.

B8. Ship iff delta >= -epsilon on the task
metric.

## Deep ladder 1 answers

L1a. Build: one-time index cost. Query: marginal
per-request cost.

L1b. 12.2 ms at 1M. Break-even 160,000 queries.

L1c. N = B/(brute - q) = 7200/0.045 = 160,000.

L1d. Amortized() as in the lab. Build O(corpus),
query O(log corpus).

L1e. ANN: 0.95+ recall, ms latency, hours build.
Brute: 1.00 recall, 50 ms, zero build.

L1f. Query volume below break-even.

L1g. Assumption: static index. Counterexample:
hourly news rebuilds reset the amortization.

L1h. Measure build and query on a corpus subset.
Accept if the formula predicts crossover within
2x.

Rubric: must solve for N. Red flag: "indexes are
always worth it." Fix: the break-even.

## Deep ladder 2 answers

L2a. Cascade: staged funnel. Gate: quality
contract. Defense: oral exam on the work.

L2b. 5 + 400 = 405 ms.

L2c. The reranker is 400/405 of the cost. Top 10:
205 ms.

L2d. Gate() as in the lab. Costs one eval run per
candidate.

L2e. Cascade: better quality, 405 ms. ANN-only:
5 ms, lower ceiling.

L2f. The ANN's top-k missed the gold doc. The
funnel, not the reranker, failed.

L2g. Assumption: the funnel keeps the gold.
Counterexample: ANN recall miss: the reranker
never sees it.

L2h. Sweep funnel width, plot recall vs latency.
Expect a knee.

Rubric: must find the dominant stage. Red flag:
blaming the reranker for funnel misses. Fix:
trace the gold doc.

## Analytical answers

A1. D: 0.031258. E: 0.022643. D wins because
fusion rewards agreement across systems over one
loud rank.

A2. Stated, derived, opinion. With a sound
table: measured. Still check: baselines current,
seeds reported, metric ungameable.

## Debug answer

D1. Bug: epsilon is 5%, not 0.5% (0.05 vs
0.005). Fix: return delta >= -0.005. Invariant:
the gate matches the documented epsilon.

## Changed-constraint answers

S1. Free rebuilds: indexes always win on latency
(build cost zero). The choice becomes pure
recall-vs-latency. Cascade: widen funnels freely
(rebuilds are free, but query latency still
binds).

S2. Gameable metric: gate on a held-out human
eval or a metric family (no single number).
Defense: lead with the gaming analysis. The
first question becomes "how did you prevent
gaming."

## Research critique answer

R1. Five audits: (1) query types? Invalidate:
only semantic queries tested. (2) BM25 tuned?
Invalidate: default parameters vs tuned dense.
(3) latency matched? Invalidate: the retriever
ran 10x slower. (4) corpus? Invalidate: one
friendly corpus. (5) significance? Invalidate:
no seeds or error bars.
