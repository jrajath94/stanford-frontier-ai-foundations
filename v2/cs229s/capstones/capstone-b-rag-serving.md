# Capstone B: RAG assistant serving, applied/FDE design

Date: 2026-10-06. Status: model EXECUTED. All numbers
HYPOTHETICAL (labeled, not measured). Script:
`capstones/rag_serving_model.py`. Figure:
`capstones/rag_cost_latency.png` (metadata stripped).

## Discovery

Support team answers repeat questions from a 1M-doc
knowledge base. Today: keyword search plus human
triage. Pain: slow answers, inconsistent quality.
Users: support agents (internal). Volume: bursty,
business hours.

## Workflow baseline

Query -> keyword search (50 ms, hypothetical) ->
agent reads 5 docs -> writes answer. Median handle
time 6 minutes (stated by the team, not measured
here).

## Objectives

- p50 answer latency under 4 s, p99 under 10 s
  (hypothetical SLOs).
- Answer grounded in retrieved docs (citations).
- Cost under $0.005 per request (hypothetical
  budget).

## Constraints

- 1M docs, hourly updates (index rebuild cost
  recurs. U10 C01).
- No PII in logs (trust boundary).
- GPU budget: 4 GPUs (hypothetical).

## Trust boundaries

User query -> retrieval (internal) -> LLM (internal)
-> agent screen. The LLM never calls external tools.
Logs strip PII before storage. Prompt injection
from docs is a known risk: retrieved text is
untrusted input (treat as data, not instruction).

## Alternatives considered

1. Keyword only: cheap, misses semantic queries.
2. Dense only: misses exact part numbers.
3. Hybrid + rerank (chosen): best quality per the
   U10 tradeoff. The reranker dominates latency.

## Acceptance gates

- Retrieval recall@20 >= 0.90 on a labeled eval
  set (hypothetical bar).
- Quality gate: task score delta >= -0.5% vs the
  human baseline (U10 C11).
- Latency: p50 <= 4 s, p99 <= 10 s in load test.
- Cost: <= $0.005/request at the hypothetical
  price.

## Cost/latency model (hypothetical)

Executed in `rag_serving_model.py`:

- Retrieval: 405 ms (ANN 5 ms + 20 docs x 20 ms
  rerank).
- Prefill 512 tokens: 256 ms. Decode 128 tokens:
  2560 ms.
- Total: 3221 ms p50-ish. Fits the 4 s SLO with
  779 ms of headroom. P99 needs the tail analysis
  (U08 C05).
- Cost: $0.0011/request, $1.67/MTok at the
  hypothetical $3/GPU-hour. Fits the budget with
  4.5x headroom.

## Rollout

Shadow mode (2 weeks): serve alongside humans,
log only. Then 10% of agents, then 50%, then
all. Each stage needs the gates green.

## Monitoring

Latency p50/p99, recall@20 on sampled queries,
quality-gate delta weekly, cost per request
daily, PII-leak scanner on logs.

## Rollback

Feature flag per stage. Rollback trigger: gate
red for 1 hour, or any PII incident. Rollback
restores keyword search in minutes (the old path
stays deployed).

## Ownership and handoff

ML platform owns the serving stack. The support
team owns the eval set and the quality bar. I
hand off: the model script, the eval set, the
runbook, and this document. The team checks and
ships every stage. Nothing here auto-deploys.

## Stakeholder defense

Five questions (U10 C12): claim (4 s p50 at
$0.0011/req, hypothetical inputs), evidence (the
executed model plus the load test to come),
baseline (6-minute human handle time, stated),
failure (reranker tail, hourly rebuild cost),
cost ($0.0011 vs $0.005 budget). Honest gaps:
the p99 is unmodeled, the price is hypothetical,
the eval set does not exist yet.
