# Interview bank U08: Serving and sparse MoE

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Define continuous batching. What wait does it
remove?

B2. Define admission control and backpressure. What
happens with neither?

B3. Contrast prefill and decode as scheduling
classes. What does chunked prefill fix?

B4. Write the KV bytes-per-token formula. What does
LRU evict?

B5. Define p99 and state Little's law. Why do means
lie?

B6. Write the top-k expert routing formula.

B7. Write the expert capacity formula. What happens
past capacity?

B8. Write the all-reduce and all-to-all cost
formulas.

## Deep ladder 1: serving economics

L1a. Define: what sets a request's latency?
L1b. Toy: 4 requests at t=0,1,2,3, 10 tokens at
0.1 s/token. Compute continuous vs static mean
latency.
L1c. Derive: show the 0.5 s gap is formation
waiting, not speed.
L1d. Implement and complexity: write the scheduler
comparison and state the per-iteration cost.
L1e. Compare: continuous vs static under bursty vs
simultaneous arrivals.
L1f. Debug: decodes stall 0.25 s on every new
arrival. Name the missing mechanism.
L1g. Critique: state the free-insertion assumption.
Construct the huge-prefill counterexample.
L1h. Design: propose the burst-replay simulation.
Name the expected gap behavior.

## Deep ladder 2: MoE reality

L2a. Define: routing, capacity, imbalance.
L2b. Toy: logits [2.1,0.3,1.5,-0.2,0.8,1.1,-1.0,0.1],
top-2. Compute the routing weights.
L2c. Derive: capacity at 64 tokens, 8 experts,
top-2, factor 1.25. Drops when expert 0 gets 40.
L2d. Implement and complexity: write capacity()
and state what sets buffer memory.
L2e. Compare: token-choice vs expert-choice
routing on drops and balance.
L2f. Debug: one expert's loss will not fall. Name
the mechanism and the check.
L2g. Critique: state the balance-loss assumption.
Construct the over/under-weighted
counterexamples.
L2h. Design: propose the capacity-factor sweep.
Name the expected knee.

## Analytical exercises

A1. Tail: 90 requests at 1.0 s, 10 at 3.0 s.
Compute mean, p50, p99. At 50 req/s, what
concurrency does Little's law give? The tail is
fixed (all 1.0 s): recompute all four numbers.

A2. Dispatch: 64 tokens, top-2, d=4096, fp16, 32
MoE layers. Compute one-way, round-trip, and
32-layer bytes per step. At 50 steps/s, what is
the dispatch bandwidth per GPU?

## Implementation / debug task

D1. The router below trains, but 7 of 8 experts
never receive tokens after the first epoch. Find
the bug class, fix it, and state the invariant.

```python
def route(logits, k=2):
    top = sorted(range(len(logits)),
                 key=lambda i: logits[i],
                 reverse=True)[:k]
    return top  # bug: no balance term anywhere
```

## Changed-constraint scenarios

S1. Constraint change: interconnect bandwidth is
infinite. Does MoE still beat dense? Rebuild the
comparison on FLOPs, memory, and quality alone.

S2. Constraint change: every request has a hard
10 s SLO and arrivals are adversarial. Redesign
admission, scheduling, and eviction under this
constraint.

## Research critique

R1. A paper claims "our MoE serves 8x cheaper
than a dense model of equal quality." List five
audit questions, and for each state the answer
that would invalidate the claim.
