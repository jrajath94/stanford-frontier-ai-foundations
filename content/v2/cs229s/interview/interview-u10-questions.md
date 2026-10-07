# Interview bank U10: Retrieval, guests, synthesis

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Write the index amortization formula. When does
brute force win?

B2. Write the dense and PQ storage formulas.

B3. Define recall@k. How does a recall floor pick a
system?

B4. Write the RRF formula. Why do ranks beat raw
scores?

B5. Write the cascade cost formula. Which stage
dominates?

B6. Name the four evidence grades in order.

B7. Name the five proposal rubric criteria.

B8. Write the quality-gate rule.

## Deep ladder 1: retrieval economics

L1a. Define: build cost vs query cost.
L1b. Toy: build 7200 s, query 5 ms, brute force 50
ms. Compute amortized cost at 1M queries and the
break-even N.
L1c. Derive: solve B/N + q = brute for N.
L1d. Implement and complexity: write amortized()
and state the index build/query complexities.
L1e. Compare: ANN vs brute force on recall,
latency, and build.
L1f. Debug: the index never pays off. Name the
cause.
L1g. Critique: state the static-index assumption.
Construct the news-corpus counterexample.
L1h. Design: propose the break-even measurement.
Name the acceptance band.

## Deep ladder 2: from retrieval to defense

L2a. Define: cascade, gate, defense.
L2b. Toy: ANN 5 ms, rerank 20 docs at 20 ms.
Compute the cascade cost.
L2c. Derive: show the last stage dominates and
compute the tightened cost at top 10.
L2d. Implement and complexity: write gate() and
state what it costs to run.
L2e. Compare: cascade vs ANN-only on quality and
latency.
L2f. Debug: quality drops though the reranker is
great. Name the failure point.
L2g. Critique: state the funnel-keeps-gold
assumption. Construct the recall-miss
counterexample.
L2h. Design: propose the funnel-width sweep. Name
the expected knee.

## Analytical exercises

A1. Fusion: doc D ranks (3, 5), doc E ranks (1,
100), k=60. Compute both RRF scores. Explain in
one sentence why D wins.

A2. Evidence: three claims: "2x faster" (no
table), "O(T d^2)" (derivable), "SSMs are the
future". Grade each. A table then appears for the
first claim with sound baselines: what is the new
grade, and what would you still check?

## Implementation / debug task

D1. The gate below ships everything, including a
candidate 2% worse. Find the bug, fix it, and
state the invariant.

```python
def gate(delta):
    return delta > -0.05  # bug: 5% not 0.5%
```

## Changed-constraint scenarios

S1. Constraint change: rebuilds are free and
instant. What happens to the index-vs-brute-force
choice and to the cascade design?

S2. Constraint change: the quality metric is
perfectly gameable and everyone knows it. Redesign
the gate and the defense under this constraint.

## Research critique

R1. A paper claims "our retriever beats BM25 on
every query type." List five audit questions, and
for each state the answer that would invalidate
the claim.
