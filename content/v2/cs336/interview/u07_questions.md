# U07 interview bank , questions

Closed-book. Keys in `u07_key.md`. Quotas: 6 breadth, 2 deep ladders of
5, 2 analytical, 1 implementation/debug, 2 changed-constraint, 1
research-critique.

## Breadth (6)

B1. What is a tile, and what sets the tile size?
B2. How does the online softmax avoid a first pass for the max?
B3. What does FlashAttention store for the backward pass?
B4. What is kernel fusion, and when is it legal?
B5. In Triton, what does the program id map to?
B6. Name four kernel testing invariants for attention.

## Deep ladders (2 x 5)

L1. IO-aware forward.
- L1.1 Define the HBM traffic of naive attention.
- L1.2 Toy: compute the 537 MB vs 25.2 MB traffic.
- L1.3 Derive the online update with the V accumulation folded in.
- L1.4 Implement the tiled forward, state the match check.
- L1.5 Compare with linear attention, debug the missing division,
  critique the fp32 assumption, propose the causal check.

L2. Backward via recomputation.
- L2.1 Define what is stored and what is recomputed.
- L2.2 Toy: write the dV line.
- L2.3 Derive dS = P*(dP - rowsum(dP*P)).
- L2.4 Implement the tiled backward, state the finite-diff check.
- L2.5 Compare with storing scores, debug the fp16 statistics,
  critique the cheap-recompute assumption, propose the dK check.

## Analytical exercises (2)

E1. SRAM budget 100 KB, dh=128, bf16, kernel needs Q,K,V,O plus
statistics (count as one extra tile). (a) Compute Bc. (b) T=8192:
how many blocks? (c) The device doubles SRAM. Recompute and state
what does not change.
E2. A fused dropout+add+norm kernel processes (B=4,T=2048,d=1024)
fp16. (a) Compute separate vs fused traffic. (b) A reduction is
added to the chain. What breaks, and what is the fix?

## Implementation/debug task (1)

D1. This tiled softmax is wrong but passes a casual glance. Find the
bug and fix it.

```
m, l, acc = -inf, 0.0, zeros
for block in blocks:
    mb = block.max()
    m_new = max(m, mb)
    l = l + exp(block - m_new).sum()      # <-- suspect
    acc = acc + exp(block - m_new) @ Vb   # <-- suspect
    m = m_new
out = acc / l
```

## Changed-constraint scenarios (2)

S1. T grows to 1M tokens. The tiled forward still works. Name the
two things that break first (inner-loop passes, fp16 statistics)
and the fix for each.
S2. The mask is data-dependent (per input). Block-sparse skipping
was planned. What breaks, and what are the two options?

## Research critique (1)

R1. A paper claims a fused kernel is "4x faster" with no baseline
named, no shapes, and no traffic analysis. List three gaps and the
measurement for each.
