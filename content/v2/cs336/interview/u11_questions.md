# U11 interview bank , questions

Closed-book. Keys in `u11_key.md`. Reference: L=32, GQA 8 KV heads,
dh=128, bf16, 7B.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Why is decode memory-bound and prefill compute-bound?
B2. Write the KV-cache per-token formula.
B3. What does continuous batching change vs static batching?
B4. State the speculative-decoding speedup formula.
B5. What three sampling controls reshape the output distribution?
B6. Define TTFT, TPOT, and p99.

## Deep ladders (2 x 5)

L1. KV cache.
- L1.1 Define the cache and the append rule.
- L1.2 Toy: compute 128 KB/token and 8.59 GB at b=16, T=4096.
- L1.3 Derive the per-token byte count from the architecture.
- L1.4 Implement `kv_bytes`, state the doubling check.
- L1.5 Compare with recompute, debug the MHA-vs-GQA mistake,
  critique the no-sharing assumption, propose the h_kv sweep.

L2. Speculative decoding.
- L2.1 Define draft and verify.
- L2.2 Toy: alpha=0.7, k=3 gives E=2.53, speedup 1.95x.
- L2.3 Derive E=(1-alpha^{k+1})/(1-alpha).
- L2.4 Sketch the loop, state the exactness check.
- L2.5 Compare with Medusa, debug the style-mismatch collapse,
  critique the independence assumption, propose the k sweep.

## Analytical exercises (2)

E1. 70B model, L=80, h_kv=8, dh=128, bf16, batch 8, T=4096.
(a) Compute the KV cache bytes. (b) The GPUs have 80 GB each and
weights take 140 GB. How many GPUs for weights+cache? (c) Name
two ways to cut the cache.
E2. TTFT toy: mean 120 ms, p99 533 ms. (a) What service
distribution gives p99/mean ~4.4? (b) The SLO needs p99 < 500 ms.
Name two fixes and which phase each attacks.

## Implementation/debug task (1)

D1. This paged-cache allocator leaks blocks. Find the bug.

```
def evict(seq):
    for b in seq.blocks:
        free_list.append(b)
    # seq.blocks not cleared, seq reused later
    schedule(seq)  # reuses stale block list
```

## Changed-constraint scenarios (2)

S1. All requests share an identical 2000-token system prompt, 100
concurrent. Size the cache with and without sharing, decide if a
dedicated prefix GPU is worth it.
S2. The draft model is 1/2 the target size, not 1/10. Recompute
the toy speedup at alpha=0.7 and judge the scheme.

## Research critique (1)

R1. A serving paper reports 3x throughput vs a baseline with no
batch-size, no sequence-length distribution, and no latency
numbers. List three gaps and the measurement for each.
