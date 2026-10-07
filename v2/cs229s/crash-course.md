# CS229S crash course: Systems for Machine Learning

Date: 2026-10-06. Baseline: October 6, 2026.
Condensed from all 10 units (U01-U10). Every number
is a lesson toy computed in the unit lessons, not a
vendor claim. Claim class: PLANNED / SOURCE
ATTRIBUTION PENDING unless a calendar anchor is
named. Calendar anchors: Week 1 Sep 27 (U01), Week 2
Sep 30 / Oct 04 (U02, U03), Week 3 Oct 07 / Oct 11
(U04), Week 4 Oct 14 (U05), Oct 18 / 21 / 25 (U06),
Week 6 Oct 28 / Nov 01 (U07), Week 7 Nov 04 / Nov 08
(U08), Oct 25 / Oct 30 / Nov 15 / Nov 18 (U09), Week
10 Dec 02 (U10).

## U01: the transformer workload

A transformer block is attention plus MLP. Per
token per layer: attention ~4n^2 + 4Tn (projections
plus scores), MLP ~8n^2 (up+down with d_ff=4n).
KV cache: 2*L*T*n bytes per float. Decode is
memory-bound. Prefill is compute-bound. Rule: count
FLOPs and bytes before touching the machine.

## U02: hardware-aware design

GPUs are a memory hierarchy: registers, shared
memory, HBM. Arithmetic intensity I = FLOPs/byte
decides the bound. The roofline caps attainable
FLOP/s at min(peak, I * bandwidth). Kernels fuse
to keep tensors on chip. Launch costs and
occupancy decide whether the fusion pays. Rule:
measure intensity first, optimize second.

## U03: transformer accounting and speculative decoding

Training: 6*N*D FLOPs. Decode: 2*N per token plus
KV traffic. KV bytes: 2*L*T*n per float. Batching
amortizes weight reads: throughput rises
sublinearly. Speculative decoding: a draft model
proposes, the target verifies. Exactness holds
because verification samples from the target.
Acceptance rate sets the speedup. Rule: decode is
a memory problem. Speculation trades draft
compute for fewer target passes.

## U04: CUDA and efficient attention

Threads form warps (32), warps form blocks.
Coalesced access merges transactions. Tiling fits
blocks in shared memory. Online softmax streams
with a running max. FlashAttention tiles attention
so the T x T matrix never materializes: exact,
IO-aware. Backward recomputes instead of storing.
Rule: never materialize what you can stream.

## U05: quantization, sparsity, structured operators

Linear quantization: s = (max-min)/(2^b-1), z =
round(-min/s). In-range error <= s/2. Per-channel
beats per-tensor on skewed weights (218x on the
quiet channel in the toy). 2:4 sparsity is the
pattern hardware likes. Butterfly: 2*n*log2(n)
params with full mixing. Monarch generalizes it.
Ship gates: perplexity, task, latency. Rule:
shrink the bytes, prove the quality.

## U06: training, adaptation, data pipelines

Loss is linear in log-log space: fit two points,
predict the third (slope -0.0748/decade in the
toy). Few-shot conditions without weight updates. 
threshold metrics manufacture emergence jumps.
SFT masks the instruction. RLHF climbs reward with
a KL leash. Constitutions move human input to the
rules. LoRA trains r(d+k) params and merges free.
The pipeline runs at the slowest stage. Packing
lifts real-token throughput 1.56x in the toy. 
report mean plus seed spread. Rule: predict in
log space, gate on quality, seed everything.

## U07: linear attention, SSMs, FFT

Quadratic attention: 2T^2d FLOPs, T^2 score
memory (256 MiB fp32 per head at T=8192). Linear
attention regroups: phi(Q)(phi(K)^T V), O(Td^2),
64x fewer FLOPs in the toy. SSM: h = Ah+Bx,
y = Ch. Unrolls to a convolution kernel K_k =
CA^kB. Scans parallelize it. FFT: conv in O(n
log n), 33x at n=1024. Stability: A^n compounds. 
fp16 dies at step 1115 for A=1.01. Price: d^2
state cannot hold T exact slots (64x short in the
toy). Rule: pay T^2 for exact routing or Td^2
for a blur. Grade guest claims by artifact.

## U08: serving and sparse MoE

Continuous batching boards on arrival (1.0 vs 1.5
s mean latency in the toy). Prefill is a gulp,
decode is sips: chunk the gulp. KV: 0.5 MiB per
token in the toy (L=32, n=4096, fp16). Report
p99, not means. MoE: top-k routing, capacity =
tokens*k/experts*factor, drops past capacity,
imbalance = max/mean (5x in the toy). Dispatch:
tokens*k*d bytes each way. Batching bends toward
an asymptote. Cost = price/rate. Rule: every win
lands in tokens per second.

## U09: parallelism, clusters, scheduling

Four splits: data, tensor, pipeline, expert. Each
has a tax (sync, collectives, bubbles, dispatch).
Ring all-reduce: 2(n-1)/n*S/BW (0.35 s in the
toy). Keep TP inside the node (12x wire gap in
the toy). Bubbles: (p-1)/(m+p-1). ZeRO-3: 16
bytes*P/n per GPU. Checkpoint: (N/k)*cost.
Straggler tax: (max-median)/median. DRF: max
demand ratio. MFU: achieved/peak. Cluster MTBF:
per-GPU/n (4.3 days at 256 in the toy). Rule:
name every tax with a formula.

## U10: retrieval, guests, synthesis

Index: amortize build over queries (break-even
160k in the toy). Dense 2.86 GiB vs PQ 61 MiB
(48x). ANN: set the recall floor, then pick.
RRF fuses by rank. Cascade: 405 ms, mostly
rerank. Grade guest claims: measured > derived >
stated > opinion. Proposals need failure
criteria. Profile from a hypothesis. Seeds: mean
plus spread. Gate: delta >= -epsilon. Defend
without bluffing. Rule: build it, measure it,
gate it, defend it.

## The one-page decision list

1. Count FLOPs and bytes first (U01-U03).
2. Never materialize T^2 (U04, U07).
3. Shrink bytes, prove quality (U05).
4. Fit laws, mask SFT, leash RL, seed runs (U06).
5. Batch continuously, chunk prefills, watch p99
   (U08).
6. Split wisely, mind the thin link (U09).
7. Amortize indexes, fuse ranks, gate quality
   (U10).
8. Grade every claim by its artifact (all units).
