# U08 answer key

All numeric claims from `../visuals/compute_u08.py` or the lab run
(executed 2026-10-06). 7B reference: 14+14+56 = 84 GB. Claim class:
REQUESTED-BRANCH.

## R1 (remediation)

13B: params 26, grads 26, optim 104 = 156 GB replicated. ZeRO-3 over
8: 156/8 = 19.5 GB/GPU.

## A1

(a) All-reduce the shard grads, divide by N. (b) The mean identity
holds exactly (lab T1). (c) Batchnorm statistics differ per shard:
use synchronized or no batchnorm (transformers use LayerNorm).

## A2

(a) Ring: 2*(N-1)/N*bytes/bw. Tree: 2*log2(N) latency hops. (b)
0.122 s, at N=1024 the ring gives 0.140 s ~= 2*bytes/bw. (c) N=1024:
ring for large grads (bandwidth-optimal, N-independent), tree for
small control messages.

## A3

(a) Stage 1: optimizer states, 2: +grads, 3: +params. (b) 35.0,
22.8, 10.5 GB/GPU at 8 GPUs. (c) ZeRO-3 still shards the 5B layer
(0.625B params/GPU) but the all-gather per layer is large: add
tensor parallelism (U09) for the layer itself.

## A4

(a) Bucket grads in reverse layer order, all-reduce while the next
layer computes. (b) 96 ms vs 36 ms. (c) Larger buckets (fewer, later
all-reduces are fine since compute dominates), watch the tail.

## A5

(a) Scale loss by S, unscale grads, skip on overflow, adapt S.
(b) 6e-8*1024 = 6.14e-05, representable, unscaled recovers 6e-8.
(c) Lower S (dynamic scaling halves it), find the overflow source.

## A6

(a) All-gather params, compute, release, reduce-scatter grads.
(b) Gathered 2.0 MB bf16, resident 0.5 MB on the toy. (c) Wrap
coarsely (several layers per unit) to amortize latency.

## A7

(a) Gather-to-one vs per-rank shards. (b) One 84 GB file vs 8 x
10.5 GB, reshard 8->4 round-trips exactly. (c) Always save sharded
with N in the metadata, reshard on load.

## A8

(a) Exposed = max(0, comm - compute). (b) 0.0 ms and 8.0 ms.
(c) Reduce comm (bucketing, compression), increase compute per
bucket, or buy bandwidth, check stream dependencies first.

## A9

(a) T1/(N*TN). (b) 89 percent. (c) Per-rank timings (straggler?),
communication fraction, data loading stalls.

## A10

(a) Checkpoint every k, restart from latest, lost work <= k steps.
(b) 500 s expected lost work at k=100, 10 s steps. (c) Shorten k
(e.g. 20-50): expected waste scales with k.

## A11

(a) Shard by rank, seed from (seed, epoch, rank). (b) Disjoint and
covering on the toy. (c) Recompute the sharding for the new N and
re-seed, the epoch's data order changes, so log it.

## A12

(a) DP==big-batch, shards disjoint/covering, step counts agree,
reshard round-trips. (b) All pass, the missing-/N bug fails
invariant 1. (c) The sharding-correctness invariant (disjoint,
covering, deterministic) before any performance work.
