# U09 interview key

## B1-B6

B1. Column-split the first matmul, row-split the second, one
all-reduce sums partial outputs per sublayer.
B2. Idle fill/drain time: (p-1)/(m+p-1). Shrink with more
microbatches, fewer stages, or 1F1B.
B3. One-forward-one-backward schedule: steady-state overlap with ~p
microbatches in flight.
B4. TP innermost (fast links), PP middle, DP outermost.
B5. Decode is latency-bound: TP splits weight reads per token, PP
adds stage hops per token.
B6. Product = N, tp <= node, HBM fit, m >> p, shard group matches
reduction group.

## L1

L1.1 Column split needs no comm, the following row split needs the
summed input.
L1.2 Exact sum, 67.1 MB per layer.
L1.3 Attention-out and MLP-down each sum partial products: 2 per
layer.
L1.4 The simulator, check concatenation equals unsharded.
L1.5 ZeRO-3 shards without per-layer comm but cannot split a layer.
Debug: keep TP intra-node. Critique: shapes are reference only.
Experiment: sweep tp, verify exactness.

## L2

L2.1 Stages hold layer ranges, microbatches flow through.
L2.2 (4-1)/(4+4-1) = 42.9 percent.
L2.3 Ideal m, actual m+p-1 stage-times.
L2.4 The simulator, check total = (m+p-1) stage-times.
L2.5 1F1B bounds memory at ~p in flight. Debug: m<p idles stages.
Critique: uniform-stage assumption. Experiment: sweep m.

## E1

(a) tp=8, pp=2, dp=2 (8*2*2=32, tp fills the node). (b) 140/16 =
8.75 GB/GPU. (c) 2*80*1*4096*8192*2 = 10.7 GB/step. Rubric: (a) 1
pt, (b) 1 pt, (c) 2 pts. Red flag: tp=32 across nodes.

## E2

(a) The slowest stage. (b) The formula assumes uniform stages, the
2x stage stretches every microbatch and the real bubble exceeds 8
percent. (c) Rebalance by measured cost, reduce p (fewer, larger
stages).

## D1

Bug: tp=16 exceeds the node size 8: the per-layer all-reduces would
span slow inter-node links. The product check is necessary but not
sufficient. Fix: tp=8, pp=4, dp=2 (or tp=8, pp=2, dp=4). Rubric: find
2 pts, fix 1 pt, state the insufficiency 1 pt.

## S1

Choose TP=8: per-token latency falls with tp (toy: 2.5 vs 16.0 ms).
PP=8 loses: each token pays 8 stage hops plus p2p latency, p99
misses the SLO. Risk of TP: per-layer comm at batch 1 needs fast
links.

## S2

Options: (1) GPipe schedule with m=2: bigger bubble, fits memory.
(2) Activation checkpointing: frees memory for ~4 in flight at +33
percent compute. (3) Fewer stages (p=2): less bubble pressure.
State the cost of each.

## R1

Gaps: (1) no triple: the result is not reproducible, give it. (2) no
link tiers: placement unknown, give them. (3) no per-axis breakdown:
cannot tell which axis helped, report comm/compute per axis.
