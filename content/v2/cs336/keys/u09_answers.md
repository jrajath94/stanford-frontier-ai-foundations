# U09 answer key

All numeric claims from `../visuals/compute_u09.py` or the lab run
(executed 2026-10-06). Reference: d=4096, L=32, 7B, bf16. Claim class:
REQUESTED-BRANCH.

## R1 (remediation)

7B fp16 = 14 GB, TP=4: 14/4 = 3.50 GB/GPU.

## A1

(a) Attention: split heads, MLP: column then row, 2 all-reduces per
layer. (b) 67.1 MB per layer, 2.1 GB per 32-layer step. (c) d=8192:
per-layer doubles to 134.2 MB (linear in d).

## A2

(a) Stage i holds 8 layers, p2p activations between stages. (b) 8
layers/GPU, 35 stage-times. (c) p=8, m=16: bubble = 7/23 = 30.4
percent: too big, raise m or use 1F1B.

## A3

(a) (p-1)/(m+p-1). (b) 42.9, 8.6, 46.7 percent. (c) (p-1)/(m+p-1) <
0.10 with p=8: m > 63: minimum m=64.

## A4

(a) Alternate 1F/1B per stage, steady state overlaps. (b) ~9.4
percent bubble, ~4 in flight. (c) Use fewer microbatches with
checkpointing, or accept the GPipe bubble, do not exceed memory.

## A5

(a) TP innermost, PP middle, DP outermost. (b) (8,2,4), (4,4,4),
(2,8,4). (c) N=256, node 8: e.g. tp=8, pp=4, dp=8: TP fills the node,
PP across 4 nodes, DP over 8 groups, defend by link tiers.

## A6

(a) TP: 2.1 GB/step in 64 messages, PP: p2p per boundary, DP: 14 GB
in 1 message. (b) High latency hurts TP most (many small messages).
(c) Batch the all-reduces (fewer, larger) or move TP fully
intra-node.

## A7

(a) max/mean stage time. (b) 1.14 naive, cost-based split reaches
1.00. (c) Split by measured time including the slow stage's
constraint, or reduce p.

## A8

(a) Memory O(L)->O(sqrt(L)) per microbatch, compute +33 percent.
(b) 100 vs 32 on the toy. (c) CPU offload or fewer layers per stage
(more PP).

## A9

(a) Decode is memory-bound: TP splits weight reads. (b) TP=4: 2.5
ms, PP=4: 16.0 ms on the toy. (c) PP (throughput over latency).

## A10

(a) Params shard over tp*pp, optimizer over dp. (b) (8,2,4): 0.875
and 14.0 GB/GPU, (4,4,4): same. (c) dp=1: no optimizer sharding,
64 GPUs need bigger tp*pp.

## A11

(a) Imbalance: stage skew. Bubble: fill/drain gaps. Tier: comm-
dominated step. (b) All three signatures on the toy. (c) Raise m or
switch to 1F1B, do not add stages.

## A12

(a) The five: product, tp<=node, HBM fit, m>>p, group matching.
(b) (8,2,4) passes, (16,2,2) fails tp<=node. (c) The link-tier
assumption: with 2 tiers the tp<=node rule may relax or tighten.
