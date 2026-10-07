# U07 interview key

## B1-B6

B1. Splitting work into SRAM-sized blocks. The SRAM budget equation
sets the size: tiles*Bc*dh*bytes <= SRAM.
B2. It rescales the running statistics when a new block raises the
max: alpha = exp(m - m_new).
B3. Only m and l per row (O(T)), scores are recomputed.
B4. Merging elementwise passes into one kernel. Legal while the
working set fits in SRAM and ops stay elementwise.
B5. The tile index, the grid is the loop over tiles.
B6. Rows sum to 1, outputs in the convex hull of V, causality,
block-size independence.

## L1

L1.1 B*h*T*T*bytes for the scores.
L1.2 537 MB vs 25.2 MB = 21.3x.
L1.3 acc' = acc*alpha + exp(S-m')V, l' = l*alpha + sum(exp(S-m')).
L1.4 The nested loops, check max deviation ~1e-15 vs naive.
L1.5 Linear attention is approximate O(T), tiled is exact. Debug:
the final /l. Critique: fp32 accumulators required. Experiment:
causal variant vs masked naive.

## L2

L2.1 Store m, l, recompute S, P tiles.
L2.2 dV = P^T dO.
L2.3 From d(softmax): the Jacobian gives P*(dP - rowsum(dP*P)).
L2.4 The tiled loops, check vs finite differences (3.41e-07 toy).
L2.5 Storing scores costs O(T^2) HBM. Debug: fp16 statistics
corrupt P. Critique: recompute assumes FLOPs are cheap vs HBM.
Experiment: dK vs finite differences.

## E1

(a) Bc = 102400//(5*128*2) = 80. (b) ceil(8192/80) = 103 blocks.
(c) Bc = 160, 52 blocks, the algorithm, exactness, and O(T)
statistics are unchanged. Rubric: (a) 2 pts, (b) 1 pt, (c) 1 pt.
Red flag: forgetting the statistics tile.

## E2

(a) Per pass: 4*2048*1024*2 = 16.8 MB, separate 50.3 MB, fused
16.8 MB. (b) The reduction needs the whole row: the tile-at-a-time
pattern breaks. Fix: a separate reduction pass or the online form.
Red flag: "fuse it anyway".

## D1

Bug: the old l and acc are not rescaled when the max increases.
`l = l + ...` and `acc = acc + ...` are only correct if m_new == m.
Fix: alpha = exp(m - m_new), l = l*alpha + exp(block-m_new).sum(),
acc = acc*alpha + exp(block-m_new)@Vb. The bug hides when blocks are
processed in decreasing-max order. Rubric: find 2 pts, fix 1 pt,
state the hiding condition 1 pt.

## S1

Breaks: (1) the inner loop does T/Bc passes per query block:
quadratic work returns in the pass count, fix with block-sparse
skipping. (2) fp16 statistics underflow over 1M elements, fix with
fp32 statistics. Red flag: "it just works".

## S2

The block mask cannot be precomputed: either compute it per forward
(paying the cost) or fall back to dense tiled. The skip fraction
must be measured, not assumed.

## R1

Gaps: (1) no baseline: the ratio is undefined, name it. (2) no
shapes: traffic math impossible, give them. (3) no traffic analysis:
cannot tell memory from compute win, report both.
