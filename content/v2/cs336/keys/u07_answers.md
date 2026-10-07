# U07 answer key

All numeric claims from `../visuals/compute_u07.py` or the lab run
(executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

Bc = 102400 // (4*64*2) = 200.

## A1

(a) Bc = SRAM // (tiles*dh*bytes). (b) 200, 21 blocks. (c) Bc grows
(400 at 200 KB), the block count halves, the algorithm is unchanged.

## A2

(a) m' = max(m, mb), l' = l*exp(m-m') + sum(exp(block-m')). (b) The
toy gives m=4.0, l=1.5530, exact match. (c) Nothing structural: the
same loop with more blocks, the statistics stay O(1) per row.

## A3

(a) Outer Q blocks, inner K/V blocks, online statistics, final
divide by l. (b) Max deviation 3.33e-16, traffic 537 MB -> 25.2 MB.
(c) The K/V inner loop grows linearly in T, SRAM still bounds the
tiles, and the online statistics stay exact, very long T needs the
block-sparse variants (C05).

## A4

(a) Store m, l per row, recompute S and P tiles. (b) dQ matches
finite differences to 3.41e-07. (c) Smaller tiles: more blocks, more
passes, same math.

## A5

(a) Skip fully-masked blocks before loading. (b) 21 blocks, 3 kept,
18 skipped. (c) The block mask must be computed per input: either
recompute it per forward (cost) or accept approximate structure.

## A6

(a) Fuse while the working set fits in SRAM and ops stay elementwise.
(b) 101 MB versus 34 MB. (c) A softmax needs row statistics: use the
online form (C02) or a separate reduction pass.

## A7

(a) pid -> tile index, the grid is the loop over tiles. (b) The mask
covers the tail, the numpy analog matches. (c) Double the resident
set in the budget equation, Bc shrinks.

## A8

(a) Tiny hand case, random small vs naive, edge shapes, dtype
variants. (b) 3.33e-16 on the toy. (c) Loosen the tolerance to the
dtype (1e-2-ish for bf16) or compare in fp32.

## A9

(a) Baseline, shapes, dtype, conditions. (b) 537/25.2 = 21.3x less
traffic at (B=2,h=8,T=4096) bf16. (c) The exact shapes, dtype,
baseline, measurement method, and variance.

## A10

(a) Bc = (SRAM-stats)//(tiles*dh*bytes). (b) 266/200/160 for 3/4/5
tiles, 100 for dh=128. (c) 100.

## A11

(a) Statistics and accumulators in fp32, I/O in the working dtype.
(b) fp16 accumulators deviate 3.77e-03 on the toy scores. (c)
Accumulate in fp32 anyway (register blocking) or shorten the
reduction, there is no free fix.

## A12

(a) Rows sum to 1, outputs in the convex hull of V, causality,
block-size independence. (b) All pass on the toy. (c) Row sums still
1 in expectation is wrong: dropout breaks exactness, causality and
block independence survive, the convex-hull claim needs the dropout
scaling accounted for.
