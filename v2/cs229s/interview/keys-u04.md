# keys-u04.md: interview answer keys, U04

Date: 2026-10-06. Minimum sufficient explanation, strong
answer, red flags, rubric, remediation per item.

## B1

Minimum: warp 32 lockstep threads, block on one SM,
grid all blocks. 256/32 = 8 warps.
Strong: notes divergence cost.
Red flags: "threads are independent."
Rubric: 2 points. Remediation: C01.

## B2

Minimum: stride 32 -> 32 transactions, stride 1 -> 1.
Strong: computes the 32x waste.
Red flags: "contiguous data is enough."
Rubric: 2 points. Remediation: C02.

## B3

Minimum: tiling blocks the working set for on-chip
reuse. Naive: 1024 reads per A element, tiled: 16.
Strong: names the k-loop as the tiled axis.
Red flags: "tiling changes FLOPs."
Rubric: 2 points. Remediation: C03.

## B4

Minimum: all block threads arrive before any proceeds.
Deadlock: barrier inside divergent control flow.
Strong: distinguishes fence from barrier.
Red flags: "barriers are free."
Rubric: 2 points. Remediation: C04.

## B5

Minimum: active/max warps. 48 KB blocks: 2 per SM =
16 warps = 25%.
Strong: names the limiter (shared).
Red flags: "100% is the goal."
Rubric: 2 points. Remediation: C05.

## B6

Minimum: m' = max(m, m_b), l' = l*exp(m-m') +
l_b*exp(m_b-m').
Strong: explains the rescale.
Red flags: dropping the exp factors.
Rubric: 2 points. Remediation: C09.

## B7

Minimum: outer streams K/V blocks, inner streams Q
blocks.
Strong: states the on-chip footprint per step.
Red flags: swapping the loops' roles.
Rubric: 2 points. Remediation: C10.

## B8

Minimum: forward stores (m, l), O(T), backward
recomputes score tiles from Q, K.
Strong: quantifies 8K vs 16.7M numbers at T=4096.
Red flags: "backward stores scores."
Rubric: 2 points. Remediation: C11.

## Deep ladder 1

L1a. M: running row max. L: running sum of
exp(x - m).
L1b. After [1,2]: m=2, l=1.368. After [3,100]:
m=100, l~1.0.
L1c. True normalizer = sum exp(x-m') = sum over
blocks of l_b*exp(m_b-m'), old blocks rescale by
exp(m-m'). Induction.
L1d. O(1) memory per row (two scalars).
L1e. Online: 1 pass, O(1) memory. Two-pass: 2
passes, O(T) memory.
L1f. Missing exp(m-m'): toy gives l=2.368 instead
of 1.0, probabilities sum wrong.
L1g. Not order-dependent: max and the rescaled sum
are associative. Prove by the merge formula.
L1h. Blocks with values 1000 apart, expect match to
1e-6 vs the reference, naive unshifted overflows.
Scoring: 1 point per rung.

## Deep ladder 2

L2a. 4 passes x 4096^2 x 2 bytes = 134 MB per head.
L2b. (4/2)*(4/2) = 4 block pairs.
L2c. Each block pair applies the online update, after
all K/V blocks, (m,l,o) equal the full softmax
statistics by the merge identity.
L2d. FLOPs O(T^2 d) unchanged, HBM traffic from
O(T^2) to tiled O(T^2 d^2/M).
L2e. Sparse: 16x less work, approximate. Flash: same
work, far less traffic, exact.
L2f. Tail masking when T is not a multiple of block
size.
L2g. Blocks must fit SRAM, too big fails to launch.
L2h. Time both over T, expect naive faster below T*
and FlashAttention faster above, report T*.
Scoring: 1 point per rung.

## A1

Scores: 4*8192^2*32*2 = 17179869184 = 16.0 GiB.
Q/K/V/O: 4*8192*4096*2 = 268435456 = 0.25 GiB.
Ratio: 64x.

## A2

By threads: 2048/256 = 8. By shared: 100//48 = 2. By
regs: 65536/(32*256) = 8. Blocks = 2 (shared
limits). Occupancy: 2*8/64 = 25%.

## D1

Bug: `l = l + lb` drops the rescale: old sums were
normalized to the old max. Fix: track m_new first,
then l = l*exp(m-m_new) + lb*exp(mb-m_new). Test:
compare against the two-pass reference on blocks
with different maxima (e.g. [1,2],[3,100]), the buggy
version gives l=2.368, correct is 1.0.

## S1

Infinite SRAM: one block covers the whole matrix, the
loops collapse to a single pass, (m,l) still needed
only if streaming, else plain softmax, traffic drops
to one read of Q,K,V and one write of O. Tiling has
no reason to remain except parallelism across SMs
(still tile for occupancy, not for memory).

## S2

Pick: sparse is fastest but risky on long-range,
low-rank similar, FlashAttention exact but slower
than sparse. With a 1% budget and a retrieval task,
start with FlashAttention (no risk), then test sparse
gated on a distance-stratified retrieval eval: sparse
only ships if accuracy at max distance stays within
budget.

## R1

(1) Same shapes/dtype/hardware and synced timing?
Invalid: no.
(2) Is the baseline FlashAttention-2 at its best
config? Invalid: untuned baseline.
(3) "Identical outputs" measured how (tolerance,
seeds)? Invalid: eyeballed.
(4) Which T regime? Invalid: only short T where
tiling overhead dominates.
(5) Independent reproduction script? Invalid: none.
