# keys.md, U04 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Warp: 32 threads in lockstep. Block: threads on one SM
with shared memory. Grid: all blocks of a launch.

## E02

Warps: ceil(1000/32) = 32. Blocks of 256: ceil(1000/256)
= 4.

## E03

Divergence: lanes of a warp take different branches, so
the warp executes both paths serially.

## E04

Warp 1: no divergence possible (each thread independent),
but no lockstep amortization either, shuffle/vote
intrinsics vanish and scheduling overhead per thread
rises.

## E05

Hypothesis: the divergent branch costs ~2x the uniform
branch at warp granularity. Report both timings.

## E06

Coalesced: the 32 threads of a warp touch adjacent
addresses, merging into minimal transactions.

## E07

fp16, stride 8: addresses 2 bytes apart * 8 = 16-byte
spacing. 32 threads span 512 bytes = 4 lines. 4
transactions.

## E08

Strided or scattered access: each thread pulls its own
cache line. Fix the index math or stage via shared.

## E09

32-byte lines, fp32 stride 1: 32*4 = 128 bytes = 4
transactions (no longer 1).

## E10

Hypothesis: copy time steps up with each new cache line
touched, ~32x at stride 32. Report the curve.

## E11

Tiling splits the working set into blocks sized for
on-chip memory, reusing each loaded element many times.

## E12

Naive: each A element read 1024 times (once per k).
Tiled (64): 1024/64 = 16 times. 64x fewer A reads.

## E13

Shared memory (or registers): the tile does not fit the
per-block budget.

## E14

One tile for the whole matrix: each element loads once.
The limit becomes registers per thread.

## E15

Hypothesis: time falls with tile size until shared
memory caps it, then launch fails. Report the best tile.

## E16

__syncthreads guarantees all threads in the block reach
the barrier before any proceeds: writes before the
barrier are visible after it.

## E17

1e6 * 20 = 2e7 cycles.

## E18

Divergent barrier: threads not taking the if never reach
__syncthreads, so the block deadlocks.

## E19

Blocks cannot share data without shared memory: only
via global memory with kernel-boundary synchronization
(separate launches).

## E20

Hypothesis: removing barriers causes wrong answers first
at loads feeding shared reads. Report which barrier
breaks the sum.

## E21

Occupancy = active warps / max warps per SM.

## E22

12 KB/block: by_shared = 100//12 = 8 blocks, by_threads
= 2048//256 = 8, 8*8/64 = 100%.

## E23

Suspects: memory-bound kernel (bandwidth, not warps),
or instruction stalls (divergence, bank conflicts).

## E24

Threads per SM and shared memory per block remain,
register pressure vanishes.

## E25

Hypothesis: time improves with occupancy to ~50% then
flattens. Report the knee.

## E26

ceil(log2(4096)) = 12 rounds.

## E27

sum(0..4095) = 4095*4096/2 = 8386560.

## E28

Pad odd lengths with the identity element (0 for sum)
before halving.

## E29

Use warp shuffles within a warp and global atomics
across warps, or a grid-stride serial tail.

## E30

Hypothesis: 2-4 elements per thread is fastest, 1 wastes
launch, 16 wastes registers. Report the best.

## E31

Write scores (T^2), read for softmax, write probs, read
for AV: four HBM trips over the score matrix.

## E32

4 * 8192^2 * 32 * 2 = 17179869184 bytes = 16.0 GiB per
layer.

## E33

Fix direction: stop materializing scores (tile the
attention: FlashAttention) or approximate.

## E34

Traffic drops to the Q/K/V/O reads and writes: O(Td)
per head instead of O(T^2).

## E35

Hypothesis: attention time scales as T^2 and DRAM
bandwidth saturates. Report both.

## E36

Sparse (local window): O(Tw). Low-rank: O(Tr). Kernel
linearization: O(T).

## E37

8192/512 = 16x less work than dense.

## E38

The window broke a dependency longer than w that the
task needs. Exact attention (or a bigger w) restores it.

## E39

Only exact IO-aware methods (FlashAttention) survive a
zero error budget, all approximations change the math.

## E40

Hypothesis: sparse accuracy degrades with retrieval
distance, exact matches dense at all distances. Report
accuracy vs distance.

## E41

m' = max(m, m_b), l' = l*exp(m-m') + l_b*exp(m_b-m').

## E42

m' = 100. l' = 1.368*e^-98 + 1.0 ~ 1.0.

## E43

The rescale factor exp(m - m') is absent: old sums
were normalized to the old max.

## E44

No: the update is associative (max and the rescaled
sum). Order does not change the result.

## E45

Hypothesis: online matches the two-pass reference to
1e-6 on extreme values, naive unshifted exp overflows.
Report max error.

## E46

Outer loop streams K/V blocks, inner loop streams Q
blocks.

## E47

(4/2)*(4/2) = 4 block pairs.

## E48

Suspect: the tail handling (masking) for the last
block when T is not a multiple of Br/Bc.

## E49

With infinite SRAM, one block covers everything: the
loops collapse to a single exact pass (which is just
standard attention done on chip).

## E50

Hypothesis: FlashAttention wins past crossover T*,
below it naive wins. Report T*.

## E51

Forward stores (m, l): O(T) numbers. Backward
recomputes the score tiles from Q, K blocks.

## E52

Scores: 4096^2 = 16.7M numbers. (m, l): 8192 numbers.
2048x less storage.

## E53

The backward materialized something the forward did
not: likely the full score/probability matrix. Switch
to recompute.

## E54

Store everything: scores, probs, all intermediates.
FLOPs minimal, memory maximal.

## E55

Hypothesis: recompute wins past the memory wall and
loses below it. Report the crossover T.

## E56

Same shapes/dtype/hardware, warmup, device sync,
median-of-N, strongest baseline config.

## E57

Mean 70 us, median 50 us. Report the median.

## E58

Without sync the timer measured launch overhead, not
work: the 9x is fake.

## E59

Detect: repeat the benchmark interleaved (A/B/A/B),
a downward drift across repeats signals throttling.
Cool down and pin clocks if possible.

## E60

Hypothesis: each broken rule inflates the claimed
speedup, no-sync inflates most. Report inflation per
rule.
