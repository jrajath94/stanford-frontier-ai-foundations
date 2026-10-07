# Lab 04: Tiling and online softmax

Unit: cs229s-U04. Date: 2026-10-06.
Run: `python3 verify_lab04.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-04.md`.

## Objective

Verify U04 mechanisms on CPU with numpy: warp/thread
index math, coalescing transaction counts, tiling read
counts, the barrier contract (threads), occupancy bounds,
tree reduction, attention traffic scaling, cost ratios of
the four attention families, online softmax vs the
two-pass reference (including extreme values), the
FlashAttention toy vs naive attention, and backward
recomputation of dV with finite differences.

## Setup

```bash
python3 verify_lab04.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Check warp/lane mapping.
2. Count transactions vs stride.
3. Count tiled vs naive HBM reads.
4. Verify the barrier contract with threads.
5. Compute occupancy bounds, find the limiter.
6. Tree reduction vs sum().
7. Attention traffic table vs T.
8. Cost ratios: dense/sparse/low-rank.
9. Online softmax matches reference, extreme-value
   stress.
10. FlashAttention toy matches naive attention.
11. Recomputed dV matches reference, finite-difference
    check.

## Replication proposal (PROPOSED, not executed)

Question: does FlashAttention beat naive attention past
a crossover T* on a real GPU, and does the measured T*
match the SRAM-fit prediction? Hypothesis: yes, with
T* in the low hundreds. Method: time both over T in
{16..8192} at fixed d, also time the backward with and
without recompute. Budget: one GPU afternoon. Failure
criterion: no crossover found below 8K, or recompute
never wins.
