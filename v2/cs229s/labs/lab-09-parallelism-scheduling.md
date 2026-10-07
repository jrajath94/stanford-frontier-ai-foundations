# Lab 09: Parallelism, clusters, and scheduling

Unit: cs229s-U09. Date: 2026-10-06.
Run: `python3 verify_lab09.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-09.md`.

## Objective

Verify U09 mechanisms on CPU with numpy: per-GPU bytes for
the four splits, ring all-reduce times, the 12x placement
ratio, the 40% straggler tax, bubble fractions, ZeRO-3 vs
DP bytes, checkpoint overheads, the FIFO makespan, the
preemption rework fraction, DRF dominant shares, MFU, and
cluster MTBF.

## Setup

```bash
python3 verify_lab09.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Four-split bytes.
2. Ring all-reduce times.
3. Placement ratio.
4. Straggler tax.
5. Bubble fractions.
6. ZeRO-3 sharding.
7. Checkpoint overhead.
8. FIFO makespan.
9. Preemption rework.
10. DRF shares.
11. MFU.
12. Cluster MTBF.

## Replication proposal (PROPOSED, not executed)

Question: does all-reduce time follow the ring slope in S,
and do pipeline bubbles follow (p-1)/(m+p-1)? Hypothesis:
both formulas fit measured data within 20%. Method: NCCL-
style timing across S on 8 GPUs (or a faithful simulator
on CPU), plus a pipeline simulator across m. Budget: GPU
hours for the real run, CPU for the simulator. Failure
criterion: systematic deviation (then the model misses a
term, e.g. latency).
