# Lab 02: Roofline triage

Unit: cs229s-U02. Date: 2026-10-06.
Run: `python3 verify_lab02.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-02.md`.

## Objective

Verify the U02 hardware reasoning on CPU with numpy: the
memory hierarchy timing model, kernel index math, fusion
traffic savings, arithmetic intensity, the roofline with
kernel points, layout strides, vectorized trip counts,
launch overhead, overlap, tensor-core shape rules, and
precision range behavior.

## Setup

```bash
python3 verify_lab02.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Model two-level read time and find the
   latency/bandwidth crossover size.
2. Check grid/block index math and idle threads.
3. Verify fusion traffic savings on the toy.
4. Compute intensity for matmul vs matvec toys.
5. Place kernel points on the roofline and check the
   ridge continuity.
6. Check numpy strides for C vs Fortran order.
7. Time row vs column sums (numpy) and compare ratios.
8. Count scalar vs vector loads.
9. Verify launch overhead linearity toy.
10. Verify overlap = max(compute, copy).
11. Check tensor-core shape qualification.
12. Demonstrate fp16 overflow and bf16 range.

## Replication proposal (PROPOSED, not executed)

Question: does the roofline triage (three tests in C12)
agree with profiler classification on real kernels?
Hypothesis: at least 8 of 10 kernels agree, disagreements
trace to occupancy or launch overhead. Method: 10 kernels
spanning matmul, matvec, reductions, and elementwise ops,
run the batch sweep, precision halving, and intensity
computation, compare against a profiler's bound label.
Budget: one GPU afternoon. Failure criterion: systematic
disagreement with no occupancy explanation.
