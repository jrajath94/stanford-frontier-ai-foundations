# Lab 05: Quantization and pruning

Unit: cs229s-U05. Date: 2026-10-06.
Run: `python3 verify_lab05.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-05.md`.

## Objective

Verify U05 mechanisms on CPU with numpy: the (s, z)
round-trip and zero property, the s/2 error bound,
per-channel vs per-tensor MSE, dynamic activation
params, percentile vs min/max calibration, 2:4 pattern
validation, magnitude pruning masks, the realized-speed
model, butterfly full mixing and param counts, Monarch
block structure, the error-propagation bound, and the
ship/no-ship gates.

## Setup

```bash
python3 verify_lab05.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Round-trip (s, z) on the toy, zero maps to zero.
2. In-range error bounded by s/2 on random data.
3. Per-channel MSE << per-tensor MSE on mixed ranges.
4. Dynamic params track the batch.
5. Percentile scale far finer than min/max with an
   outlier.
6. 2:4 validation.
7. Magnitude pruning mask.
8. Realized-speed arithmetic.
9. Butterfly: full mixing, param count.
10. Monarch: block structure.
11. Error bound values.
12. Ship gates.

## Replication proposal (PROPOSED, not executed)

Question: does per-channel int8 hold task quality
within the gates (C12) on a real model while
per-tensor fails? Hypothesis: per-channel passes all
three gates, per-tensor fails the task gate. Method:
one open 7B-class model, quantize weights per-tensor
vs per-channel (activations fp16), run perplexity, a
task suite, and latency. Budget: one GPU day. Failure
criterion: per-channel fails any gate (then the
granularity story is wrong for this model).
