# U02 interview bank , questions

Closed-book. Keys in `u02_key.md`. Quotas: 6 breadth, 2 deep ladders of
5, 2 analytical, 1 implementation/debug, 2 changed-constraint, 1
research-critique.

## Breadth (6)

B1. What is the shape contract of a batched matmul, and how do you read
it right to left?
B2. What does the einsum string "bhid,bhjd->bhij" compute?
B3. What is the difference between a view and a copy, and how do strides
tell them apart?
B4. Why does training memory exceed parameter memory by so much?
B5. State the 2MNK rule and use it to estimate one transformer layer.
B6. What is arithmetic intensity, and what does a value of 1.0
FLOP/byte tell you?

## Deep ladders (2 x 5)

L1. Resource accounting.
- L1.1 Define the five rows of the training resource ledger.
- L1.2 Toy: compute optimizer bytes for 41.55M params under Adam fp32.
- L1.3 Derive why mixed precision totals the same bytes as fp32 Adam.
- L1.4 Implement the ledger, state the verdict rule with margin.
- L1.5 Compare with a framework memory snapshot, debug the 8-GPU OOM
  the ledger missed, critique the single-device assumption, propose the
  NCCL-row extension experiment.

L2. From FLOPs to wall time.
- L2.1 Define MFU and its denominator.
- L2.2 Toy: intensity 50 on a 100 TFLOP/s, 1 TB/s device. What rate
  does the roofline allow?
- L2.3 Derive the roofline cap formula and the machine balance.
- L2.4 Implement mfu with the cap, state when MFU > 1 indicts the
  FLOP count.
- L2.5 Compare MFU tracking with hardware counters, debug a "90
  percent MFU" claim on a memory-bound shape, critique the honest-count
  assumption, propose the batch-size MFU sweep.

## Analytical exercises (2)

E1. A model has d=1024, dff=4096, h=16, L=24, V=32000, trained in mixed
precision with Adam. Compute (a) parameter count, (b) optimizer+param
bytes, (c) per-layer forward GFLOP at B=8, T=512. Show each step.
E2. Decode runs with batch 1, sequence 2048, d=4096. Estimate the
arithmetic intensity of one QKV projection and name the bottleneck and
the first optimization lever.

## Implementation/debug task (1)

D1. This FLOP counter undercounts a SwiGLU layer. Find the bug and fix
it. Then state which lesson claim the fix changes.

```
def layer_flops_buggy(B, T, d, dff, h):
    qkv = 3 * 2 * B * T * d * d
    scores = 2 * B * h * T * T * (d // h)
    out = 2 * B * T * d * d
    ffn = 2 * 2 * B * T * d * dff   # <-- suspect
    return qkv + scores + out + ffn
```

## Changed-constraint scenarios (2)

S1. The device memory halves but the model and batch are fixed. Name
three ledger rows you can attack, the technique for each, and the cost.
S2. Sequence length grows 8x for a long-context fine-tune. Which ledger
row explodes, which FLOP term crosses over, and what are the first two
mitigations in order?

## Research critique (1)

R1. A blog claims "our kernel reaches 95 percent of peak FLOP/s" with no
stated FLOP count, no intensity, and no baseline. List three gaps and
the measurement that closes each.
