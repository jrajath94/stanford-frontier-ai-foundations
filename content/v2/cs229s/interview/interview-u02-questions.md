# Interview bank U02: Hardware-aware design and compilers

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Name the four GPU memory levels from fastest to
slowest, with one size order each.

B2. Write the global thread index formula for a 1-D grid.
For N = 100000 and 512 threads per block, how many blocks
launch and how many threads idle?

B3. Define kernel fusion. What does it save: FLOPs or
bytes?

B4. Define arithmetic intensity with units. Compute it for
a 64x64 fp16 matmul.

B5. State the roofline formula. Define the ridge point and
compute it for pi = 3e14 FLOP/s, beta = 2e12 B/s.

B6. For a (4, 8) fp16 C-order matrix, write the strides in
bytes. Why is a column read expensive?

B7. Name the three parts of the tensor-core contract
(dtype, shape, alignment).

B8. State the fp16 vs bf16 tradeoff in one line each.

## Deep ladder 1: roofline triage

L1a. Define: what is arithmetic intensity, and what two
counts does it need?
L1b. Toy: 64x64 fp16 matmul. Compute FLOPs, bytes, and I.
L1c. Derive: from I, pi, and beta, derive the attainable
performance formula.
L1d. Implement and complexity: write `roofline(I)` and
`bound(I)`, state the complexity of the triage itself.
L1e. Compare: the same linear layer at batch 1 (I~1) and
batch 512 (I~512). Name each bound and ceiling at the toy
specs.
L1f. Debug: I = 800 but achieved FLOP/s is 5 percent of
peak. The roofline says compute-bound. Name two causes
that keep the kernel below the roof.
L1g. Critique: state the sustained-roof assumption and
name a workload where nameplate pi misleads.
L1h. Design: propose the three-test protocol (C12) for a
new kernel. Name the confounder that breaks test A and
how to check for it.

## Deep ladder 2: fusion economics

L2a. Define: what is a fused kernel, and what crosses the
HBM boundary in the fused vs unfused case?
L2b. Toy: y = relu(x@W + b), x (64,8), W (8,8), fp16.
Count unfused vs fused traffic in bytes.
L2c. Derive: show the traffic ratio as a function of the
number of passes over the intermediate.
L2d. Implement and complexity: write `traffic()` and
state what fusion does NOT change.
L2e. Compare: fusing a 536 MB MLP intermediate vs fusing
a 2 KB intermediate. When is fusion pointless?
L2f. Debug: torch.compile made the model slower. Name
the first diagnostic and the fix direction.
L2g. Critique: state the on-chip-fit assumption. What
happens when the intermediate is 4 GB?
L2h. Design: propose a measurement of HBM traffic with
and without compilation for an MLP block. Name the
expected drop and what it must match.

## Analytical exercises

A1. A transformer layer launches 50 kernels per step at
5 us overhead each, plus 2 ms of real work. Compute the
overhead share. Then compute it again if fusion cuts the
layer to 4 kernels. Show each step.

A2. Copy 1 GB at 2 TB/s with 0.5 us latency, then copy
1 KB under the same spec. Compute both times and the
latency share in each. What does this imply for small
kernels?

## Implementation / debug task

D1. The launch plan below should cover N = 1000 elements
with 256 threads per block, one thread per element. It
produces wrong outputs in the tail. Find the bug, fix it,
and state the invariant the fix restores.

```python
import math
def launch(n, block=256):
    blocks = n // block  # bug is here
    return blocks
```

## Changed-constraint scenarios

S1. Constraint change: shared memory is removed from the
architecture (only registers, L2, HBM remain). Which two
U02 techniques break first, and what replaces tiling?

S2. Constraint change: launch overhead drops to zero and
HBM bandwidth is infinite, but SM count is fixed. What
now limits a 64x64 matmul? What now limits decode of a
7B model? Give the new bottleneck in each case.

## Research critique

R1. A vendor slide claims "our kernel achieves 95 percent
of roofline on all shapes." List five audit questions you
ask before believing it, and for each, state the answer
that would invalidate the claim.
