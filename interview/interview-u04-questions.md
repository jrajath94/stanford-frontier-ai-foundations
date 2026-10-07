# Interview bank U04: CUDA and efficient attention

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Define warp, block, grid. How many warps in a
256-thread block?

B2. 32 threads read fp32 with stride 32. How many
128-byte transactions? What about stride 1?

B3. Define tiling. For a (1024,1024,1024) matmul with
64-wide tiles, how many times is each A element read
from HBM vs naive?

B4. What does __syncthreads guarantee? Name the
deadlock mistake.

B5. Define occupancy. A 256-thread block uses 48 KB
shared per SM-budget 100 KB: what is the occupancy at
64 max warps?

B6. Write the online softmax (m, l) update for a new
block.

B7. Name the two FlashAttention loops and what each
streams.

B8. In the FlashAttention backward pass, what is stored
in the forward and what is recomputed?

## Deep ladder 1: online softmax

L1a. Define: what are m and l for a row?
L1b. Toy: blocks [1,2] then [3,100]. Compute m and l
after each block.
L1c. Derive: prove the merge formula l' = l*exp(m-m')
+ l_b*exp(m_b-m') gives the true normalizer.
L1d. Implement and complexity: write `online_softmax`
and state memory per row.
L1e. Compare: online vs two-pass softmax on memory and
passes.
L1f. Debug: probabilities do not sum to 1. Name the
missing factor and show its effect on the toy.
L1g. Critique: is the update order-dependent? Prove or
refute.
L1h. Design: propose the extreme-value stress test.
Name the expected error bound.

## Deep ladder 2: FlashAttention

L2a. Define: what HBM traffic does standard attention
pay per head at T=4096, fp16?
L2b. Toy: T=4, Br=Bc=2, d=2. How many block pairs run?
L2c. Derive: show the tiled loop computes exact
attention using the online update.
L2d. Implement and complexity: trace `flash_toy` on the
toy and state FLOPs vs HBM traffic complexity.
L2e. Compare: FlashAttention vs sparse attention (w=256)
at T=4096 on work, traffic, and exactness.
L2f. Debug: outputs match at T=64 but not T=65. Name
the suspect.
L2g. Critique: state the SRAM-fit assumption and the
failure when blocks are too big.
L2h. Design: propose the crossover experiment (naive vs
FlashAttention over T). Name the expected shape of the
result.

## Analytical exercises

A1. Standard attention, T=8192, h=32, fp16, 4 HBM
passes over scores. Compute total score traffic in GiB
per layer. Then compute the Q/K/V/O traffic for n=4096
(B=1). Give the ratio.

A2. A kernel uses 256-thread blocks, 32 registers per
thread, 48 KB shared per block. SM limits: 2048
threads, 100 KB shared, 65536 registers, 64 warps max.
Compute blocks per SM by each limiter and the
occupancy. Which limits?

## Implementation / debug task

D1. The tiled softmax below is wrong. Find the bug, fix
it, and state the test that catches it.

```python
import numpy as np
def bad_online(blocks):
    m = -np.inf
    l = 0.0
    for x in blocks:
        mb = x.max()
        lb = np.exp(x - mb).sum()
        m = max(m, mb)
        l = l + lb  # bug is here
    return m, l
```

## Changed-constraint scenarios

S1. Constraint change: SRAM is infinite. Redesign
attention: what happens to the two loops, the (m,l)
notebook, and the HBM traffic? Is there any reason to
keep tiling?

S2. Constraint change: the error budget allows 1%
relative error in attention outputs. Compare three
options (sparse w=256, low-rank r=256, FlashAttention)
on speed and risk for a long-range retrieval task.
Which do you pick and what test gates it?

## Research critique

R1. A paper claims a new attention variant is "2x
faster than FlashAttention-2 with identical outputs."
List five audit questions, and for each state the
answer that would invalidate the claim.
