# U03 interview bank , questions

Closed-book. Keys in `u03_key.md`. Quotas: 6 breadth, 2 deep ladders of
5, 2 analytical, 1 implementation/debug, 2 changed-constraint, 1
research-critique.

## Breadth (6)

B1. What are the shapes into and out of one transformer block?
B2. Why is attention scaled by 1/sqrt(dh)?
B3. What does the causal mask do, and where does it enter the
computation?
B4. What is the difference between pre-norm and post-norm?
B5. How does RoPE encode position without adding position vectors?
B6. What breaks if the residual connections are removed?

## Deep ladders (2 x 5)

L1. Attention mechanics.
- L1.1 Define Q, K, V and their shapes.
- L1.2 Toy: hand-compute one attention row for 2x2 scores [[2,0],[1,1]]
  with dh=4.
- L1.3 Derive why the scale prevents softmax saturation.
- L1.4 Implement stable masked attention, state the invariants.
- L1.5 Compare single-head with multi-head, debug the missing scale,
  critique the unit-variance assumption, propose the scale-sweep
  experiment.

L2. Normalization and stability.
- L2.1 Define RMSNorm and LayerNorm.
- L2.2 Toy: compute RMSNorm of [3,4] with unit gain.
- L2.3 Derive the residual gradient identity term.
- L2.4 Implement pre-norm and post-norm wirings, state the zero-block
  test for each.
- L2.5 Compare the two wirings at depth, debug post-norm divergence,
  critique the small-correction assumption, propose the stream-norm
  experiment.

## Analytical exercises (2)

E1. A block has d=1024, h=16, dff=4096, T=2048, B=8. Compute (a) the
attention score FLOPs, (b) the SwiGLU FFN FLOPs, (c) which dominates
and by what factor.
E2. RoPE uses base 10000 and dh=128. What is the rotation angle for the
highest-frequency pair at position 100? What happens to that angle if
the base doubles?

## Implementation/debug task (1)

D1. This RoPE implementation scrambles positions. Find the bug and fix
it. State the invariant the fix restores.

```
def apply_rope_buggy(x, ang):
    # x: (B, T, h, dh), ang: (T, dh//2)
    xr = x.reshape(*x.shape[:-1], -1, 2)
    cos = np.cos(ang)[None, None, :, :]   # <-- suspect
    sin = np.sin(ang)[None, None, :, :]
    o0 = xr[..., 0] * cos - xr[..., 1] * sin
    o1 = xr[..., 0] * sin + xr[..., 1] * cos
    return np.stack([o0, o1], -1).reshape(x.shape)
```

## Changed-constraint scenarios (2)

S1. The model must run with T=131072 at fixed d. Which two block
components break first (one in FLOPs, one in memory), and what is the
replacement for each?
S2. You must cut the block's parameters by 30 percent without changing
d or L. Name two options, the exact parameter math for each, and the
likely quality cost.

## Research critique (1)

R1. A paper introduces a new position encoding and shows better
perplexity at T=2048 but never tests past the training length and never
ablates against RoPE. List three gaps and the experiment for each.
