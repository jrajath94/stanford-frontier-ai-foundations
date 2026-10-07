# U03 lab , transformer block mechanics

Environment: CPython 3 with numpy. Run: `python3 u03_lab_run.py`.
Expected outputs in `u03_lab_key.md` (executed 2026-10-06). Toy config
throughout: B=2, T=8, d=16, h=4, dh=4, dff=32, V=64, seed 0.

## Task 1 , embedding lookup

Implement `embed` with the range assert. Check output shape, row 0
mapping, and that id V raises.

## Task 2 , QKV

Implement `qkv` (separate maps) and a fused variant, assert allclose on
seeded data and report shapes.

## Task 3 , attention invariants

Implement stable `attention`. Assert row sums to 1 (max dev < 1e-12),
uniform input gives uniform weights, and a one-hot score row selects
the argmax position.

## Task 4 , causal mask

Build the mask, assert future weights are exactly 0, rows sum to 1,
and position 0 attends only to itself. Also demonstrate the
after-softmax bug: zero the future after softmax and show the row sum
breaks.

## Task 5 , head reshape

Implement `split_heads`/`merge_heads`, assert the roundtrip is exact
and head slices are disjoint. Assert a bad h (not dividing d) raises.

## Task 6 , residual identity

With zeroed sub-blocks assert the block output equals the input
exactly. Compare correction norm versus input norm at init and report
the ratio.

## Task 7 , pre versus post norm

Implement both wirings sharing sub-blocks. With zeroed sub-blocks
assert pre-norm output equals input and post-norm output equals
norm(input).

## Task 8 , norms

Implement RMSNorm and LayerNorm. Assert RMSNorm row rms in
[0.999, 1.001] and LayerNorm row mean/std at 0/1 within 1e-6. Check the
eps guard on a zero row (no NaN).

## Task 9 , FFNs

Implement SwiGLU and ReLU FFN. Assert shape roundtrips. Count params
for both at d=16, dff=32 and at matched budgets (ReLU dff=48 vs SwiGLU
dff=32: both 1536).

## Task 10 , RoPE

Implement `rope_angles`/`apply_rope`. Assert norm preservation (max dev
< 1e-12), position 0 is identity, and the shift-invariance of dot
products (max dev < 1e-9). Assert RoPE-on-V changes values (the bug
demo).

## Task 11 , output head

Implement tied logits. Assert shape (B,T,V) and softmax rows sum to 1.
Count grad rows: with a toy autograd-free check, show the embedding
matrix receives updates from head-side usage (structural check: the
same array object is used).

## Task 12 , init

Implement `xavier`. Assert sampled std within 10 percent of target for
a (256,256) matrix. Show that zero init gives identical head outputs
(symmetry demo).
