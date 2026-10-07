# U03 interview key

## B1-B6

B1. (B,T,d) in, (B,T,d) out. Every sub-block preserves the shape.
B2. QK^T has variance dh under unit-variance components, without the
scale softmax saturates and gradients vanish.
B3. It adds -inf above the diagonal before softmax, so future weights
are exactly zero and training matches autoregressive inference.
B4. Pre-norm: x + f(norm(x)), highway exact. Post-norm: norm(x + f(x)),
norm on the gradient path.
B5. It rotates Q/K pairs by position-dependent angles, the dot product
then depends on relative distance. No vectors are added.
B6. Gradients lose the identity path, deep stacks stop training past a
few layers.

## L1

L1.1 Q,K,V: (B,T,d), reshaped to (B,h,T,dh).
L1.2 Scores [[2,0],[1,1]]/sqrt(4) = [[1,0],[0.5,0.5]]. Row 0: softmax =
[e/(e+1), 1/(e+1)] = [0.731, 0.269]. Row 1: [0.5, 0.5].
L1.3 Var(QK^T) = dh, dividing by sqrt(dh) restores unit variance, the
softmax's healthy regime.
L1.4 Stable masked attention: max-subtract, mask before softmax,
invariants: rows sum to 1, future weights zero.
L1.5 Heads: pattern diversity at fixed width. Debug: peaked weights and
dead gradients indict the missing scale. Critique: components are
rarely exactly unit-variance. Experiment: sweep the scale, measure
gradient norms, expect the peak near 1/sqrt(dh).

## L2

L2.1 RMSNorm: x/rms*g. LayerNorm: (x-mean)/std*g + b.
L2.2 rms = sqrt((9+16)/2) = 3.54, output [0.85, 1.13].
L2.3 dy/dx = I + J_f, the identity term survives even when J_f
vanishes.
L2.4 Pre: zero blocks give output == input. Post: zero blocks give
output == norm(input).
L2.5 At depth pre-norm trains easily, post-norm needs warmup and init
care. Debug: gradient norms by layer, add warmup or switch wiring.
Critique: corrections should stay small. Experiment: stream norm
versus depth, expect growth for pre-norm, flat for post-norm.

## E1

dh = 64. (a) Scores: 2*8*16*2048^2*64 = 68.7 GFLOP (the mix costs the
same again). (b) FFN: 3*2*8*2048*1024*4096 = 412.3 GFLOP. (c) FFN
dominates by 6.0x on scores alone, 3.0x counting scores+mix. Rubric:
2 pts per part. Red flag: using 2 FFN maps.

## E2

Highest frequency: i=0, w = 10000^0 = 1. Angle at position 100: 100
radians. Doubling the base leaves w_0 = 1 unchanged, so this angle does
not move, the base stretches the low-frequency pairs. Red flag:
"the angle halves".

## D1

Bug: cos/sin have shape (1,1,T,dh//2) but xr[...,0] has (B,T,h,dh//2),
the head axis is absent, so broadcasting fails (or silently
misaligns in frameworks with looser rules). Fix: index as
`np.cos(ang)[None, :, None, :]`. Restored invariants: norm
preservation and shift invariance of dot products. Rubric: find 2 pts,
fix 1 pt, invariant 1 pt.

## S1

FLOPs: the T^2 attention scores (scores+mix) explode, replacement: a
subquadratic attention variant (U04) or sequence parallelism (U09).
Memory: the (B,h,T,T) weights and the KV cache, replacement: tiled
exact attention without materializing weights (U07) plus KV
quantization or sharing (U04). Red flag: "just use a bigger GPU".

## S2

Option 1: untie nothing but shrink dff by the needed ratio, SwiGLU
params 3*d*dff scale linearly. Option 2: share (tie) the QKV or FFN
weights across layers, saves L-1 copies of those matrices. Math: state
both counts. Cost: capacity loss, sharing also couples gradients
across layers. Red flag: cutting heads (h) without noting dh changes.

## R1

Gaps: (1) no length extrapolation test, run past training length.
(2) no RoPE baseline, ablate head-to-head at matched compute.
(3) no mechanism analysis, test the claimed property (e.g.
shift-invariance) numerically. Red flag: accepting perplexity alone.
