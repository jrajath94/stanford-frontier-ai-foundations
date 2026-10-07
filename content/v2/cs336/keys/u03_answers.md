# U03 answer key , lesson assessments

## R1 (remediation)

T=4, B=2, d=16, h=4, dh=4, dff=32, V=64. ids (2,4), embeddings (2,4,16),
q,k,v (2,4,16), heads (2,4,4,4), scores/weights (2,4,4,4), block out
(2,4,16), logits (2,4,64). Rubric: each shape 0.5 pt.

## A1

(a) In (B,T) ids, out (B,T,d), params V*d, grads sparse on used rows.
(b) Ladder. Lookup: row indexing. (2,3) ids -> (2,3,d). Zero FLOPs
because no arithmetic runs. one-hot: same math, wasteful. Debug: assert
0 <= ids < V before lookup. Critique: pipelines can emit bad ids.
Transfer: the embedding row moves by V*d*bytes, the head (if untied)
moves the same again.

## A2

(a) Wq,Wk,Wv: (d,d), q,k,v: (B,T,d), cost 3*2*B*T*d*d.
(b) Ladder. Roles: ask, advertise, carry. Toy: 3*2*2*8*16*16 = 24576
FLOP. Three maps because the roles differ. Fused: one (d,3d) matmul
then split. Debug: separate the maps, the shared map collapses roles.
Critique: dense only, structured variants exist. Transfer: the QKV
term (d^2) grows fastest.

## A3

(a) P = softmax(QK^T/sqrt(dh)), output = PV.
(b) Ladder. Distribution view: expected value under the row
distribution. One row by hand: subtract max, exp, normalize. Scale:
keeps score variance near 1 so softmax stays soft. Stable softmax:
max-subtraction. Unscaled: saturation at large dh. Debug: gradients
near zero with peaked weights, add the scale. Critique: assumes
unit-variance components. Transfer: scores spread ~sqrt(4) = 2x wider,
softmax sharpens and gradients shrink.

## A4

(a) The mask enters before softmax as -inf above the diagonal.
(b) Ladder. One-way mirror: past visible, future not. 3x3 hand mask:
zeros on and below diagonal, -inf above. Before softmax because
normalization must exclude the future. Prefix masks: chosen positions
see more. Debug: move the mask before softmax and re-check row sums.
Critique: left-to-right only. Transfer: a padding mask adds -inf on
pad key positions, entering at the same place.

## A5

(a) (B,T,d) -> (B,h,T,dh) -> scores -> (B,h,T,dh) -> (B,T,d).
(b) Ladder. Head: one (T,dh) attention unit. (1,2,4) into 2 heads:
[[[a,b],[c,d]]] split as head0 [a,b], head1 [c,d] per position.
Transpose puts h next to B for batching. One head: less diversity.
Debug: assert d == h*dh and test merge(split(x)) == x. Critique:
divisibility. Transfer: h=1 keeps capacity similar but kills pattern
diversity, KV cache unchanged in shape (still d per position) for MHA.

## A6

(a) y = x + f(x), dy/dx = I + J_f.
(b) Ladder. Highway: input reaches output directly. Derivative: I plus
the sub-block Jacobian. Depth: identity terms compound instead of
Jacobians. Gates: learned mixing, more params. Debug: check gradient
norms by layer, early layers near zero indicts the missing residual.
Critique: corrections should stay small. Transfer: growing corrections
signal instability, the norm placement (C07) and init (C12) are the
knobs.

## A7

(a) Pre: x + f(norm(x)). Post: norm(x + f(x)).
(b) Ladder. Highway: pre-norm keeps it exact, post-norm routes it
through the norm Jacobian. Pre-norm trains deeper stacks easily.
Implement: six lines each sharing sub-blocks. Compare: same FLOPs,
different dynamics. Debug: add warmup and check init, or switch to
pre-norm. Critique: gain is learnable. Transfer: warmup and the output
init scale matter most, keep the checkpoint's wiring.

## A8

(a) RMSNorm: x/rms*g. LayerNorm: (x-mean)/std*g + b.
(b) Ladder. rms: sqrt(mean(x^2)). [3,4]: rms = sqrt(12.5) = 3.54,
out = [0.85, 1.13]. Mean: residual streams keep their mean, skipping
it saves a reduction. Cost: both O(BTd), memory-bound. Debug: set eps
to 1e-6 and guard zero rows. Critique: last axis only. Transfer: fp16
norms need fp32 accumulation inside, eps choice matters more.

## A9

(a) (swish(xWg) * (xWu))Wd, three maps.
(b) Ladder. Gate: multiplicative modulation. Toy params: 3*16*32 =
1536. 2/3 rule: 3 maps * (2/3)*4d = 8d^2, matching ReLU's 2*4d = 8d^2.
ReLU: two maps, dff=4d. Debug: rescale dff by 2/3 when switching.
Critique: matched parameters, not matched FLOPs. Transfer: the three
weight matrices dominate HBM traffic in decode, quantization targets
them first.

## A10

(a) Pair rotation by t*w_i on consecutive entries.
(b) Ladder. Relative: (R_t q).(R_s k) depends on t-s. 90-degree rotation
of (1,0): (0,1). Q/K only because values must not move with position.
Absolute: fixed vectors, fail past training length. Debug: remove RoPE
from V. Critique: dh even. Transfer: the base frequency (theta),
larger base stretches the effective horizon.

## A11

(a) (B,T,d) -> (B,T,V), cost 2*B*T*d*V, tying shares E.
(b) Ladder. Logits: unnormalized scores. Toy: 2*2*8*16*64 = 32768
FLOP. Tying: geometry shared, gradients flow both ways. Untied: more
params. Debug: check the head weight shape is (V,d) or (d,V) per the
code's convention. Critique: frozen embeddings break the sharing.
Transfer: the head matmul (V*d per position) and the embedding table
(V*d params) dominate.

## A12

(a) std = gain*sqrt(2/(fan_in+fan_out)), keeps variances stable.
(b) Ladder. fan-in: input count. Var(y) = n*Var(W)*Var(x). 2/(m+n):
balances both directions. Zeros: symmetry never breaks. Compare:
Kaiming for ReLU. Debug: check per-layer gradient norms, identical
heads indict zero init. Critique: linear at init only. Transfer: scale
the output-projection gain by 1/sqrt(2L), halving per doubling of
depth in variance terms.
