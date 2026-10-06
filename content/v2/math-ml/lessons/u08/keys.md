# Keys, lesson 08 neural and sequence architectures

Date: 2026-10-06. All numbers computed 2026-10-06 unless marked
as hand arithmetic (verifiable by hand).

## E01

Row 1: 1(1)+0(2)+0 = 1. Row 2: 0(1)+1(2)-1 = 1. Row 3:
-1(1)+1(2)+0.5 = 1.5. Confirmed.

## E02

New output 0.25. The exact zero was luck of b2 =
0.25 canceling 1 - 2 + 0.75 = -0.25.

## E03

(W2 W1) x + (W2 b1 + b2). Two affine maps compose to
one affine map.

## E04

z = 1, a = 1, o = 2, L = 2. do = 2. dw2 = 2(1) = 2,
db2 = 2, da = 2(2) = 4, dw1 = 4(1) = 4, db1 = 4.

## E05

dL/do = o - y = 1.3 - 3.0 = -1.7. db2 = dL/db2 =
dL/do = -1.7.

## E06

Worse: truncation error grows with h (O(h^2) for
central differences is still small, but roundoff
stays flat. at h = 1e-5 the balance shifts toward
truncation). Measured agreement degrades slightly.
Accept a tested answer either way with numbers.

## E07

[6, 9]. The kernel sums each window: a local sum
detector.

## E08

7 - 3 + 1 = 5. Rule: n - k + 1.

## E09

784(64) + 64 = 50,176 + 64 = 50,240.

## E10

3(3)(3)(16) + 16 = 432 + 16 = 448.

## E11

(32 - 5 + 4)/2 + 1 = 31/2 + 1 = 15.5 + 1. Not an
integer: 31/2 = 15.5 floors to 15, output 16 with
one column dropped. The divisibility check fails.
pad 2 with stride 2 on 32 does not divide evenly.

## E12

24(24)(8) = 4,608 floats = 18,432 bytes.

## E13

tanh(1.0) = 0.7616, tanh(0.5) = 0.4621. Confirmed.

## E14

h_1 = tanh(W_x(-0.5)) = tanh([-0.5, -0.25]) =
[-0.4621, -0.2449]. No memory of x_0: each step
reads only the current input.

## E15

0.9^20 = 0.1216 (decay), 1.1^20 = 6.727 (growth).

## E16

0.97^50 = 0.218. Decay, but mild: the gradient
keeps a fifth of its size.

## E17

sigmoid(-0.38) = 1/(1 + e^0.38) = 1/2.4621 =
0.4061. Confirmed.

## E18

c_new = c, h_new = o tanh(c). The cell copies
itself. the gradient flows back unchanged. That
is the carousel: an unguarded identity path.

## E19

Sum = 2.6065. Weights: 1/2.6065 = 0.3837,
0.6065/2.6065 = 0.2327, 1/2.6065 = 0.3837. Sum
1.0.

## E20

0.2327 + 1.1511 + 1.9185 = 3.3023 (rounds to
3.3019 with full precision). second: 0.4654 +
1.5348 + 2.3022 = 4.3024 (4.3019 full precision).
Confirmed within rounding.

## E21

Softmax of scores spread over tens of units is
nearly one-hot: the max key takes ~1.0, the rest
~0. Gradients through it are ~0. the keys stop
learning.

## E22

Per-head dim 8/4 = 2. Q shape per head (5, 2).

## E23

(5,8) + (5,16) is illegal. Fix: project the FFN
output back to (5, 8) before the add (the second
FFN layer does this).

## E24

Mean 4, std 0: division by zero. Standard fix:
add eps inside the sqrt, i.e. divide by
sqrt(var + eps).

## E25

Mean 2, std 1. Normalized: [-1, 1].

## E26

v_hat = [0.25, 0.09] as before. update = 0.01
[0.5, -0.3]/( [0.5, 0.3] + 1e-3 ) = [0.00998,
-0.00997]. Changes in the 4th digit: negligible
here, decisive when v_hat is tiny.

## E27

v_10 = g(1 + 0.9 + ... + 0.9^9) = g(1 - 0.9^10)/
0.1 = g(6.513). The series sums the geometric
progression.

## E28

Accept code with asserts like
assert abs(dw1 - fd) < 1e-6.

## E29

Accept code with assert np.allclose(A.sum(axis=1),
1.0).

## E30

Accept any hypothesis of the form: claim,
baseline, metric, falsification condition.

## Ladders

L01. (1) z = Wx + b with matching inner dims,
then a per-item nonlinearity. (2) The C01 trace.
(3) (m,n)(n,) contracts the n axis. any mismatch
is not a matrix product. (4) Accept asserts.
(5) W1 (3,2) against x (4,): the first matmul
breaks.

L02. (1) One backward walk gives all parameter
gradients via the chain rule. (2) -5.1, -1.36,
-2.55, -1.7. (3) do = o - y. da = do w2. dz =
da [z>0]. dw1 = dz x. (4) Accept the
implementation. (5) See interview keys-u08 T1.

L03. (1) Slide, dot, no padding. (2) [-2,-2].
(3) The kernel's left edge visits positions
0..n-k: n-k+1 stops. (4) Accept the loop.
(5) Stride 2: ceil/floor of (n-k+1)/2 stops.
with n=4,k=3: 1 stop.

L04. (1) One kernel reused at all positions.
(2) 102,500 vs 208. (3) Count = (k_h k_w
in_ch + 1) out_ch. (4) Accept the counter.
(5) Absolute-position tasks, e.g. "dot in the
top-left corner".

L05. (1) floor((W-K+2P)/S)+1. (2) 24 then 12.
(3) Placements 0..W-K step S: (W-K)/S + 1.
(4) Accept the helper. (5) Assert (W-K+2P) % S
== 0 before building.

L06. (1) h_t carries the past in fixed dim.
(2) [0.7616,0.4621] then [0.0169,0.0572].
(3) Repeated application of the same cell.
(4) Accept the loop. (5) Gradient first: the
w^T product dies or explodes long before memory
runs out.

L07. (1) The T-step factor is the per-step slope
to the T. (2) 9.77e-4, 57.67. (3) Chain rule
over T identical links. (4) Accept the plot.
(5) Gating treats the cause (the multiplication).
clipping treats the symptom (the size).

L08. (1) Forget, input, output: soft switches on
erase, write, read. (2) 0.6106, 0.4061, 0.5581,
0.6502, 0.4098, 0.2525. (3) f=1,i=0 gives
c_new = c: gradient 1. (4) Accept the step.
(5) f = sigmoid(0.1) ~ 0.52 with the toy bias.
with bias 0 exactly 0.5: the cell halves memory
each step at init.

L09. (1) Softmaxed scores over keys. (2) The 2x3
matrix, outputs [3,4] and [3.3019,4.3019].
(3) exp(s - max) / sum is identical to exp(s) /
sum and avoids overflow. (4) Accept the code.
(5) 100k^2 = 1e10 weights: 40 GB in float32.
Alternative: linear attention O(n d^2).

L10. (1) Attention, residual+norm, FFN,
residual+norm. (2) The nine-row trace. (3) X +
sublayer(X) is only defined for equal shapes.
(4) Accept the asserts. (5) "Attention is all
you need": attack with O(n^2 d) memory vs O(n)
state, and the C06 erasure showing recurrence
still has a streaming niche.
