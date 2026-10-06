# Interview bank, U08 neural and sequence architectures

Date: 2026-10-06. Questions only. Keys in interview/keys-u08.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Write the forward-pass shapes for x (4,), W1 (5,4),
b1 (5,), W2 (2,5), b2 (2,). What is the output shape?
B2. State the chain-rule step that turns do into dw1 in a
scalar net. What does reverse mode buy over forward mode
here?
B3. A 1D signal has length 9, kernel length 4, valid mode.
What is the output length? What breaks if the kernel has
length 10?
B4. Count parameters: 64x64 image to 50 hidden units dense,
versus eight 3x3 filters. Give both numbers with biases.
B5. Write the attention weight formula with the scale.
What happens to the softmax if the scale is dropped at
d = 512?
B6. One Adam update from zero state on g = [0.5, -0.3],
eta = 0.01: what is the update? What does it equal in
terms of sign(g)?

## Deep ladder D1, backprop to trust (5 follow-ups)

D1.1. Define reverse-mode differentiation in one sentence.
D1.2. Toy: the lesson's C02 numbers. Give all four
gradients and the finite-difference agreement.
D1.3. Derive or justify: dw1 = do w2 x for the scalar
net.
D1.4. Implement/debug: see T1 below.
D1.5. Changed constraint: the activation is tanh instead
of relu. Which line of the backward pass changes, and
what is the new local slope?

## Deep ladder D2, sequences (5 follow-ups)

D2.1. Define the RNN hidden state update in one sentence.
D2.2. Toy: the lesson's C06 numbers. Give h_0 and h_1
and say what happened to the first input's trace.
D2.3. Derive or justify: the w^10 law for the scalar
case, and name what replaces |w| for matrices.
D2.4. Compare: LSTM gates vs gradient clipping as fixes
for the vanishing gradient. Which treats the cause?
D2.5. Research critique: "Transformers made recurrence
obsolete." Attack with the C06 vs C09 cost comparison
and name the surviving niche.

## Analytical/quantitative (2)

Q1. Scalar net o = w2 relu(w1 x + b1) + b2, loss
0.5(o-y)^2, with w1 = 1, b1 = 0, w2 = 2, b2 = 0, x = 1,
y = 0. Without a computer: forward values, all four
gradients, and the finite-difference estimate of dw1
with h = 1e-7 (state the formula. give the value to 6
digits from the analytic structure).
Q2. Attention toy: Q = [[1,1]], K = [[1,0],[0,1],[1,1]],
V = [[2,0],[0,2],[1,1]], d = 2. Without a computer:
scores, stable softmax weights, output row, and a proof
that the weights sum to 1.

## Implementation/debug (1)

T1. A teammate's gradient check on the scalar relu net
fails: analytic dw1 = 0.0 but the two-sided finite
difference gives -4.35. Their net has w1 = 0.5, b1 =
-1.0, x = 2.0 (so z = 0 exactly), w2 = 1.5, y = 3.0.
(a) Reproduce: compute z, the analytic subgradient, and
the finite-difference value. (b) Diagnose in one
sentence. (c) State the fix: where should a gradient
check never be run, and what is the correct check at a
kink?

## Scenarios (2)

S1. Your RNN trains fine on sequences of length 20 but
diverges (loss NaN) at length 200 with the same weights.
Walk through your diagnosis: what do you measure first,
what are the two candidate causes, and what is the
cheapest fix to try before changing the architecture?
S2. A transformer inference service OOMs at 32k tokens
but fits at 4k. Name the term that blows up, give its
scaling, and propose two fixes with their trade-offs
(one exact, one approximate).

## Research critique (1)

R1. "Our 200-layer MLP trains fine, so depth needs no
special care." Attack with the C01 collapse argument
and the C11/C12 stabilization story, then state the
two mechanisms that actually let deep nets train.
