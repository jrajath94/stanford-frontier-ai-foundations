# Diagnostic answer key

Score each item 0, 1, or 2 per the guide in `../prerequisites.md`.
Max 28.

## D1

`é` is U+00E9. UTF-8: two bytes, [195, 169]. Full marks: both bytes
as integers. Partial: one byte or the code point only.

## D2

```
best = max(counts.items(), key=lambda kv: kv[1])
```
`best` is `(pair, count)`. Full marks: returns the pair and its
count. Partial: returns only the pair or only the max count.

## D3

P(A|B) = P(B|A) P(A) / P(B). Full marks: exact form. Partial: Bayes
named but terms misplaced.

## D4

Shape (4, 3). FLOP count: 2*4*8*3 = 192. Full marks: shape and 192.
Partial: shape only.

## D5

dz/dx = f'(g(x)) * g'(x). Full marks: the product form with one line
of working. Partial: statement without the working.

## D6

Forward: save x and W (needed for the backward). Backward: dx =
dy W^T, dW = x^T dy, db = sum(dy over batch). Full marks: data flow
plus the saved tensors x and W. Partial: flow without saved tensors.

## D7

2*3*4 = 24 elements. Strides (12,4,1): element (i,j,k) sits at
12i+4j+k, which is the contiguous layout: contiguous, yes. Full
marks: 24 and contiguous with the reason. Partial: 24 only.

## D8

Perplexity is the exponential of the mean negative log probability
per token: PPL = exp(-(1/N) sum log p(x_i)). Full marks: sentence
and formula. Partial: one of the two.

## D9

Block: x -> norm -> attention(Q,K,V) -> +x residual -> norm ->
MLP -> +residual. Q,K,V are the attention projections, the residual
paths bypass each sublayer, norms sit before the sublayers
(pre-norm). Full marks: all labels. Partial: missing norm placement.

## D10

Exactly two singular values are nonzero (equivalently: U is m x 2,
Sigma is 2 x 2, V^T is 2 x n in the thin SVD). Full marks: the
two-nonzero-singular-values statement. Partial: "low rank" only.

## D11

w <- w - eta * grad_w. Full marks: the update with the learning
rate multiplying the gradient. Partial: sign error.

## D12

Intensity = 10 GFLOP / 1 GB = 10 FLOP/byte. Knee = 100/1 = 100
FLOP/byte. 10 < 100: memory-bound. Time >= max(10/100, 1/1) = 1 s
from bandwidth. Full marks: 10 FLOP/byte, knee 100, memory-bound.
Partial: verdict without arithmetic.

## D13

All-reduce, equivalently the reduce-scatter plus all-gather pair
that composes it. Full marks: all-reduce named (the pair accepted).
Partial: only broadcast or only reduce.

## D14

H(p,q) = -sum_x p(x) log q(x). It measures the mean surprise of q
when the truth is p: the extra bits versus the entropy H(p). Full
marks: formula and the "extra cost over entropy" meaning. Partial:
formula only.
