# keys-cs229s.md: diagnostic answer keys

Date: 2026-10-06.

## D01

64 numbers. Transpose over last two axes: (2, 8, 4).

## D02

Broadcasting aligns trailing axes and stretches size-1
axes. (4, 1) + (8,) -> (4, 8).

## D03

theta := theta - eta * dL/dtheta. theta: parameter.
eta: learning rate. dL/dtheta: gradient of the loss.

## D04

2F: one F for gradients w.r.t. inputs, one F for
gradients w.r.t. weights (dense layer).

## D05

Perplexity is exp(mean negative log-likelihood). The
uniform model has perplexity 1000.

## D06

Teacher forcing feeds the true previous tokens as
inputs during training instead of the model's own
predictions.

## D07

d = 4. Q: (1, 2, 4, 4). Scores: (1, 2, 4, 4). Merged
output: (1, 4, 8).

## D08

The cache stores past keys and values per layer. A new
step needs only its own K/V, the rest are read, not
recomputed.

## D09

I = FLOPs/bytes = 1 FLOP/byte. Below ridge 150:
memory-bound.

## D10

Registers, shared memory/L1, L2, HBM.

## D11

A condition number of 1e6 means the system is
ill-conditioned: small input errors can grow ~1e6x in
the solution.

## D12

A low-rank approximation keeps the top r singular
components. It saves work when the spectrum decays
fast (most energy in few components).
