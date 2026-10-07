# keys.md, U07 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate.

## E01

z_j = w_j^{[1]T} x + b_j^{[1]}, a_j = ReLU(z_j),
out = w^{[2]T} a + b^{[2]}.

## E02

Without it, stacked linear layers collapse to one
linear map. The activation is the only source of
nonlinearity.

## E03

Given dJ/d(output), each module returns dJ/d(input)
and dJ/d(params) via the chain rule, using only local
information.

## E04

LN(alpha W x + alpha b) = LN(W x + b) for alpha > 0:
scaling the weights leaves the normalized output
unchanged.

## E05

All hidden units compute the same function and receive
the same gradient, so symmetry never breaks. Depth is
wasted.

## E06

dJ/d theta_i ~= [J(theta + h e_i) - J(theta - h e_i)]
/ (2h).

## E07

z = 0.5 - 1 + 1.5 = 1.0. a = 1.0. dJ/da = 1 - 3 = -2.
ReLU open: dJ/dz = -2. dJ/dw = -2 * [1, 2] = [-2, -4].
dJ/db = -2. dJ/dx = -2 * [0.5, -0.5] = [-1, 1].

## E08

(W2 (W1 x + b1) + b2) = (W2 W1) x + (W2 b1 + b2): one
linear map with W = W2 W1.

## E09

Diagnosis: dead ReLUs in the middle layers, likely
from too-large updates early in training. Fixes:
(1) lower the learning rate and retrain. (2) Switch
to LeakyReLU or add LayerNorm. (3) Re-initialize the
dead layers.

## E10

Team A ships: hand backprop is O(N) per step, finite
differences are O(N) forward passes per step. Team B
never finishes training. Team B's method is only a
checking tool.

## E11

Setup: MLPs with p in {1e3, 1e4, 1e5, 1e6}, time one
forward and one backward pass. Claim: the ratio stays
in [1.5, 4] across sizes. Falsified if it grows with
p.

## E12

Reference: two-layer MLP, tanh or ReLU hidden,
manual backward per SL-04, gradient check to 1e-7,
then Adam/SGD on the ring data. Checks: gradient
check passes before training. Train accuracy >= 0.95
within a few thousand steps.

## L01 key

(1) Upstream gradient in, local gradients out, via the
chain rule.
(2) See SL-04 worked example.
(3) dJ/dw = dJ/da * 1{z>0} * x.
(4) Code per SL-04. Finite-difference match to 1e-7.
(5) Backprop: ~2 passes. Finite differences: 2
passes per parameter.
(6) Causes: dead ReLUs (z <= 0 everywhere). The loss
does not depend on the parameter (disconnected).
(7) At kinks (ReLU at 0) and hard thresholds. Use
subgradients or smooth surrogates.
(8) Log per-module gradient norms. Find the first
inf. Check the module's forward inputs for overflow.

## L02 key

(1) Linear (m,d)->(m,). ReLU same shape. LayerNorm
same shape. Softmax (k,)->(k,). Conv1D (m,)->(m,).
(2) [-1.2247, 0, 1.2247].
(3) Mean and std both scale with alpha. The ratio
cancels.
(4) Forward: normalize, scale, shift. Backward:
through the statistics (they depend on z).
(5) BatchNorm uses batch stats (train/test differ).
LayerNorm uses feature stats (no mode split).
(6) Cause: BatchNorm in eval mode with stale stats,
or the mode flag wrong.
(7) Lost: the scale information of the activations.
the model cannot use raw magnitudes downstream.
(8) LLM: LayerNorm or RMSNorm (no batch dependence,
stable at scale).
