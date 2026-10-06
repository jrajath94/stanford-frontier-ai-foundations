# Lab keys 08, score and autoregressive mechanics

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4,
float64, seed 0 where RNG is used. Ground truth:
compute_run5b.py.

## Task 1

(a) Predict: s(0) = 0 (symmetry), s(1.5) = 0
(mode), s(1.0) > 0 (uphill right).
(b) Measured: 0, 1.999926, ~0 (< 1e-6),
-1.999926.
(c) Zero score marks stationary points: modes,
minima, and saddles. The valley at 0 is a
minimum of p, hence a stationary point, not a
mode.
(d) Wrong var 0.5: s(1.0) = 0.98516426 vs
true 1.999926. The magnitude roughly halves
(the denominator doubles and the weights
shift), direction unchanged here. The error:
underconfident arrows, slower Langevin climb,
wrong stationary distribution.

## Task 2

(a) (1.5 - 1.7)/0.04 = -5.0. Verified.
(b) 0.0, 0.039759, 0.060272, 0.354732,
0.608576, 0.617063.
(c) a = 0.1: mean 0.1610, std 1.5873, frac>
0.56, crossings 56. a = 0.5: mean 0.0428,
std 1.6385, frac> 0.5173, crossings 414.
True std 1.5811.
(d) Not broken: the stationary std matches
(1.5873 vs 1.5811), so the chain targets the
right distribution. It faces a mixing limit: 56
crossings cannot balance two modes in 20000
steps, hence mean 0.161. The fix is an annealed
schedule
or larger a, not a bugfix.

## Task 3

(a) 1.73697 bits, perplexity 1.82574.
(b) 3.2675 bits, per token 1.6338 vs 0.8685,
gap 0.7653 bits/token.
(c) The one-token version scores only the
second token given a sampled first token, the
two-token version also scores the sampled
first token itself. Different random objects,
different numbers.
(d) Row 0 = [0.5, 0.5, 0] (on the toy scores):
position 0 sees x_1. Training loss for
predicting x_1 collapses, generation breaks
because the future does not exist at test
time.

## Task 4

(a) X = [[1, 0, 1, 0], [0.1, 1.1, 0.1, 1.1],
[1.2, 1.2, 0.2, 0.2]].
(b) O rows: [1, 0, 1, 0], [0.3214, 0.8294,
0.3214, 0.8294], [0.8617, 0.8964, 0.3581,
0.3928]. O[0] == X[0] verified.
(c) max |diff| in row 1: 0.0983.
(d) P buys order-sensitivity: identical tokens
at different positions get different outputs.
The mask buys the chain rule: each position
sees only its past, so one forward pass
trains all conditionals honestly.

## Task 5

(a) T=0.5: [0.8282, 0.1121, 0.0412, 0.0185],
ent 0.8754. T=1.0: [0.5745, 0.2114, 0.1282,
0.0859], ent 1.6174. T=2.0: [0.4056, 0.2460,
0.1916, 0.1569], ent 1.9017.
(b) Kept [0, 1], renorm [0.7311, 0.2689, 0,
0].
(c) 603979776 (0.60 GB), 2415919104 (2.42
GB), 9663676416 (9.66 GB). Ratios: 4x, 16x.
The 16x rule holds (n quadruples).
(d) Audio waveforms: continuous, ordered.
Score/diffusion fits continuity (no
arbitrary discretization), AR fits order
and gives exact likelihood. Pick: diffusion
for raw waveforms (order-free local
structure, continuous), AR for symbolic
audio (MIDI-like tokens). Justify with the
mixing cost (56 crossings) vs the O(n^2)
memory (0.60 GB at 1024).
