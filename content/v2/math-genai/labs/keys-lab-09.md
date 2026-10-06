# Lab keys 09, LLM inference and alignment mechanics

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4,
float64. Ground truth: compute_run5b.py. C04 numbers
are authored arithmetic.

## Task 1

(a) Predict: T=0.5 sharper (pmax ~0.83, low
entropy), T=2.0 flatter (pmax ~0.41, high
entropy).
(b) Measured: T=0.5: [0.8282, 0.1121, 0.0412,
0.0185], ent 0.8754. T=1.0: [0.5745, 0.2114,
0.1282, 0.0859], ent 1.6174. T=2.0: [0.4056,
0.2460, 0.1916, 0.1569], ent 1.9017.
(c) Kept [0, 1], renorm [0.7311, 0.2689, 0, 0].
(d) Temperature is the adventure dial, top-p is
the quality floor.

## Task 2

(a) 2 12 8 64 512 2 = 12582912 = 12.00 MiB.
Verified.
(b) n=128: 3145728, 512: 12582912, 2048:
50331648, 8192: 201326592. Ratios 4x per
doubling: linear.
(c) Prefill 3221225472, decode/token 6291456,
ratio 512 = n. Verified.
(d) Wrong: 6291456 (6.00 MiB). Consequence: the
deploy budget is half the truth, OOM at 2x the
planned context.

## Task 3

(a) s = 3.5/255 = 0.013725, zp = round(1.2/s) =
87. Verified.
(b) Max err 0.005882, mean 0.004167. Bound s/2
= 0.006863: holds.
(c) s = 20/255 = 0.078431, bound 0.039216:
6.7x the true bound (0.039216/0.005882).
(d) The range buys resolution: the ticks are
sized to the data, not to a guess.

## Task 4

(a) margin 0.9, P = 0.7109, loss 0.3412 nats.
Verified.
(b) (1.3, 0.5): min(0.65, 0.6) = 0.6000, the
clip binds. (0.5, -0.4): min(-0.2, -0.32) =
-0.3200, pessimism (the worse term wins).
(c) margin 0.07, loss 0.6588, implicit 0.0400
and -0.0300. Verified.
(d) 0.0995 bits. pi = pi_ref: 0.0.

## Task 5

(a) With bonus: r_A = 1.00, r_B = 1.20, B
wins. Without: 0.9 vs 0.7, A wins.
(b) [0.7109, 0.4750, 0.6225, 0.5250], mean
0.5834. Verified.
(c) 4 title-level: C06 (W12L52), C07 (W11L50),
C08 (W12L53), C09 (W11L51 adjacent). 7
authored bridges: C01-C05, C10, C11. 1 audit:
C12. 4+7+1 = 12.
(d) The model ships when the held-out BT
win-rate (or human eval) passes the gate AND
the audit supports the claims. The still-
tunable knob: the decoding pair (T, top-p)
and the guidance-scale analogues, the
training knobs (loss weights, beta) are
frozen.
