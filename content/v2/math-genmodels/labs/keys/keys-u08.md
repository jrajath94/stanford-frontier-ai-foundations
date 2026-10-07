# Lab answers: U08

Unit: math-genmodels-U08. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

A: -10.1073, B: -70.949 confirmed. Per-point gaps (A minus B):
-2.1: 31.68, 1.9: -1.82, -1.8: 27.08, 2.2: -1.91, 0.1: 5.80.
The point -2.1 contributes most: B assigns it near-zero density
(-33.85 nats) while A gives -2.16.

## E2

Exact -1.7371, gap 0.125, bound -1.8621 confirmed. Second VAE
with bound -1.70: you can conclude its ELBO beats the flow's
exact number, and nothing else. Its true likelihood lies in
[-1.70, infinity): it could be -1.65 (better) or -1.70. The
ranking is undecided until the gap is measured.

## E3

Radius 1.0: A precision 0.4708, recall 1.0. B precision 0.9555,
recall 0.5. Radius 0.5 (same seed-13 draws): A precision
0.2357, recall still 1.0. B precision 0.6685, recall still 0.5.
The shrink hurts precision for both (tighter manifold) and
favors neither on recall. it punishes the blurry model more in
absolute terms.

## E4

FID(truth, A) = 0 confirmed. Discrete law: mass 1/3 at
-2.5612, 0, +2.5612 (sqrt(6.375)): mean 0, variance 4.25, FID
0. Lesson: second moments cannot see shape. a perfect FID can
hide three spikes, one hump, or two modes.

## E5

NN distances 0.0066 versus 0.4666 confirmed. Noise sweep
(seed 13): 0.01 -> 0.0064, 0.1 -> 0.0284, 0.5 -> 0.0866, 1.0
-> 0.221, 2.0 -> 0.607. The screen stops flagging around noise
2.0, where the memorizer's NN distance exceeds the honest
model's 0.47: but at that noise the memorizer is no longer
memorizing, it is a bad sampler.

## E6

Mean 0.4835, std 0.0115 confirmed. Rival 0.490 on one seed:
gap 0.0065, less than one standard deviation (0.0115). Not
better: the difference is noise. Two standard errors span
+/- 0.023 around 0.4835. 0.490 sits inside.
