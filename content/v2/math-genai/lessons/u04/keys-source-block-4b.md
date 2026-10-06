# Answer keys, lesson 04b (third source block)

Date: 2026-10-06. Ground truth: compute_run4.py.
Title-level mapping only. No transcript inspected.

## E01

The WGAN trains the generator against the
Wasserstein-1 dual instead of the JS game: a
real-valued 1-Lipschitz critic replaces the
discriminator, and the maximized gap
E_p[f] - E_q[f] replaces the minimax value V.

## E02

W4L10 "Saturation of GAN training" names the
dead-gradient problem: JS is flat on disjoint
supports. W4L11 "Wasserstein GANs" is the
repair: W1 = |theta| has gradient 1.0 where JS
has 0. The playlist order (16 then 17) reads
as problem then answer. That reading is a
playlist fact, not a claim about lecture
content.

## E03

(1) Generator G(z): same as vanilla. (2)
Critic f(x), real output: replaces the
sigmoid discriminator. (3) Gap losses:
replace cross-entropy. (4) Lipschitz
enforcement (clip or penalty): new, no
vanilla counterpart. (5) Alternating loop
with n_critic: extends the k-step loop.
(6) Calibration diagnostic: new, no vanilla
counterpart.

## E04

Not valid. W4T9 was never inspected beyond
its title, so no hyperparameter can be cited
to it. n_critic = 5 is an authored choice in
the lesson's reference loop. The inspection
boundary forbids the citation.

## E05

W4L12 -> C09, anchored by z* = 1.5. W4L14 ->
C09, anchored by E([1.5, 3.0]) = 1.5. W4L15
-> C09, anchored by the gradient flip 0.5 to
-0.5. W4T8 -> C09 implementation angle. No
code inspected, so no number anchors it.

## E06

The grouping suggests the course treats
generator inversion as one topic with three
routes: direct optimization (W4L12), joint
encoder (W4L13), learned regression (W4L14).
It does not prove the lectures share
notation, toys, or conclusions. Playlist
adjacency is not content equivalence.

## L01

The WGAN game is min over generators of the
max 1-Lipschitz critic gap. Toy: f(x) = -x
gives gap 1.0 = W1. Price story: prices, no
arbitrage, max profit. Implement the gap
with an assert that the critic's slope bound
holds on a grid. Title-level claims: the
titles promise the idea and the tutorial.
inspected evidence: none beyond the titles.
Debug: a sigmoid output makes the critic a
classifier again and the gap is no longer
the dual. Critique: "adversarial" still
fits: the critic still maximizes what the
generator minimizes. Design: an inspected
transcript or opened notebook showing the
dual objective and an enforcement step would
promote the row.
