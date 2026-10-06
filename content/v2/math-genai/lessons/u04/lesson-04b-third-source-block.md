# Lesson 04b, Third source block: Wasserstein GANs and the WGAN tutorial

Unit: math-genai-U04 (source block: W4L11, W4T9). Date:
2026-10-06. Baseline: October 6, 2026.

## Source mapping

This lesson follows the third coherent source block of the
playlist (SRC-04): W4L11 "Wasserstein GANs" and W4T9
"Tutorial: Implementation of WGAN". Mapping is title-level
only: no transcript was inspected (source_gaps.md G2), so
every claim about the instructor's treatment stays at the
title boundary. The examples below are original toys, not
lecture reproductions. Source attribution for the leaf
concepts is PENDING. All numbers computed 2026-10-06, numpy
1.26.4, float64, seed 0 where RNG is used.

Playlist-order note: W4L11 sits at playlist position 17,
right after W4L10 "Saturation of GAN training" (position
16). W4T9 sits at position 25, after W4T8 "Tutorial:
Implementation of UDA" (position 24). Title order is a
playlist fact, not a course claim. The two titles form one
logical block: the Wasserstein idea, then its
implementation. The block reads as the course's answer to
the saturation lecture that precedes it.

## Scope and objectives

Scope: what the two titles name. The Wasserstein GAN idea
(W4L11). The WGAN implementation tutorial (W4T9). Plus the
adjacent titles placed at the title boundary: W4L12
"Inversion with GANs", W4L14 "GAN inversion via latent
regression", W4L15 "Domain Adversarial Networks", W4T8
"Tutorial: Implementation of UDA".

Objectives: after this block the learner can state the
WGAN idea in one sentence, list what a minimal WGAN
implementation must contain, and place each adjacent
title into the U04 leaf it belongs to.

Dependencies: U04 lesson C01-C09. This block re-covers
the same ground from the source titles' angle. It adds
provenance, not new mechanisms.

---

## SB01, W4L11 Wasserstein GANs

Motivating question: what does the title promise as one
idea?

Start from zero. "Wasserstein": the distance is the
transport cost W1, not JS. "GANs": the game stays, but
the discriminator becomes a critic with a real-valued
output and a Lipschitz constraint.

The one idea. Replace the JS objective with the W1
dual: max over 1-Lipschitz f of E_p[f] - E_q[f], min
over the generator. On the toy: P0 = delta_0, P_theta
= delta_1, the critic f(x) = -x gives gap 1.0 = W1,
and dW/dtheta = 1.0 where the JS gradient was 0. The
title promises this repair right after the saturation
lecture (W4L10): the W game is the course's answer to
dead gradients on disjoint supports.

Why the title matters for the course. Everything in
the U04 lesson hangs off this title: the dual (C03),
the critic (C04), the two enforcements (C05, C06),
and the training dynamics (C07, C08).

What the title does not say (title boundary). Which
enforcement the lecture teaches (clipping, penalty,
or both), which toy it uses, and whether it states
the Kantorovich-Rubinstein name are not inspected.
The dual derivation here is authored.

Assessment: E01-E02. Keys in
lessons/u04/keys-source-block-4b.md.

---

## SB02, W4T9 Tutorial: Implementation of WGAN

Motivating question: what must a minimal WGAN
implementation contain?

Start from zero. The W4L11 idea needs code. The
tutorial title promises the runnable form. A minimal
implementation needs six parts: (1) a generator net
G(z), (2) a critic net f(x) with a real-valued
output (no sigmoid), (3) the two losses (critic:
maximize E_p[f] - E_q[f]. Generator: minimize the
maximized gap), (4) the Lipschitz enforcement
(clip weights to [-c, c] after each critic step,
or add the gradient penalty on interpolated
points), (5) the alternating loop with n_critic
critic steps per generator step (the authored
reference uses 5), (6) the calibration diagnostic
(gap versus exact W1 on a reference pair, C07).

The one idea. The tutorial is the dual made
runnable. The differences from the vanilla GAN
tutorial (W2_T5) are exactly three: the critic
output is real-valued, not a probability. The
loss is the gap, not cross-entropy. The loop
carries an enforcement step the vanilla loop
lacks. Everything else (sampling, batches,
optimizers) is shared machinery.

What the title does not say (title boundary).
Framework, architecture sizes, dataset,
hyperparameters, n_critic value, and run results
are not inspected. No tutorial code was
executed. The six-part list is authored from the
dual formulation. The n_critic = 5 in the U04
reference loop is an authored choice, not a
claim about W4T9.

Assessment: E03-E04. Keys in
lessons/u04/keys-source-block-4b.md.

---

## Adjacent titles (title-level, handled in the main lesson)

These playlist titles belong to U04's leaf concepts
and are covered as authored content in lesson 04,
not in this block. Mapping stays at the title
boundary.

- W4L12 "Inversion with GANs" -> C09 (GAN
  inversion: z* = argmin ||G(z) - x||^2. The
  linear toy solves to z* = 1.5).
- W4L14 "GAN inversion via latent regression" ->
  C09 (train E on (G(z), z) pairs. The toy
  encoder gives E([1.5, 3.0]) = 1.5). Pairs
  with W4L12 as the two inversion routes:
  per-x optimization versus a learned inverse.
- W4L15 "Domain Adversarial Networks" -> C09
  (gradient reversal: the domain gradient flips
  sign, 0.5 to -0.5 at lambda = 1, so features
  shed domain signal).
- W4T8 "Tutorial: Implementation of UDA" ->
  C09 implementation angle (unsupervised domain
  adaptation: the domain-adversarial idea made
  runnable. No code inspected). Pairs with
  W4L15 the way W4T9 pairs with W4L11: idea,
  then implementation.

Playlist adjacency worth noting: W4L12 and W4L14
bracket W4L13 "Bi-directional GANs" (positions
18, 19, 20). Three consecutive titles on
inverting the generator: direct optimization,
joint encoder, learned regression. The
grouping is a playlist fact. What the lectures
share is not inspected.

---

## Source-block exercises (questions. Answers in keys-source-block-4b.md)

E01. State the WGAN idea in one sentence. Name
what replaces the discriminator and what
replaces the JS objective.
E02. On the toy, the JS gradient at theta = 1
is 0 and the W gradient is 1.0. Explain in two
sentences why the W4L11 title follows W4L10
in the playlist.
E03. List the six parts of a minimal WGAN
implementation. For each, name the vanilla-GAN
tutorial part it replaces or extends.
E04. A classmate cites W4T9 for n_critic = 5.
Is the citation valid? Use the
inspection-boundary rule.
E05. Place W4L12, W4L14, W4L15, W4T8 into
their U04 leaves. For each, give the one toy
number from lesson 04 that anchors it.
E06. W4L12, W4L13, W4L14 are consecutive in
the playlist. State what the grouping
suggests and what it does not prove.

## Deep oral ladder (questions. Answers in keys-source-block-4b.md)

L01. Define the WGAN game in one sentence. Toy:
gap 1.0 = W1 at theta = 1. Derive the dual
from the price story. Implement the gap with
the Lipschitz assert. Compare title-level
source claims versus inspected evidence for
W4L11/W4T9. Debug: a WGAN critic with a
sigmoid output. Critique: is the title
"adversarial" still accurate when the critic
is not a classifier? Design: what artifact
would promote the W4L11 row from PENDING to
source-confirmed?
