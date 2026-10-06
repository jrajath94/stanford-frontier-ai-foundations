# Lesson 03b, Second source block: GAN introduction to classifier-guided sampling

Unit: math-genai-U03 (source block: W2_L6, W2_L7, W2_T5,
W3L8). Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

This lesson follows the second coherent source block of the
playlist (SRC-04): W2_L6 "Generative adversarial networks:
introduction", W2_L7 "Generative adversarial networks:
formulation", W2_T5 "Tutorial: implementation of generative
adversarial network", and W3L8 "GANs as classifier-guided
generative sampler". Mapping is title-level only: no
transcript was inspected (source_gaps.md G2), so every claim
about the instructor's treatment stays at the title
boundary. The examples below are original toys, not lecture
reproductions. Source attribution for the leaf concepts is
PENDING. All numbers computed 2026-10-06, numpy 1.26.4,
float64, seed 0 where RNG is used.

Playlist-order note: W2_T5 sits between W2_L7 and W3L8 in
the playlist, and W3L8 opens Week 3 while the GAN block
started in Week 2. Title order is a playlist fact, not a
course claim. The four titles form one logical block: idea,
formulation, implementation, reinterpretation.

## Scope and objectives

Scope: what the four titles name. The GAN idea (W2_L6). The
minimax formulation (W2_L7). The implementation tutorial
(W2_T5). The classifier-guided sampler view (W3L8).

Objectives: after this block the learner can state the GAN
idea in one sentence, write the minimax formulation, list
what a minimal implementation must contain, and explain the
classifier-guided sampler reinterpretation on the toy.

Dependencies: U03 lesson C01-C05. This block re-covers the
same ground from the source titles' angle. It adds
provenance, not new mechanisms.

---

## SB01, W2_L6 Generative adversarial networks: introduction

Motivating question: what does the title promise as one
idea?

Start from zero. "Generative": the goal is samples from
p_data. "Adversarial": the training signal comes from an
opponent, not a likelihood. "Networks": both players are
neural nets.

The one idea. Two networks train each other: a generator
that fakes and a discriminator that detects. No density is
ever written down. On the toy: p_data = N(0,1), p_g =
N(1,0.5), D*(0) = 0.7870, D*(1) = 0.2327. The introduction
title promises this setup before any equation.

What the title does not say (title boundary). Which
examples, which history, and which motivation the lecture
uses are not inspected. The toy is authored.

Assessment: E01. Keys in
lessons/u03/keys-source-block-3b.md.

---

## SB02, W2_L7 Generative adversarial networks: formulation

Motivating question: what does "formulation" add to the
introduction?

Start from zero. The idea needs an equation both players
can optimize. The formulation title promises exactly one
thing: the minimax value function.

The one idea. V(D, G) = E_{p_data}[log D(x)] + E_z[log(1 -
D(G(z)))], min over G, max over D. From it follow D* =
p_data/(p_data + p_g) and max_D V = 2 JS - 2, both derived
in U03-C03/C04 and numerically verified on the toy
(-1.3537 both ways, identity error 4.4e-16).

Why the title matters for the course. This is the equation
the rest of the GAN weeks build on: saturation (W4L10) is
a property of this V, the Wasserstein replacement (W4L11)
is a repair of this V, and Bi-GAN (W4L13) generalizes this
V to pairs.

What the title does not say (title boundary). Whether the
lecture derives D*, states the JS connection, or discusses
the non-saturating variant is not inspected. The
derivations here are authored.

Assessment: E02-E03. Keys in
lessons/u03/keys-source-block-3b.md.

---

## SB03, W2_T5 Tutorial: implementation of generative
adversarial network

Motivating question: what must a minimal implementation
contain?

Start from zero. The formulation is math. The tutorial
title promises code. A minimal implementation needs five
parts: (1) a generator net G(z), (2) a discriminator net
D(x) with a sigmoid output, (3) the two losses (binary
cross-entropy on the two piles for D. Minimax or
non-saturating for G), (4) the alternating update loop
with k D-steps per G-step, (5) the eps clip on D's output
(C07).

The one idea. The tutorial is the formulation made
runnable. The U03-C02 pseudocode is the authored minimal
reference: sample two piles, ascend on D k times, descend
on G once, repeat. The failure modes it must guard: D
outputs leaving (0,1) (the keys' debug task), the log
base (bits, not nats), and k = 0 (frozen D, C02 failure
case).

What the title does not say (title boundary). Framework,
architecture sizes, dataset, hyperparameters, and run
results are not inspected. No tutorial code was executed.
The five-part list is authored from the formulation.

Assessment: E04. Keys in
lessons/u03/keys-source-block-3b.md.

---

## SB04, W3L8 GANs as classifier-guided generative sampler

Motivating question: what does "classifier-guided" mean?

Start from zero. A classifier usually maps x to a label.
Here the "classifier" is the discriminator: it maps x to
"data" versus "fake". "Guided" says the generator's
training signal is exactly this classifier's gradient.

The one idea. Rewrite the generator's job: G follows
-dV/dG, and dV/dG flows through D. The discriminator is a
density-ratio estimator (C03: D* = p_data/(p_data+p_g),
so D*/(1-D*) = p_data/p_g), and its gradient points G
toward regions where the ratio favors data. "Sampler"
says the product is samples, not densities: the guided
object never becomes a formula for p_data.

On the toy: at x = 0, D*/(1-D*) = 0.7870/0.2130 = 3.69:
the ratio says data country. At x = 1, 0.2327/0.7673 =
0.30: fake country. G's gradient pushes its mass from
0.30 toward 3.69. The classifier guides the sampler.

What the title does not say (title boundary). Whether the
lecture connects this to score-based guidance (U09) or to
the f-GAN ratio view is not inspected. The ratio numbers
are authored.

Assessment: E05-E06 and ladder L01. Keys in
lessons/u03/keys-source-block-3b.md.

---

## Adjacent titles (title-level, handled in the main lesson)

These playlist titles belong to U03's leaf concepts and
are covered as authored content in lesson 03, not in this
block. Mapping stays at the title boundary.

- W3L9 "Deep Convolution GANs and Conditional GANs" ->
  C08 (conditional GAN), C09 (DCGAN).
- W4L10 "Saturation of GAN training" -> C05, C07.
- W4L13 "Bi-directional GANs" -> C10.
- W4L16 "Evaluation of Generative Models" -> C12.
- W4T7 "Tutorial: Implementation of Bi-GAN" -> C10
  implementation angle (no code inspected).
- W4T9 "Tutorial: Implementation of WGAN" -> reserved
  for U04 (not this block).

---

## Source-block exercises (questions. Answers in keys-source-block-3b.md)

E01. State the GAN idea in one sentence. Name the two
players and what each maximizes or minimizes.
E02. Write the minimax formulation. Label the inner max
and the outer min.
E03. From the formulation alone, derive D* in one line
per step (pointwise maximization).
E04. List the five parts of a minimal GAN implementation.
For each, name one silent failure it guards against.
E05. Explain "classifier-guided generative sampler" on
the toy: compute D*/(1-D*) at x = 0 and x = 1 and say
which direction G's mass should move.
E06. A classmate cites W2_T5 for the non-saturating loss.
Is the citation valid? Use the inspection-boundary rule.

## Deep oral ladder (questions. Answers in keys-source-block-3b.md)

L01. Define the GAN game in one sentence. Toy: V =
-1.3537, D*(0) = 0.7870. Derive D* and the JS identity.
Implement the V check. Compare title-level source claims
versus inspected evidence for W2_L6/W2_L7/W2_T5/W3L8.
Debug: a teammate trains G against a frozen D. Critique:
is the title "adversarial" still accurate? Design: what
artifact would promote the formulation row from PENDING
to source-confirmed?
