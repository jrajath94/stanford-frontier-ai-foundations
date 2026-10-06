# Lesson 05b, Fourth source block: latent variable models to VQ-VAE

Unit: math-genai-U05 (source block: W5L17-W5L20, W5T10,
W5T11, W6L21-W6L25, W6T12). Date: 2026-10-06. Baseline:
October 6, 2026.

## Source mapping

This lesson follows the fourth coherent source block of
the playlist (SRC-04): W5L17 "Introduction to latent
variable models", W5L18 "Evidence Lower Bound (ELBO)",
W5L19 "Gaussian Mixture Models: Expectation-Maximization
Algorithm", W5L20 "Variational Autoencoder (VAE)", W5T10
"Proof of Jensen's inequality", W5T11 "GMM", W6L21
"Training VAE: Reparameterization methods", W6L22
"Training VAE", W6L23 "Inference with a trained VAE",
W6L24 "Beta-VAE", W6L25 "Vector Quantized VAE (VQ-VAE)",
and W6T12 "Implementation of VAE". Mapping is
title-level only: no transcript was inspected
(source_gaps.md G2), so every claim about the
instructor's treatment stays at the title boundary.
The examples below are original toys, not lecture
reproductions. Source attribution for the leaf concepts
is PENDING. All numbers computed 2026-10-06, numpy
1.26.4, float64, seed 0 where RNG is used.

Playlist-order note: W5T10 "Proof of Jensen's
inequality" sits in Week 5 while W1L3/W1L4 are Week 1
(E-009: title order is a playlist fact, not a course
claim). W6L24 and W6L25 sit in Week 6. The coverage
rows place beta-VAE under U05 (math-genai-U05-C08) and
VQ-VAE under U06. This block covers all twelve titles
and forwards W6L25 and W6T13 to the U06 block. Title
order is a playlist fact, not a course claim.

## Scope and objectives

Scope: what the twelve titles name. The latent
variable idea (W5L17). The ELBO (W5L18). GMM and EM
(W5L19, W5T11). The VAE (W5L20). Jensen's proof
(W5T10). VAE training and reparameterization (W6L21,
W6L22). VAE inference (W6L23). Beta-VAE (W6L24).
VQ-VAE placed forward (W6L25). The VAE implementation
tutorial (W6T12).

Objectives: after this block the learner can state
each title's one idea in one sentence, place every
title into its U05/U06 leaf, and explain why W5T10
sits in Week 5.

Dependencies: U05 lesson C01-C12. This block re-covers
the same ground from the source titles' angle. It
adds provenance, not new mechanisms.

---

## SB01, W5L17 Introduction to latent variable models

Motivating question: what does the title promise as
one idea?

Start from zero. "Latent variable": a hidden z that
explains the visible x. "Models": the joint p(x, z)
= p(x | z) p(z). The title promises the setup that
every later title in the block builds on: hidden
causes, visible effects, and the intractable
marginal p(x) = int p(x, z) dz.

The one idea. Name the hidden variable, write the
joint, and face the integral. On the toy: z in R,
x in R^2, p(z) = N(0, 1), p(x | z) = N(W z, 0.25
I). The marginal has no closed form once the
posterior is unknown, which motivates W5L18.

Why the title matters for the course. Everything
in the U05 lesson hangs off this title: the triple
(C01), the bound (C02, C10), and the classical
bridge (GMM/EM, C-bridge).

What the title does not say (title boundary).
Which examples the lecture uses, whether it
starts discrete or continuous, and how it
motivates the integral are not inspected.

Assessment: E01. Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB02, W5L18 Evidence Lower Bound (ELBO)

Motivating question: what does the title promise as
one idea?

Start from zero. "Evidence": log p(x), the
log-likelihood of the data. "Lower bound": a
tractable quantity that sits below it. The title
promises the ELBO identity: log p(x) >= E_q[log
p(x | z)] - KL(q || p).

The one idea. Jensen turns the intractable log
of an integral into a tractable expectation
minus a KL price. On the toy: expected ELBO
-3.2382 nats <= log p(x) -1.9341 nats, gap
1.3040 nats.

Why the title matters. The ELBO is the VAE
training objective (C02) and the bound the
audit verifies (C10).

What the title does not say (title boundary).
Whether the lecture derives it via Jensen or
via the KL-to-posterior gap identity, and
which toy it uses, are not inspected. Both
derivations are authored here.

Assessment: E02. Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB03, W5L19 Gaussian Mixture Models: Expectation-Maximization Algorithm

Motivating question: what does the title promise
as one idea?

Start from zero. "Gaussian Mixture Models": data
from K Gaussians with unknown assignments. The
latent variable is discrete: which component
produced x_i. "Expectation-Maximization": alternate
between soft assignments (E-step) and parameter
updates (M-step), with monotone likelihood
improvement guaranteed.

The one idea. Discrete latents admit an exact
E-step: responsibilities gamma_ik. On the toy:
gamma_2 = 0.8802 at x = 0.3. One M-step moves
the means to -0.9987 and 0.9995 and raises the
log-likelihood from -3.8354547 to -3.8354470.

Why the title matters. EM is the classical
reference design for the VAE: exact E-step, no
amortization, closed-form M-step (lesson
C-bridge). It sets the bar the VAE approximates.

What the title does not say (title boundary).
The lecture's K, dataset, and initialization are
not inspected. The toy is authored.

Assessment: E03. Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB04, W5L20 Variational Autoencoder (VAE)

Motivating question: what does the title promise
as one idea?

Start from zero. "Variational": approximate the
posterior with q(z | x). "Autoencoder": encode
then decode. The title promises the triple
(prior, encoder, decoder) trained on the ELBO.

The one idea. Amortize the E-step into a neural
encoder, make sampling differentiable by
reparameterization, and train end to end. On the
toy: mu = 0.4, sigma^2 = 0.5, 1-sample ELBO
-0.9108 nats.

Why the title matters. This is the unit's
center: C01 through C08 all hang off it.

What the title does not say (title boundary).
Architecture, dataset, and whether the lecture
states the collapse failure are not inspected.

Assessment: E04. Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB05, W5T10 Proof of Jensen's inequality

Motivating question: why does a proof of Jensen
sit in Week 5?

Start from zero. Jensen: for concave f, f(E[X])
>= E[f(X)]. The ELBO derivation (W5L18) uses it
at the log-E to E-log step. The tutorial title
promises the proof itself.

The one idea. The proof is placed where it is
used: next to the ELBO, not next to the
f-divergence where it first appeared (W1L3).
Playlist position is pedagogy, not chronology
(E-009).

Why the title matters. It is the load-bearing
step of the ELBO proof (R26).

What the title does not say (title boundary).
The proof technique (tangent-line or induction)
is not inspected.

Assessment: E05. Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB06, W5T11 GMM

Motivating question: what does the tutorial
title promise?

Start from zero. A GMM tutorial makes the EM
idea runnable: implement the E-step
(responsibilities) and the M-step (weighted
means), watch the likelihood rise.

The one idea. EM in code: six lines (lesson
C-bridge). The authored reference uses the
two-component 1-D toy: log-likelihood
-3.8354547 -> -3.8354470 in one step.

What the title does not say (title boundary).
Framework, dataset, K, and run results are
not inspected. No tutorial code was executed.

Assessment: E05 (paired with SB05). Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB07, W6L21 Training VAE: Reparameterization methods

Motivating question: what does the title promise
as one idea?

Start from zero. "Training VAE": gradients of
the ELBO. "Reparameterization methods": the
plural suggests more than one way to get
gradients through samples.

The one idea. z = mu + sigma eps makes the
sample differentiable. The score-function
estimator is the fallback. On the toy: true
gradient 0.8, reparameterized SE 0.1345 vs
score SE 0.2622 at 64 samples.

Why the title matters. This is mechanism B:
without it the encoder cannot learn (C04).

What the title does not say (title boundary).
Which methods the plural covers (Gumbel?
implicit?) is not inspected. Only the
Gaussian path is authored here.

Assessment: E06. Keys in
lessons/u05/keys-source-block-5b.md.

---

## SB08, W6L22 Training VAE

Motivating question: what does the title promise
beyond W6L21?

Start from zero. W6L21 names the gradient
method. W6L22 is the training loop itself:
batches, the ELBO loss, the optimizer, the
diagnostics.

The one idea. The loop: encode, sample,
decode, ELBO, backprop, step. The authored
reference adds the collapse diagnostic
(C12): KL per dim and traversal checks that
the loss alone cannot see.

What the title does not say (title boundary).
Optimizer, schedule, batch size, and epochs
are not inspected.

---

## SB09, W6L23 Inference with a trained VAE

Motivating question: what does the title promise
as one idea?

Start from zero. "Inference with a trained
VAE": the model is fitted. Now use it. Encode
new x in one forward pass (amortized
inference), sample z from the prior and decode
for generation, traverse latents for
interpretation.

The one idea. Training fits the machine. Inference runs it. On the toy: encode four
points in four forward passes. Mean 1-sample
ELBO -1.9467 nats over the batch (C11).

What the title does not say (title boundary).
Which inference tasks the lecture demos are
not inspected.

---

## SB10, W6L24 Beta-VAE

Motivating question: what does the title promise
as one idea?

Start from zero. "Beta-VAE": the ELBO with a
KL weight beta. The title promises the
reconstruction-vs-KL trade controlled by one
knob.

The one idea. Objective: E_q[log p(x|z)] -
beta KL. On the toy: beta = 1, 2, 4, 8 give
-0.9108, -1.0874, -1.4405, -2.1468 nats.
Coverage note: the row math-genai-U05-C08
places beta-VAE under U05 even though the
title sits in Week 6.

What the title does not say (title boundary).
Whether the lecture frames beta as
disentanglement or as collapse control is
not inspected.

---

## Forwarded titles (title-level, handled in the U06 block)

- W6L25 "Vector Quantized VAE (VQ-VAE)" ->
  U06 (discrete latents, codebook,
  straight-through). Forwarded to the U06
  lesson and the U06 source block.
- W6T13 "Tutorial: Implementation of VQ-VAE"
  -> U06 implementation angle. Forwarded
  with W6L25 the way W6T12 pairs with
  W6L21/W6L22: idea, then implementation.

---

## Source-block exercises (questions. Answers in keys-source-block-5b.md)

E01. State the latent variable idea in one
sentence. Write the joint and the marginal
on the toy.
E02. State the ELBO identity. Name the step
where Jensen is used. Give the toy gap in
nats.
E03. Compute the responsibility of component
2 at x = 0.3 on the toy. State the EM
guarantee in one sentence.
E04. State the VAE idea in one sentence.
Name the three parts and the objective.
E05. W5T10 sits in Week 5 while Jensen
first appeared in Week 1. Explain the
placement in two sentences using E-009.
E06. State the reparameterization idea in
one sentence. Give the toy SE comparison
and the conclusion.

## Deep oral ladder (questions. Answers in keys-source-block-5b.md)

L01. Define the ELBO in one sentence. Toy:
the toy gap 1.3040 nats. Derive the
five-line proof. Implement the expected
ELBO. Compare title-level source claims
versus inspected evidence for W5L17/W5L18.
Debug: a VAE whose ELBO exceeds the grid
marginal on one sample. Critique: does the
title "evidence lower bound" promise too
much for the 1-sample estimate? Design:
what artifact would promote the W5L18 row
from PENDING to source-confirmed?
