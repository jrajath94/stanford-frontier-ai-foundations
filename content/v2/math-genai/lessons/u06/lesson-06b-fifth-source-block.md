# Lesson 06b, Fifth source block: beta-VAE to VQ-VAE

Unit: math-genai-U06 (source block: W6L24, W6L25, W6T13).
Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

This lesson follows the fifth coherent source block of
the playlist (SRC-04): W6L24 "Beta-VAE", W6L25 "Vector
Quantized VAE (VQ-VAE)", and W6T13 "Tutorial:
Implementation of VQ-VAE". Mapping is title-level only:
no transcript was inspected (source_gaps.md G2), so
every claim about the instructor's treatment stays at
the title boundary. The examples below are original
toys, not lecture reproductions. Source attribution for
the leaf concepts is PENDING. All numbers computed
2026-10-06, numpy 1.26.4, float64, seed 0 where RNG is
used.

Playlist-order note: W6L24 and W6L25 are consecutive
in Week 6 (positions 38 and 39). The pairing reads as
the course's two answers to rate control: soft
(beta) then hard (quantization). W6T13 (position 40)
is the implementation tutorial for the VQ-VAE. Title
order is a playlist fact, not a course claim.

## Scope and objectives

Scope: what the three titles name. The beta-VAE
(W6L24). The VQ-VAE (W6L25). The VQ-VAE implementation
tutorial (W6T13).

Objectives: after this block the learner can state
each title's one idea in one sentence, explain why
W6L24 precedes W6L25 in the playlist, and list what a
minimal VQ-VAE implementation must contain.

Dependencies: U06 lesson C01-C12. This block re-covers
the same ground from the source titles' angle. It adds
provenance, not new mechanisms.

---

## SB01, W6L24 Beta-VAE

Motivating question: what does the title promise as
one idea?

Start from zero. "Beta-VAE": the VAE objective with
the KL term weighted by beta. The title promises the
soft rate knob: E_q[log p(x | z)] - beta KL(q || p).

The one idea. One scalar trades reconstruction
against latent use. On the toy (recon -0.7342 nats,
KL 0.1766 nats): beta = 1, 2, 4, 8 give ELBO
-0.9108, -1.0874, -1.4405, -2.1468 nats. Coverage
note: the row math-genai-U05-C08 teaches beta-VAE
under U05. This title sits in Week 6 and is placed
here as the bridge into discrete rate control.

Why the title matters for the course. It is the
soft answer to "what does latent information cost",
asked right before the hard answer (W6L25).

What the title does not say (title boundary).
Whether the lecture frames beta as disentanglement,
as collapse control, or as a Lagrangian is not
inspected.

Assessment: E01-E02. Keys in
lessons/u06/keys-source-block-6b.md.

---

## SB02, W6L25 Vector Quantized VAE (VQ-VAE)

Motivating question: what does the title promise as
one idea?

Start from zero. "Vector Quantized": whole vectors
snapped to codebook entries. "VAE": the
autoencoder structure with a discrete bottleneck.
The title promises the hard rate knob: K codes,
log2 K bits, no KL tax, no reparameterization.

The one idea. Replace the Gaussian latent with a
codebook lookup. The three losses (recon 0.05,
codebook 0.05, commitment 0.0125 on the toy)
replace the ELBO. The straight-through estimator
replaces reparameterization. EMA replaces the
KL-closed-form convenience.

Why the title matters. This is the unit's center:
C01 through C12 all hang off it. It is also the
course's answer to U05 mechanism B shell 7
(discrete latents break reparameterization).

What the title does not say (title boundary).
Codebook size, architecture, dataset, and whether
the lecture covers dead codes or the prior stage
are not inspected.

Assessment: E03-E04. Keys in
lessons/u06/keys-source-block-6b.md.

---

## SB03, W6T13 Tutorial: Implementation of VQ-VAE

Motivating question: what must a minimal VQ-VAE
implementation contain?

Start from zero. The W6L25 idea needs code. The
tutorial title promises the runnable form. A
minimal implementation needs seven parts: (1) the
codebook table (K, D), (2) the quantizer (pairwise
distances, argmin), (3) the straight-through
wrapper (z_e + sg[z_q - z_e]), (4) the three
losses with correct sg[.] placement, (5) the EMA
update with the dead-code guard, (6) the usage
counter (dead-code diagnostic), (7) the prior
fitting step over indices (stage 2).

The one idea. The tutorial is the discrete
bottleneck made runnable. The differences from
the VAE tutorial (W6T12) are exactly four: the
latent is an index, not a Gaussian sample. The
loss is three terms, not the ELBO. The gradient
trick is straight-through, not reparameterization.
The codebook learns by EMA, not by the KL closed
form. Everything else (encoder/decoder nets,
batches, optimizers) is shared machinery.

What the title does not say (title boundary).
Framework, K, architecture, dataset, and run
results are not inspected. No tutorial code was
executed. The seven-part list is authored from
the VQ-VAE formulation.

Assessment: E05-E06. Keys in
lessons/u06/keys-source-block-6b.md.

---

## The bridge: soft rate then hard rate

W6L24 then W6L25 is a deliberate pairing at the
title level: first price latent information with
a continuous knob (beta), then enforce it with a
discrete bottleneck (K). The lesson's C09 makes
the bridge quantitative: VQ's (bits, distortion)
table against the VAE's beta sweep. The grouping
is a playlist fact. What the lectures share is
not inspected.

---

## Source-block exercises (questions. Answers in keys-source-block-6b.md)

E01. State the beta-VAE idea in one sentence.
Give the toy ELBO at beta = 2 and beta = 8.
E02. Explain in two sentences why W6L24
precedes W6L25 in the playlist, using the
soft-then-hard reading.
E03. State the VQ-VAE idea in one sentence.
Name what replaces the ELBO's KL term and
what replaces reparameterization.
E04. Give the toy's three loss values. State
which loss moves the codebook.
E05. List the seven parts of a minimal VQ-VAE
implementation. For each, name the VAE
tutorial (W6T12) part it replaces.
E06. A classmate cites W6T13 for gamma =
0.99. Is the citation valid? Use the
inspection-boundary rule.

## Deep oral ladder (questions. Answers in keys-source-block-6b.md)

L01. Define the VQ-VAE in one sentence. Toy:
the distance table and the winner. Derive
why the argmin blocks gradients. Implement
the quantizer with the STE wrapper. Compare
title-level source claims versus inspected
evidence for W6L25/W6T13. Debug: encoder
weights frozen during training. Critique:
is "variational" still accurate when there
is no posterior approximation? Design: what
artifact would promote the W6L25 row from
PENDING to source-confirmed?
