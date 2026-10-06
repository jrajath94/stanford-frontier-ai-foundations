# Lesson 08b, Seventh source block: alternate interpretations to guided DDPM

Unit: math-genai-U08. Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

Title-level treatment of nine playlist entries (SRC-04):
W9L34 alternate interpretations of DDPMs, W9L35 DDPMs as
score-predictors, W9L36 Guided Difusion Models (playlist
spelling), W9L37 Latent Diffusion Models, W9L38
Denoising Difusion Implicit Models (DDIMs) (playlist
spelling), W9L39 Inference in DDIM, W9T17 Implementation
of DDPM Noise estimation, W9T18 Implementation of DDIM,
W9T19 Implementation of Guided DDPM. No transcript
inspected (source_gaps.md G2). Playlist order places
W9T17-T19 after W10L46, they are paired here logically
with U08 and the placement is recorded, following the
E-009 precedent. Each section states what the title
promises, the question it answers, where it lands in
the lesson, and what to verify when watching. Nothing
below claims lecture content.

## Scope and objectives

Scope: the nine titles above, mapped to U08 leaf
concepts C01-C12. Objectives: after this block the
learner can say what each title should teach, point to
the lesson section that prepares them for it, and list
the verification questions to ask while watching.

## SB01, W9L34 alternate interpretations of DDPMs

The title promises: more than one mathematical view
of the same DDPM object. The question it answers:
is a DDPM a latent-variable model, a score model,
or something else? Lands in: C07 (mean/score
estimates) and U09-C01 to C04 (score-based view).
Preparation: the C07 Tweedie/score conversion (s_hat
= -0.84738932 at t = 50) and the U07 ELBO
derivation. Verify when watching: which
interpretations are presented (ELBO, score
matching, SDE?), whether the lecturer shows the
algebra connecting them, and which one motivates
the sampling rule.

## SB02, W9L35 DDPMs as score-predictors

The title promises: the DDPM epsilon network read
as a score network. The question: what does
eps_hat estimate, really? Lands in: C07 and
U09-C01/C02. Preparation: s_hat = -eps_hat /
sqrt(1 - ab_t), computed both ways to 1e-12
agreement. Verify when watching: whether the
lecture states the identity or derives it, and
whether denoising score matching is named as the
training objective behind the epsilon loss.

## SB03, W9L36 Guided Difusion Models

The title promises: steering diffusion samples
with guidance. The question: how does the label
change the reverse step? Lands in: C08 (conditional
models) and C09 (classifier and classifier-free
guidance). Preparation: the two formulas (eps_hat
- s sqrt(1 - ab_t) grad log p(y | x_t), eps_u + g
(eps_c - eps_u)) and the toy numbers (0.25838859,
g = 0..3 table). Verify when watching: which
guidance method is taught first, whether the
Bayes-rule argument for classifier guidance is
shown, and whether the conditioning-dropout
training trick is mentioned.

## SB04, W9L37 Latent Diffusion Models

The title promises: diffusion in a compressed
latent space instead of pixels. The question:
why not diffuse the pixels directly? Lands in:
U06 (VQ-VAE/latent compression) composed with
U07/U08 (diffusion), touching C11 (runtime:
smaller latents cost less). Preparation: the
U06 rate/distortion tradeoff and the C04 FLOP
count (156672 per eval on the toy, latents
shrink H and W, which dominate the count).
Verify when watching: which autoencoder
provides the latents, whether the diffusion
math changes in latent space (it should not),
and how the decoder is trained. Note: this
title composes two units, no transcript
confirms the composition details.

## SB05, W9L38 Denoising Difusion Implicit Models (DDIMs)

The title promises: the DDIM construction.
The question: how do big jumps stay exact?
Lands in: C01 (non-Markovian construction)
and C02 (stochastic/deterministic). Preparation:
the marginal variance identity, the sigma
formula, the 100 -> 90 jump numbers
(1.60480905 -> 1.71491474), and the eta = 1
jump-variance check (0.15405473). Verify when
watching: whether the lecturer proves the
marginal match or states it, and whether eta
= 1 is connected to the DDPM posterior.

## SB06, W9L39 Inference in DDIM

The title promises: the sampling procedure.
The question: what is the exact loop? Lands
in: C02, C03 (step schedules), C12 (the
battery). Preparation: the 10-step
deterministic trajectory, tau choices, and
the six asserts. Verify when watching:
whether the loop is written with the
subsequence or the full chain, and whether
any correctness checks are shown.

## SB07, W9T17 Implementation of DDPM Noise estimation

The title promises: code for the noise
estimation piece. The question: what does
the training step look like in code? Lands
in: U07-C10/C12 and C07 (the eps target).
Preparation: q_sample, the uniform t draw,
the ||eps - eps_hat||^2 loss. The
repository (SRC-05) lists
IITM_DGM_DDPM_Noise_Estimate.ipynb, which
the title suggests covers this, contents
uninspected (G6). Verify when watching:
the timestep indexing convention (E-018:
1-indexed t with ab[t-1]), the loss
reduction (mean over batch), and the
noise-level sampling.

## SB08, W9T18 Implementation of DDIM

The title promises: code for DDIM sampling.
The question: how is the subsequence loop
written? Lands in: C01-C03, C12. The
repository lists IITM_DGM_DDIM.ipynb,
contents uninspected (G6). Preparation:
ddim_step, ddim_sigma, the tau build.
Verify when watching: how tau is chosen in
code, where eta enters, and whether the
implementation asserts the sigma bound.

## SB09, W9T19 Implementation of Guided DDPM

The title promises: code for guided
sampling. The question: how do the two
guidance methods differ in code? Lands in:
C09. The repository lists
IITM_DGM_DDPM_Guided_Diffusion.ipynb,
contents uninspected (G6). Preparation:
the cfg() blend and the classifier-gradient
term. Verify when watching: whether both
methods are implemented or one, how many
forward passes run per step under CFG, and
how g is exposed as a parameter.

## Source-block exercises (questions. Answers in keys-source-block-8b.md)

Q1. Map each of the nine titles to its U08
leaf concept(s). Flag any title whose
mapping is ambiguous from the title alone.
Q2. W9L34 promises alternate
interpretations. Name three candidate
interpretations consistent with the title,
and the lesson section that prepares each.
Q3. W9L37 composes U06 and U08. State
which U06 result and which U08 result the
composition needs, and what could break
at the seam (encode/decode round trip).
Q4. The playlist order puts W9T17-T19
after W10L46. Give two hypotheses for
why, and say which one the E-009
precedent supports.
Q5. For W9T17, write the three indexing
asserts you would want the tutorial to
show (t range, ab index, loss target).
Q6. For SB05, state the one-line
marginal contract and the number from
C01 that verifies it (0.55912725).

## Deep oral ladder (questions. Answers in keys-source-block-8b.md)

R1. Define the title boundary: what can
a title tell you and what can it not?
R2. Take W9L35: from the title alone,
predict the lecture's central equation.
R3. Check the prediction against C07:
does the lesson already contain that
equation? What would watching add?
R4. Compare W9L36 and W9T19: what does
the tutorial add that the lecture title
does not promise?
R5. Critique: "the nine titles prove the
course teaches DDIM deeply." Attack it
with G2.
