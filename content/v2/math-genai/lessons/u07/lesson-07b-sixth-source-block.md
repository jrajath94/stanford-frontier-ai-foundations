# Lesson 07b, Sixth source block: DDPM formulation to proofs

Unit: math-genai-U07 (source block: W7L26, W7L27, W7T14,
W8L28-W8L33, W8T15, W8T16). Date: 2026-10-06. Baseline:
October 6, 2026.

## Source mapping

This lesson follows the sixth coherent source block of
the playlist (SRC-04): W7L26 "Denoising Diffusion
Probabilistic Models (DDPMs)", W7L27 "DDPM:
Formulation", W7T14 "U-Net", W8L28 "ELBO for DDPM: Part
1", W8L29 "ELBO for DDPM: Part 2", W8L30 "Optimization
of DDPM loss", W8L31 "ELBO Equivalence", W8L32
"Training of DDPM", W8L33 "Inference in DDPM", W8T15
"Tutorial: Implementation of DDPM", and W8T16 "Proofs".
Mapping is title-level only: no transcript was
inspected (source_gaps.md G2), so every claim about the
instructor's treatment stays at the title boundary. The
examples below are original toys, not lecture
reproductions. Source attribution for the leaf concepts
is PENDING. All numbers computed 2026-10-06, numpy
1.26.4, float64, seed 0 where RNG is used.

Playlist-order note: W7L26 and W7L27 sit in Week 7
(positions 41-42), W7T14 follows (position 43), then
W8L28 through W8L33 in Week 8 (positions 44-49),
W8T15 and W8T16 (positions 50-51). The block reads as
the course's full DDPM arc: idea, formulation,
network, ELBO in two parts, loss optimization,
equivalence, training, inference, implementation,
proofs. Title order is a playlist fact, not a course
claim.

## Scope and objectives

Scope: what the eleven titles name. The DDPM idea
(W7L26). The formulation (W7L27). The U-Net (W7T14).
The ELBO parts 1-2 (W8L28, W8L29). Loss optimization
(W8L30). ELBO equivalence (W8L31). Training (W8L32).
Inference (W8L33). Implementation (W8T15). Proofs
(W8T16).

Objectives: after this block the learner can state
each title's one idea in one sentence, place every
title into its U07 leaf, and explain why the ELBO
takes two lectures.

Dependencies: U07 lesson C01-C12. This block re-covers
the same ground from the source titles' angle. It adds
provenance, not new mechanisms.

---

## SB01, W7L26 Denoising Diffusion Probabilistic Models (DDPMs)

Motivating question: what does the title promise as
one idea?

Start from zero. "Denoising": the model learns to
remove noise. "Diffusion": noise is added gradually
through a Markov chain. "Probabilistic models": the
reverse steps are distributions. The title promises
the whole idea: destroy data gradually with fixed
noise, learn to reverse it.

The one idea. A fixed forward chain turns data into
noise. A learned reverse chain turns noise back into
data. On the toy: beta_1 = 0.0001 to beta_100 =
0.02, alpha_bar_50 = 0.77718, x_50 = 1.99917538
from x_0 = 2.0.

Why the title matters for the course. Everything in
the U07 lesson hangs off this title: the chain
(C01-C04), the reverse (C05, C06, C11, C12), and
the objective (C07, C08, C10).

What the title does not say (title boundary).
Schedule choice, network architecture, and dataset
are not inspected.

Assessment: E01. Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB02, W7L27 DDPM: Formulation

Motivating question: what does the title promise
beyond W7L26?

Start from zero. W7L26 is the idea. W7L27 is the
mathematics: the forward kernels q(x_t |
x_{t-1}), the closed form q(x_t | x_0), the
posterior q(x_{t-1} | x_t, x_0) with its
coefficients, and the reverse parameterization.

The one idea. The formulation is the algebra that
makes the idea trainable: direct noising (C04),
the posterior triple (0.00960075, 0.03956209,
0.96013574 at t = 50), and the eps_hat
substitution (C06).

What the title does not say (title boundary).
Which derivations the lecture shows line by line
is not inspected. The completing-the-square is
authored here.

Assessment: E02. Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB03, W7T14 U-Net

Motivating question: what does the tutorial title
promise?

Start from zero. The reverse step needs eps_hat(x_t,
t): a network that takes a noisy input and a
timestep and predicts the noise. The title promises
the U-Net: the architecture that does this job.

The one idea. The U-Net is the function inside
every reverse step. This lesson treats it as a
black box (the "not yet understood" list forwards
it to U08). The tutorial title promises its
construction.

What the title does not say (title boundary).
Architecture details, conditioning method, and
code are not inspected. No tutorial code was
executed.

Assessment: E03. Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB04, W8L28 ELBO for DDPM: Part 1

Motivating question: why does the ELBO take two
lectures?

Start from zero. Part 1 sets up the variational
bound for the chain: the joint p(x_{0:T}), the
approximate posterior q(x_{1:T} | x_0), and the
telescoping sum into L_T, the middle KLs, and
L_0.

The one idea. The chain structure turns one big
ELBO into T+1 small terms. On the toy: L_T =
0.7713 nats, one middle L_49 = 0.0002337 nats,
L_0 = 0.0050005 nats.

What the title does not say (title boundary).
Where the lecture splits part 1 from part 2 is
not inspected. The split here (setup vs
simplification) is authored.

Assessment: E04. Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB05, W8L29 ELBO for DDPM: Part 2

Motivating question: what does part 2 add?

Start from zero. Part 2 finishes the ELBO: each
middle KL reduces to a squared mean difference
(same variance both sides), and the sum becomes
the weighted noise-prediction objective.

The one idea. The KLs collapse into ||eps -
eps_hat||^2 terms with coefficients. The
coefficients are what W8L30 then drops.

What the title does not say (title boundary).
The lecture's exact simplification path is not
inspected.

---

## SB06, W8L30 Optimization of DDPM loss

Motivating question: what does the title promise
as one idea?

Start from zero. "Optimization of DDPM loss": how
the objective is actually trained. The title
promises the step from the weighted ELBO to the
simplified loss: drop the coefficients, sample t
uniformly, minimize ||eps - eps_hat||^2.

The one idea. The training loss is the
simplification, not the bound. On the toy five
pairs: losses 0.01, 0.09, 0.0025, 0.04, 0.01,
mean 0.0305.

Why the title matters. It licenses the gap
between the derived ELBO and the trained loss
(mechanism C shell 7).

What the title does not say (title boundary).
Whether the lecture justifies the simplification
empirically or theoretically is not inspected.

Assessment: E05. Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB07, W8L31 ELBO Equivalence

Motivating question: what does the title promise
as one idea?

Start from zero. "ELBO Equivalence": two forms
of the bound coincide. The natural reading at
the title level: the DDPM ELBO equals the
standard VAE ELBO applied to the chain
latents x_{1:T}, or two derivations of the
same decomposition agree.

The one idea. The chain ELBO is not a new kind
of bound. It is the familiar ELBO (U05, W5L18)
with the latents being the noisy states. The
equivalence anchors DDPM in the course's
earlier variational material.

What the title does not say (title boundary).
Which two forms the lecture equates is not
inspected. The VAE-ELBO reading is the
title-level inference, marked as such.

Assessment: E05 (paired with SB06). Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB08, W8L32 Training of DDPM

Motivating question: what does the title promise
as one idea?

Start from zero. "Training of DDPM": the loop.
Sample x_0, sample t uniform, sample eps, form
x_t, predict eps_hat, squared error, backprop.

The one idea. One network evaluation per
example per step. The t sampler (C09) and the
direct noising (C04) make it O(1) per example.
The authored reference loop is train_step in
C10.

What the title does not say (title boundary).
Optimizer, batch size, and epochs are not
inspected.

---

## SB09, W8L33 Inference in DDPM

Motivating question: what does the title promise
as one idea?

Start from zero. "Inference in DDPM": the
sampling loop. From x_T ~ N(0, 1), iterate the
reverse Gaussian T times.

The one idea. Training is cheap per step. Inference costs T network evaluations per
sample. On the toy: three steps give
1.75690526, 1.74406522, 1.82846911.

What the title does not say (title boundary).
Whether the lecture covers fast sampling
(DDIM) here or in Week 9 is not inspected.
DDIM titles (W9L38, W9L39) suggest Week 9.

Assessment: E06. Keys in
lessons/u07/keys-source-block-7b.md.

---

## SB10, W8T15 Tutorial: Implementation of DDPM

Motivating question: what must a minimal DDPM
implementation contain?

Start from zero. The tutorial title promises
the runnable form. A minimal implementation
needs eight parts: (1) the schedule
(precomputed betas, alphas, alpha_bars),
(2) q_sample (direct noising), (3) the
timestep sampler, (4) the noise predictor
network (U-Net, black box here), (5) the
simplified loss, (6) the posterior
coefficient tables, (7) the reverse mean
from eps_hat, (8) the sampling loop with
the t = 1 no-noise rule.

The one idea. The tutorial is the
formulation made runnable. Every part maps
to one U07 leaf: schedule -> C02, q_sample
-> C04, sampler -> C09, loss -> C10,
coefficients -> C05, reverse mean -> C06,
loop -> C12, variance choice -> C11.

What the title does not say (title boundary).
Framework, T, architecture, dataset, and run
results are not inspected. No tutorial code
was executed. The eight-part list is authored
from the formulation.

---

## SB11, W8T16 Proofs

Motivating question: what does the title promise?

Start from zero. "Proofs": the derivations
written out fully. The natural title-level
reading: the posterior derivation
(completing the square), the ELBO
telescoping, and the closed forms, proved
rather than stated.

The one idea. Proofs turn the formulation's
claims into checked algebra. This lesson's
check battery (mechanism D) is the
computational counterpart: each assert
guards one proof step.

What the title does not say (title boundary).
Which proofs the tutorial covers is not
inspected.

---

## Source-block exercises (questions. Answers in keys-source-block-7b.md)

E01. State the DDPM idea in one sentence.
Give the toy alpha_bar_50 and x_50.
E02. State what a formulation adds to an
idea, in one sentence. Name the three
algebraic results of W7L27.
E03. State the U-Net's job in one
sentence. Name its two inputs and its
output.
E04. State the ELBO decomposition in one
sentence. Give the toy L_T in nats and
bits.
E05. State the simplified loss in one
sentence. Give the toy mean over the five
pairs. Then state the title-level reading
of "ELBO Equivalence".
E06. State the sampling loop in one
sentence. Give the toy's three x values.
Name the per-sample cost.

## Deep oral ladder (questions. Answers in keys-source-block-7b.md)

L01. Define DDPM in one sentence. Toy:
x_50 = 1.99917538. Derive q(x_t | x_0).
Implement q_sample. Compare title-level
source claims versus inspected evidence
for W7L26/W7L27. Debug: variance explodes
with t. Critique: the schedule is fixed. Should it be learned? Design: what
artifact would promote the W7L27 row from
PENDING to source-confirmed?
