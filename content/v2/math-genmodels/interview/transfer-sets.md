# Transfer sets: changed scenarios

Course: math-genmodels. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice scenarios. Not actual employer
material. Full keys in interview/keys-transfer.md. Each set moves
one course mechanism into an unfamiliar setting. Closed-book
transfer, not recognition.

## T1: a flow with a uniform base

You built the U04 coupling flow on a Gaussian base. A new sensor
produces angles in [0, 2 pi). Your colleague swaps the base to
Uniform(0, 2 pi) and keeps the coupling layers.

Q1. What breaks first: the base log-density, the support, or the
invertibility? Name it and compute the base log-density at 1.0.
Q2. The coupling layer adds t(x1) = x1 to the second coordinate.
Show a concrete input where the output leaves the support, and
state what the density evaluator returns there.
Q3. Propose the minimal fix that keeps exact likelihood, and
state what you lose.

## T2: a GAN for medical rows with a privacy bar

A hospital wants synthetic patient rows (age, diagnosis code,
lab value). The compliance bar: no synthetic row may match a
real row on all three fields. You train a GAN.

Q1. Which course concept makes the GAN unable to certify this
bar from its loss, and what test would you run instead?
Q2. The discriminator hits 100 percent on held-out real/fake
pairs. State two readings of this number and which one kills
the project.
Q3. You switch to a VAE for its likelihood. Name the new
privacy-relevant failure the VAE introduces that the GAN did
not have.

## T3: diffusion on a phone with 4 steps

Your diffusion model needs 1000 DDPM steps. The phone allows 4
network calls per image.

Q1. Name the sampler you would reach for first and the
assumption it needs about the trained net.
Q2. The 4-step samples look washed out. State the mechanism of
the degradation in one sentence and propose the cheapest fix
that needs no retraining.
Q3. Your manager suggests distilling to a 1-step student.
State what you gain, what you lose, and the experiment that
decides.

## T4: DNA with two tokenizations

You model DNA sequences. Tokenization A uses single bases
(V = 4). Tokenization B uses 3-mers (V = 64). Both models train
to convergence.

Q1. Model B reports lower perplexity. State whether B is better,
with the arithmetic that decides.
Q2. Convert both models to bits per base. Show the formula.
Q3. A colleague proposes byte-level modeling "to avoid the
question". State what changes in the cost model at L = 8192.

## T5: posterior collapse in a new domain

You train a VAE on 64x64 robot camera frames. The KL term reads
0.0003 per dimension across all 32 latent dims. Reconstructions
are sharp.

Q1. Diagnose in one sentence: what did the model learn to do?
Q2. The reconstructions are sharp but samples from the prior
are garbage. Explain the asymmetry.
Q3. Propose two fixes from the course, one on the objective
and one on the architecture, and state how each changes the
rate-distortion tradeoff.

## T6: score model on bounded sensor data

A temperature sensor reports in [0, 100] Celsius. You train a
score model on the readings with Gaussian noise at many
scales.

Q1. At the smallest noise scale, what happens to the learned
score near 0 and near 100? Give the mechanism.
Q2. You report ODE likelihoods to compare two models. State
the hidden free parameter in your numbers and how it could
flip the ranking.
Q3. Propose the minimal data preprocessing that removes the
problem, and name what it costs.

## T7: WGAN with discrete outputs

You apply WGAN-GP to generate discrete tokens. The critic is an
MLP on token embeddings, the generator outputs softmax vectors
with straight-through sampling.

Q1. The Kantorovich dual needs a 1-Lipschitz critic. State
what "Lipschitz" means when the input space is discrete token
IDs, and why the gradient penalty is conceptually awkward
there.
Q2. Interpolations x_hat = eps x + (1 - eps) y mix a real and
a fake embedding. Name what distribution these points come
from and why the penalty there may not constrain the critic
where it matters.
Q3. Propose one alternative from the course that avoids the
adversarial game entirely for discrete data.

## T8: anomaly detection under a latency SLA

You must score 50k transactions/sec for anomalies. Two
candidates: a MAF flow and an autoregressive transformer, both
with exact likelihood.

Q1. State which direction each model uses for scoring and the
per-point serial cost of each at D = 128.
Q2. The SLA allows 20 ms per batch of 1024 on one GPU. Show
the arithmetic that picks the winner.
Q3. The winner scores well but misses a new fraud shape.
State which evaluation concept explains why the likelihood
did not warn you, and the test you add.

## T9: memorization audit for a face diffusion model

A vendor sells you a face diffusion model "trained on public
data". Your audit finds 3 generated faces with nearest
training distance below 0.5 pixels in a 256x256 space.

Q1. State the null hypothesis this finding rejects, in one
sentence.
Q2. The vendor replies that diffusion models cannot memorize
because they "learn the score, not the data". Attack this
reply with one mechanism from the course.
Q3. Design the audit you would run before signing, with the
sample count, the reference set, and the decision rule.

## T10: the vendor's FID win

A vendor reports FID 8.2 versus your in-house 11.7 and claims
the better generative model.

Q1. Construct the cheapest counterexample that keeps their
number and breaks the claim, using one concept from U08.
Q2. Name the two pieces of information their report omits
that a fair comparison needs.
Q3. You have one week and no human raters. Design the
evaluation suite you would run instead, with at least three
metrics and the decision rule.
