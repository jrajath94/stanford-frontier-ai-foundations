# Role bridges, U06 discrete latent modelling and VQ-VAE

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U06 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: the straight-through estimator's bias is
the research surface. The toy quantifies it: [0,
-1] vs [0, 0]. The open question is the bias/variance map across estimators (STE vs Gumbel-softmax
vs REINFORCE) on one fixed toy. The dead-code
phenomenon (usage [4, 3, 0, 1]) is a second
surface: when does capacity die, and do restarts
really recover it.

Practice questions: state the STE bias on the toy.
Design the estimator comparison from mechanism B
shell 9. Name the evidence that would promote
W6L25 from title-level to source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do
not transfer to papers.

## Research engineering

Bridge: the numerics are the job. The EMA guard
(NaN without it, stale vector with it), the
finite-difference true-gradient check ([0, 0]
exactly), and the controlled ablation protocol
(C12: one factor varied, seed fixed) are the
three implementation habits.

Practice questions: implement ema_update with the
guard and the battery asserts. Reproduce every
number via compute_run5a.py. Benchmark the O(K D)
scan against an ANN index at large K.

Residual gap: no GPU profiling, no distributed
training, no framework internals. The numpy
reference is not a production implementation.

## FDE (forward deployed engineer)

Bridge: the discrete bottleneck is the deployable
idea: countable codes are transmittable,
auditable, and billable (rate = log2 K bits).
The discovery question: what distortion does the
application tolerate? The handoff artifact is the
codebook plus the decoder plus the fitted prior:
without the prior the system is a compressor, not
a generator.

Practice questions: read the rate/distortion
table to pick K for a distortion budget. State
why effective K, not nominal K, goes in the
capacity plan.

Residual gap: no client discovery, no bandwidth
budgeting, no monitoring pipeline. The toy has
no deployment.

## ML

Bridge: evaluation of discrete generative models
needs the two-stage view: autoencoder quality
(distortion) and prior quality (sample match)
are separate. A good stage 1 with a uniform
prior samples dead codes: the lesson's C07
failure case.

Practice questions: name the two stages and one
metric per stage. Explain the uniform-prior
failure.

Residual gap: no real datasets, no generalization
study, no error analysis at scale.

## LLM (weak fit)

Bridge: weak but less weak than U05. Discrete
latents prefigure tokenization: both turn
continuous signals into countable symbols with a
learned codebook. The rate/distortion tradeoff
prefigures the tokenizer's vocab-size tradeoff.
Do not oversell: tokenizers are not trained by
STE.

Practice questions: state the VQ-tokenizer
analogy and its limit in two sentences.

Residual gap: everything LLM-specific (U10
material). This unit is an analogy source, not
training.

## MLOps

Bridge: the artifacts are the codebook, the
decoder, the prior, and the usage monitor.
Version them together: a new codebook needs a
refitted prior. The rollback signal is the dead-
code fraction rising or the quantization error
drifting up.

Practice questions: list the four artifacts.
Design the canary: compare per-code usage
between versions.

Residual gap: no artifact store, no CI/CD, no
observability stack. The lesson names the
signals, not the systems.

## Agents (weak fit)

Bridge: weak. The honest connection: discrete
latents as a countable action or state
abstraction, and the dead-code lesson (unused
capacity the objective never revives) as a
warning for exploration design.

Practice questions: state the dead-code-to-
exploration analogy in one sentence.

Residual gap: everything agent-specific. This
unit does not serve the role.
