# Role bridges, U08 diffusion variants and implementation

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U08 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: the weighting-vs-quality gap is the research
surface. The toy quantifies it: the ELBO weight falls
to 0.02873 at t = 50 while the simplified loss holds
1.0, a 34.8x distortion at the middle timesteps, yet
the uniform weight wins on samples. The open question
is the coefficient-ratio curve (U07 shell 9, U08-C06
research extension): which reweighting helps
likelihood without hurting samples.

Practice questions: state the two weightings and the
34.8x number. Design the five-scheme experiment from
C06. Name the evidence that would promote W9L34 from
title-level to source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do not
transfer to papers.

## Research engineering

Bridge: the sampler battery is the job. The six
asserts (C12) plus the eta = 0 determinism check
(C02) plus the E-018 indexing discipline are the
implementation habits. The DDIM sigma formula is
the highest-risk line: eta = 1 must recover the
jump posterior variance (0.15405473), not the
single-step one (0.01976684), confusing the two is
the new bug class this unit adds.

Practice questions: implement ddim_step and the
battery from memory. Reproduce every number via
compute_run5b.py. Write the jump-vs-step variance
confusion from memory, then fix it.

Residual gap: no GPU profiling, no distributed
training, no trained U-Net. The scalar toy is not a
production sampler.

## FDE (forward deployed engineer)

Bridge: the serving-cost arithmetic is the
deployable constraint: 1.56672 MFLOP per sample at
DDIM-10, 3.13344 with CFG, 15.6672 at DDPM-100.
The discovery question: what latency does the
application tolerate, and is g = 1 quality enough?
The handoff artifact is the frozen quad (net,
schedule, sampler, g): changing any one changes
the product.

Practice questions: compute the serving cost for
three configs from the FLOP table. Name the
quality evidence still needed before picking one.
Write the version-freeze checklist.

Residual gap: no real latency measurement, no
client integration, no cost modeling beyond
FLOPs. Arithmetic is not a deployment.

## ML

Bridge: the guidance tradeoff is the ML surface.
The inverted-U hit-rate (0.000, 1.000, 0.000,
0.000 over g = 0..3) is the mechanism behind
"more guidance is not always better". The eval
discipline: tune g on held-out prompts with a
fixed metric, never on the training set, never
by vibes.

Practice questions: explain the inverted-U from
the linear blend. Design the g-sweep eval
protocol. Name the failure mode of tuning g on
training prompts.

Residual gap: no real dataset, no FID/recall
measurement, no generalization study. The toy
metric is not an eval.

## LLM (weak fit)

Bridge: weak fit, stated openly. This unit is
continuous diffusion (images), LLM diffusion is
a different object (discrete diffusion, not
taught here). The transferable pieces: the
timestep-embedding idea (position/time signals
as vectors, cf. U09-C08 positional encoding),
the serving-cost arithmetic pattern (evals x
cost-per-eval, cf. U10-C02/C04), and the
train/serve split discipline (g tuned at serve
time, cf. U10-C01 temperature).

Practice questions: map the DDIM sampler
choices to LLM decode choices (S to max
tokens, g to temperature). Name what does not
transfer (the Gaussian math, the U-Net).

Residual gap: no discrete diffusion, no LLM
decoding, no token-level guidance. Do not
claim diffusion expertise transfers to LLMs.

## MLOps

Bridge: the version-freeze discipline. The
battery (C12) runs in CI on every sampler
change, the frozen quad (net, schedule,
sampler, g) is the versioned artifact, the
FLOP table is the cost side of the deploy
gate. Deterministic sampling (eta = 0) gives
reproducible artifacts for regression tests.

Practice questions: write the CI gate (six
asserts + determinism check). Define the
artifact quad and its versioning. Explain why
eta = 0 helps regression testing.

Residual gap: no real CI system, no model
registry, no rollback drill. The battery is
a script, not a pipeline.

## Agents (weak fit)

Bridge: weak fit, stated openly. Diffusion
sampling is not agentic (no tools, no state,
no stopping decisions). The transferable
habit is the check battery as a pattern:
deterministic replay (eta = 0) for
debugging nondeterministic systems, and the
train/serve knob split (frozen net, tunable
g) as an analogy for frozen policy plus
tunable decoding.

Practice questions: explain why eta = 0 is
the right debugging mode. Map the frozen
quad to an agent's frozen components.

Residual gap: no tools, no state, no
permissions, no recovery. This unit teaches
no agent engineering.
