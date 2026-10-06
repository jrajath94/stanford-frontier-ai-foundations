# Role bridges, U05 variational autoencoders

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U05 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: posterior collapse (C07) is the research
surface. The collapse ledger (healthy -0.9108 nats
vs collapsed -0.4516 nats) is a clean negative
result: the objective prefers the worse model. The
beta* = 1.6007 boundary is a falsifiable prediction
on the toy. The open question is the rank
experiment (mechanism C shell 9): with a linear
decoder of rank r < d, which dims stay active.

Practice questions: state the ELBO with its
assumptions. Design the beta* verification sweep.
Name the evidence that would promote W5L18 from
title-level to source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do
not transfer to papers.

## Research engineering

Bridge: the numerics are the job. The analytical
KL vs Monte Carlo check (0.1766 vs 0.1760 nats),
the finite-difference KL derivative (0.4000000000
vs 0.4), and the five-test battery (C12) are the
three implementation habits. The ablation protocol:
beta in {0.5, 1, 2, 4, 8} on fixed seeds, ELBO and
active-dim count reported.

Practice questions: implement elbo_1sample with the
battery asserts. Reproduce every number via
compute_run5a.py. Profile the 1-sample vs 8-sample
ELBO gradient noise.

Residual gap: no GPU profiling, no distributed
training, no framework internals. The numpy
reference is not a production implementation.

## FDE (forward deployed engineer)

Bridge: amortized inference (C11) is the
deployable idea: one encoder forward pass per new
x, no per-point optimization. The discovery
question: does the deployment data match the
training distribution? If not, the encoder's q is
poor exactly where it matters. The handoff
artifact is the encoder plus the KL-per-dim
monitor: the stakeholder sees active dims, not the
ELBO.

Practice questions: state the inference/training
cost split. Design the acceptance gate: minimum
active dims and traversal smoothness on
deployment-like data.

Residual gap: no client discovery, no latency
budgeting on real hardware, no monitoring
pipeline. The toy has no deployment.

## ML

Bridge: the train/eval split for generative
models. The ELBO is the training objective. Sample
quality needs separate evaluation (the lesson's
mechanism A shell 10). Posterior collapse is the
train/eval divergence made numeric: the objective
improves while the product worsens.

Practice questions: name three evaluations beyond
the ELBO. Explain why the collapsed model passes
the training metric and fails the product.

Residual gap: no real datasets, no generalization
study, no error analysis at scale.

## LLM (weak fit)

Bridge: weak. VAEs are not the LLM training
paradigm. The honest connection: the ELBO's
reconstruction/KL split prefigures the
likelihood/regularization splits in RLHF-style
objectives, and amortized inference prefigures
single-pass encoding. Do not oversell it.

Practice questions: state one real and one
spurious VAE-to-LLM analogy.

Residual gap: everything LLM-specific (U10
material). This unit does not serve the role.

## MLOps

Bridge: the artifacts are the encoder, the
decoder, the prior, and the KL-per-dim monitor.
Version them together: a decoder without its
encoder's q statistics is undeployable. The
rollback signal is the active-dim count dropping,
not the loss rising.

Practice questions: list the four artifacts.
Design the canary: compare traversal outputs
between versions.

Residual gap: no artifact store, no CI/CD, no
observability stack. The lesson names the
signals, not the systems.

## Agents (weak fit)

Bridge: weak. VAEs do not plan or act. The
honest connection: the latent code as a
compressed state representation, and the
posterior-collapse lesson (an objective that
rewards ignoring the state) as a warning for
reward design.

Practice questions: state the collapse-to-reward-
hacking analogy in one sentence.

Residual gap: everything agent-specific. This
unit does not serve the role.
