# Role bridges, U07 DDPM derivation and parameterizations

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U07 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: the simplified-loss-vs-ELBO gap is the
research surface. The toy quantifies it: the full
ELBO weights timesteps by KL coefficients while
the trained loss weights them uniformly, and the
prior term (0.7713 nats) dominates the toy bound
12x over the middle sum (0.0629 nats). The open
question is the coefficient-ratio curve
(mechanism C shell 9): which reweighting helps
likelihood without hurting samples.

Practice questions: state the two weightings.
Design the three-variant experiment from R1. Name
the evidence that would promote W8L30 from
title-level to source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do
not transfer to papers.

## Research engineering

Bridge: the numerics are the job. The
variance-preservation check (alpha_t + beta_t =
1), the round-trip eps -> x0 -> eps test, and the
six-assert battery (mechanism D) are the three
implementation habits. The indexing discipline
(1-indexed t, ab[t-1]) is the fourth: the T1 bug
class is the most common DDPM implementation
failure.

Practice questions: implement the battery.
Reproduce every number via compute_run5a.py. Write
the T1 bug from memory, then fix it.

Residual gap: no GPU profiling, no distributed
training, no U-Net internals. The scalar toy is
not a production implementation.

## FDE (forward deployed engineer)

Bridge: the sampling loop cost is the deployable
constraint: T network evaluations per sample. The
discovery question: what latency does the
application tolerate? The handoff artifact is the
trained noise predictor plus the schedule plus
the sampler config: changing any one changes the
product.

Practice questions: compute the serving cost for
T = 100 vs T = 1000 at a stated per-evaluation
latency. State why the loop cost, not the
training loss, sets the budget.

Residual gap: no client discovery, no serving
stack, no monitoring pipeline. U08 attacks the
loop cost directly.

## ML

Bridge: train/eval discipline for diffusion
models. The training loss (0.0305 on the toy
batch) is not the product metric. Per-t loss
curves and sample quality are. The schedule
endpoint SNR (0.57) is a data/schedule mismatch
the loss cannot see.

Practice questions: name three evaluations beyond
the training loss. Explain the prior-mismatch
failure in two sentences.

Residual gap: no real datasets, no generalization
study, no error analysis at scale.

## LLM (weak fit)

Bridge: weak. The honest connection: the
timestep-conditioned network prefigures
conditioned generation, and the
predict-then-sample structure (eps_hat ->
x0_hat -> mu_rev) prefigures structured
decoding. Diffusion language models exist but
are not this course's subject.

Practice questions: state one real and one
spurious DDPM-to-LLM analogy.

Residual gap: everything LLM-specific (U10
material). This unit does not serve the role.

## MLOps

Bridge: the artifacts are the noise predictor,
the schedule, the sampler config, and the per-t
loss monitor. Version them together: a new
schedule needs revalidation of sample quality.
The rollback signal is the per-t loss curve
shifting, not the mean loss.

Practice questions: list the four artifacts.
Design the canary: compare samples across
schedule versions at fixed seeds.

Residual gap: no artifact store, no CI/CD, no
observability stack. The lesson names the
signals, not the systems.

## Agents (weak fit)

Bridge: weak. The honest connection: the
reverse chain as a sequential decision process
with T steps, and the compounding-error lesson
as a warning for long-horizon rollouts.

Practice questions: state the loop-to-rollout
analogy in one sentence.

Residual gap: everything agent-specific. This
unit does not serve the role.
