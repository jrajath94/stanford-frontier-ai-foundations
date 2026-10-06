# Role bridges, U09 score-based models and autoregressive LMs

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U09 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: the two cultures are the research surface.
The toy quantifies both failure modes: slow mixing
(56 valley crossings per 20000 steps at a = 0.1,
mean 0.161 not 0) for the score culture, exposure
bias (0.7653 bits/token) for the AR culture. The
open question is the hybrid: VQ the space, AR over
codes (U06-C07 meets C12). Which failure mode
dominates at equal budget is unmeasured.

Practice questions: state both failure numbers.
Design the hybrid experiment from C12. Name the
evidence that would promote W10L40 from
title-level to source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do not
transfer to papers.

## Research engineering

Bridge: the numerics are the job. The mask assert
(row 0 == [1, 0, 0]), the attention shape asserts
((n,n) scores, rows sum to 1), the score
antisymmetry check (s(-1) = -s(1)), and the ODE
sign discipline (sampling integrates backward:
x <- x + 0.5 beta s h) are the four habits. The
sign error caught in this build (the forward-time
ODE applied in the sampling direction) is the
canonical bug class.

Practice questions: implement the four checks
from memory. Reproduce every number via
compute_run5b.py. Write the ODE sign bug from
memory, then fix it.

Residual gap: no GPU kernels, no mixed precision,
no distributed training. The toys are numpy, not
systems.

## FDE (forward deployed engineer)

Bridge: the context budget is the deployable
constraint: 0.60 GB at n = 1024, 9.66 GB at n =
4096, 38.65 GB at n = 8192 for the toy shape.
The discovery question: what context does the
application need, and what does it cost? The
handoff artifact is the (n, precision, window)
triple with its measured memory.

Practice questions: compute the memory for a
client's (n, h, L, dtype). Name the three levers
(fewer layers, fp16, sliding window) and price
one. Write the budget sheet.

Residual gap: no real GPU measurement, no
serving stack, no client contract. Arithmetic is
not a deployment.

## ML

Bridge: the eval discipline is the ML surface.
Perplexity 1.82574 is a model-selection number,
not a product number, the vocab-comparison trap
(E17) is the classic error. The exposure gap
(0.7653) is why the training loss understates
test-time cost. Eval rule: fixed tokenizer for
perplexity, task metrics for ship decisions.

Practice questions: explain the vocab trap and
its repair. Design the perplexity-vs-task
correlation study. State when teacher forcing
stops being enough.

Residual gap: no real benchmark, no eval suite,
no error analysis on real data. The toy metrics
are not evals.

## LLM

Bridge: strong fit. This unit is the LLM
foundation: the chain rule is next-token
prediction, teacher forcing is the pretraining
step, the
causal mask is the honesty mechanism, the
transformer block is the architecture, and the
sampling knobs (T, top-p) are the product
surface U10-C01 extends. The exposure gap is the
reason decoding strategy matters.

Practice questions: map each U09 section to an
LLM lifecycle stage (pretrain, eval, decode).
Compute the sampling distributions for a new
logit vector. Explain the mask to a skeptic.

Residual gap: no real tokenizer, no real
pretraining run, no RLHF (U10). The unit is
foundations, not a shipped model.

## MLOps (partial fit)

Bridge: partial fit, stated openly. The
transferable pieces: the mask assert as a CI
check, the memory budget sheet as a deploy
gate input, the train/serve split (parallel
training vs sequential inference) as a
capacity-planning input. What does not transfer:
no artifacts, versioning, or monitoring are
taught here.

Practice questions: write the mask assert as a
CI gate. Build the memory budget sheet for a
deploy candidate. Name what the unit does not
cover (registry, rollback, drift).

Residual gap: no pipeline, no registry, no
observability. U10-C11 extends the eval side,
the ops side stays open.

## Agents (partial fit)

Bridge: partial fit, stated openly. AR sampling
is the substrate agents generate with
(sequential token emission), and the exposure
gap is one reason agent trajectories drift.
What does not transfer: no tools, no state, no
stopping, no permissions. The unit teaches the
generator, not the agent.

Practice questions: explain how exposure bias
compounds over a 50-step agent trajectory.
Name the agent pieces this unit does not teach.

Residual gap: no tool use, no memory, no
recovery, no eval of agentic behavior. The
generator is necessary, not sufficient.
