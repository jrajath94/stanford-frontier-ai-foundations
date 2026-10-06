# Role bridges, U04 Wasserstein and improved adversarial training

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U04 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: the dual's three assumptions (C03) are the
research surface. Assumption 2 (Lipschitz everywhere)
fails first in practice. The C06 blind spot
(penalty 5.1e-22 on-segment, gradient 26.7
off-segment) is a publishable-style negative
result on a toy. Assumption 3 (rich critic class)
is the identifiability question: the weak critic
reports 0.02 against an exact 2.0 (C11), so W is
not identified from the gap alone.

Practice questions: state the duality with its
assumptions. Design the falsifiable extension in
C06 shell 9. Name the evidence that would promote
W4L11 from title-level to source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do
not transfer to papers.

## Research engineering

Bridge: the numerics are the job. The sv-product
bound (C04: 2.5 vs measured 1.0), the
finite-difference gradient-norm check (C06:
2.49999999996 vs 2.5), and the clip-then-assert
pattern (C05) are the three implementation
habits. The ablation protocol: clipping vs
penalty under the C12 fixed budget (5/13/23
units per G step), same seeds, calibration
ratio reported.

Practice questions: implement w1_dual_gap with
the slope assert (lesson debug task). Profile
the critic step with and without the penalty's
second backward pass. Reproduce every number
via compute_run4.py.

Residual gap: no GPU profiling, no distributed
training, no framework internals. The numpy
reference is not a production implementation.

## FDE (forward deployed engineer)

Bridge: domain adversarial networks (W4L15,
C09) are the deployable idea: when the
deployment domain differs from the training
domain, the gradient-reversal update (0.5 to
-0.5 at lambda = 1) trains features the
domain classifier cannot use. The W4T8 UDA
tutorial angle is the handoff artifact: idea
then implementation. The stakeholder line:
"the critic gap is a lower bound on the
distance, not a quality certificate" (C11).

Practice questions: write the reversal update
for a scalar feature. Name the failure when
the domain label correlates with the task
label (interview S2). List the six C11
limitations in stakeholder language.

Residual gap: no client discovery, no
production data, no latency or cost model for
the client's stack. The toy domains are not
a deployment.

## ML (machine learning engineer)

Bridge: the C12 comparison protocol is the ML
surface: same architectures, same seeds,
fixed flop budget, calibration ratio next to
every gap, coverage diagnostics next to every
sample sheet. The stability diagnostic (C07:
ratio 1.0 vs 0.5) is the training-health
monitor. The failure taxonomy (C11) is the
error analysis.

Practice questions: write the five-rule
protocol from memory. Compute the budget
table for n_critic = 1/5/10. Dismantle the
research-critique claim section by section.

Residual gap: no real datasets, no
hyperparameter search at scale, no
deployment pipeline. The protocol is
practiced on toys only.

## LLM (LLM engineer)

Bridge: weak fit, labeled honestly. The U04
machinery (adversarial games, Lipschitz
critics, transport distances) is not the LLM
training story: LLMs train by likelihood
(U09), align by preference methods (U10).
The one transferable habit is the C12
protocol discipline: fixed budgets, same
seeds, reported uncertainty. Nothing else
in U04 applies directly.

Residual gap: tokenization, architecture,
post-training, inference, and evaluation
are all uncovered. U04 contributes almost
nothing to LLM readiness.

## MLOps

Bridge: the calibration ratio (C07) is the
monitoring signal: log gap and ratio per
checkpoint, alert when the ratio drops below
0.5. The critic checkpoint is the artifact:
version it with the generator, because the
gap is meaningless without the critic that
produced it. The n_critic multiplier (C08)
is the cost model: 13 units per G step at
n_critic = 5 vs 5 at n_critic = 1.

Practice questions: design the logging schema
for gap, ratio, and n_critic. State the
rollback rule when the ratio collapses.
Compute the training-budget delta of
n_critic = 5 vs 1 over 10k steps.

Residual gap: no CI/CD, no artifact store,
no drift detection on real data, no access
controls. The schema is a design, not a
system.

## Agents (agent engineer)

Bridge: weak fit, labeled honestly. U04 has
no tools, state, memory, permissions, or
recovery content. The transferable habits
are the audit procedure (C10: names are not
evidence) and the falsifiable-extension
discipline (shell 9 of each mechanism).
Nothing else in U04 applies directly.

Residual gap: the entire agent surface:
tools, state, stopping, approvals, side
effects, reliability. U04 contributes
almost nothing to agent readiness.
