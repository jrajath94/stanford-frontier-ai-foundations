# role_gap_map.md, math-ml

Date: 2026-10-06. Role bridges populated in RUN 4 for U05/U04c and
extended to U06-U10 by the fix builder 2026-10-06. This file holds
the map skeleton and the honesty rule.

## Honesty rule

Courses alone do not replace DSA, distributed systems, domain expertise,
collaboration, and research experience. State residual gaps per role. Never imply a course makes the learner hireable.

## Skeleton (populated in RUN 4)

- Research: assumptions, identifiability, objectives, evidence, baselines,
  uncertainty, open questions.
- Research engineering: implementation, numerics, profiling, reproduction,
  controlled ablations.
- FDE: discovery, system integration, security, business acceptance,
  communication, handoff.
- ML: data, evaluation, generalization, errors, deployment.
- LLM: tokenization, architecture, training/post-training, inference,
  evaluation.
- MLOps: artifacts, versioning, CI/CD, observability, drift, rollback,
  access and cost.
- Agents: tools/state/memory, stopping, permissions, approvals, side
  effects, recovery, reliability.

## RUN 4 role bridges, labeled by U05/U04c evidence

### Research

Bridge: the bias-variance split (U05 C04, lab-05 T3) and the EM
monotonicity argument via Jensen (U04c SB22) are assumption-first
reasoning: state the clauses, then compute. Identifiability
(U04c SB24, label-switching swap test) trains the habit of asking
whether the data can separate two parameters before estimating.
Residual gap: this course does not teach literature review, novel
claim defense, or peer critique. Open questions named in lessons:
nested CV theory, double descent, shift detection.

### Research engineering

Bridge: every number in U05/U04c is computed from a script
(compute_run4.py) with seed, dtype, and versions stated. The EM
implementation (SB22 code) and the LOOCV loop (C08 code) are
minimal references with expected outputs. The interview debug
tasks (U05 T1 shuffle bug, executed premise) train
implementation forensics. Residual gap: no profiling, no GPU
numerics, no large-scale reproduction. All toys are n <= 500.

### FDE

Bridge: leakage (U05 C11) and distribution shift (U05 C12) are the
two failure modes that kill deployed models. The interview
scenarios S1 (churn leak diagnosis) and S2 (label-buying decision
from a learning curve) practice discovery, stakeholder
communication, and handoff language. Residual gap: no real client
system, no security review, no acceptance-criteria negotiation.
The scenarios are paper exercises.

### ML

Bridge: full coverage: risk, loss selection, splits, CV, capacity,
learning curves, leakage, shift (U05 C01-C12), plus threshold
selection via ROC/NP/minmax (U04c SB20) and k-NN capacity (SB26).
This is the unit's home role. Residual gap: no production data
pipeline, no A/B testing, no fairness or monitoring practice.

### LLM

Bridge: indirect. Regularization and capacity thinking transfer to
fine-tuning choices (weight decay as MAP, early stopping as
capacity control), and shift monitoring transfers to prompt/data
drift. Correction 2026-10-06 (fix builder): the residual gap below
is superseded. Attention (U08-C09: soft lookup, 2x3 weights, rows
sum to 1.0), the transformer block shape trace (U08-C10), training
dynamics (U08-C07: gradient decay and explosion, U08-C12: Adam
against SGD), normalization (U08-C11), and gating (U08-C08) are
taught in U08. See the U08 bridges. Residual gap: tokenization and
post-training (RLHF/DPO) are not covered. No pretraining at scale.

### MLOps

Bridge: the split discipline (C07), CV (C08), and shift detection
(C12) are the evaluation half of MLOps: versioned data splits,
honest model selection, and input monitoring. Residual gap: no
artifact versioning, no CI/CD, no rollback drills, no cost
accounting. The unit teaches what to monitor, not how to run the
platform.

### Agents

Bridge: none direct in RUN 4. The stopping discipline (lock the
model before scoring, C07) and the approval-like gate of held-out
evaluation transfer loosely to agent evaluation hygiene.
Residual gap: this unit says nothing about tools, state, memory,
permissions, or recovery. Do not claim agent competence from it.

## Standing note

Each bridge above names the exact lesson evidence it rests on.
Where no evidence exists, the gap is stated instead of a bridge.
Review prompts are not hiring probabilities.

## RUN 5 and completion-run role bridges, labeled by U06-U10 evidence

### U06, linear models, kernels, margins (lesson-06, compute_run5.py)

#### Research

Bridge: duality reasoning (U06-C09: a* = 0.25, primal 0.25 equals
dual 0.25) and the kernel Gram PSD certificate (U06-C07:
eigenvalues 0.3911, 0.9656, 1.6433) are assumption-first
arguments: state the certificate, then compute it. Numerical
conditioning (U06-C11: kappa 3.35e6 against 2.618 after
standardization) trains distrust of raw solves. Residual gap: no
novel estimator design, no asymptotic theory, no peer-critique
practice.

#### Research engineering

Bridge: every number is computed from a script (compute_run5.py)
with seed, dtype, and versions stated. The OLS closed form
(U06-C01: w_hat = [0.3, 0.8], RSS 1.8) and the logistic one-step
likelihood lift (U06-C03: -1.3863 to -1.2691) are minimal
references with expected outputs. Residual gap: no profiling, no
GPU numerics, no large-scale reproduction. Toys are n <= 9.

#### FDE

Bridge: model choice under noise (U06-C12: logistic 0.7167 against
SVM 0.7833 after 9 label flips) is a stakeholder-facing tradeoff:
accuracy against robustness, stated in numbers. Residual gap: no
real client system, no security review, no acceptance
negotiation. The comparison is a toy.

#### ML

Bridge: home territory continues: linear and logistic regression,
softmax, kernels, SVM margins, model choice (U06-C01 through
C12). The margin computation (U06-C08: width 1.4142) and the
hinge-versus-logistic noise test are the model-selection
judgments ML engineers make. Residual gap: no production data
pipeline, no feature engineering at scale, no deployment.

#### LLM

Bridge: the logistic and softmax machinery (U06-C03/C04: sigmoid
step, logits [2, 1, 0] to probabilities summing to 1.0, CE
0.4076) is the output layer of every language model. Residual
gap: tokenization, attention, training, and inference are U08
scope, not U06.

#### MLOps

Bridge: numerical conditioning (U06-C11) is the monitoring half
of MLOps: a kappa of 3.35e6 on a raw Gram is the kind of silent
failure that input validation must catch. Residual gap: no
artifact versioning, no CI/CD, no rollback, no cost accounting.

#### Agents

Bridge: none direct. The model-choice judgment (U06-C12)
transfers loosely to agent-component choice by measured tradeoffs. Residual gap: nothing about tools, state, memory,
permissions, or recovery.

### U07, trees and ensembles (lesson-07, compute_completion.py)

#### Research

Bridge: the variance decomposition (U07-C06: rho sigma^2 +
(1 - rho) sigma^2/B, computed 0.37/0.604/0.208) is a
quantitative argument about what averaging can and cannot
remove. The impurity comparison (U07-C02: 0.32/0.5004/0.2 at
counts [4, 1]) shows three definitions agreeing at pure nodes.
Residual gap: no new splitting-criterion design, no consistency
theory.

#### Research engineering

Bridge: bagging measured honestly (U07-C05/C06, f04: val MSE
0.0513/0.0278/0.0344/0.0335, non-monotone curve kept) trains
reporting what the seed gave, not the smoothed story. The
AdaBoost reweighting (U07-C08: missed point to 1/2, rest to
1/6, alpha 0.5493) is a minimal reference. Residual gap: no
profiling, no distributed training. Toys are n <= 6.

#### FDE

Bridge: accuracy hiding skew (U07-C10: 0.95 accuracy against F1
0.5714) is the stakeholder trap: the headline number lies and
the honest metric must be named in the handoff. Residual gap:
no real client data, no acceptance negotiation.

#### ML

Bridge: home territory: trees, bagging, random forests,
boosting, class imbalance, baselines (U07-C01 through C12).
Every score needs a dumb reference (U07-C12: 0.5 against 1.0)
is the evaluation discipline. Residual gap: no production
feature pipeline, no online learning.

#### LLM

Bridge: indirect. Tree ensembles are not language-model
machinery. The transfer is evaluation honesty (baselines,
honest metrics). Residual gap: nothing about tokenization,
attention, training, or inference. Those are U08 scope.

#### MLOps

Bridge: the OOB discipline (U07-C07: honest votes from
replicates) and the baseline rule (U07-C12) are the evaluation
half of MLOps: lock the reference before scoring. Residual
gap: no artifact versioning, no CI/CD, no drift monitoring, no
rollback.

#### Agents

Bridge: none direct. The baseline discipline transfers loosely
to agent evaluation hygiene. Residual gap: nothing about
tools, state, memory, permissions, or recovery.

### U08, neural and sequence architectures (lesson-08, compute_completion.py)

#### Research

Bridge: the gradient-fate argument (U08-C07: 0.5^10 = 9.77e-4
against 1.5^10 = 57.67) is a mechanism-first claim with a
computed verdict. The backprop check (U08-C02: finite
differences agree to 8 digits) is the verification habit.
Residual gap: no novel architecture design, no convergence
theory, no literature critique.

#### Research engineering

Bridge: shape discipline through the stack (U08-C01/C10: X
(5, 8) nine-row trace, x (2,) to out 0.0) with every shape
named is the implementation contract. The gradient check
(C02) is the reference test. Residual gap: no GPU numerics,
no mixed precision, no distributed training, no profiling.
Toys are tiny.

#### FDE

Bridge: the transformer shape trace (U08-C10) is the
integration contract: shapes must line up at every add, which
is what a handoff document must state. Residual gap: no real
client system, no latency or cost negotiation.

#### ML

Bridge: MLPs, CNNs, RNNs, attention, normalization,
optimization (U08-C01 through C12) are the model zoo. The
Adam-versus-SGD step (U08-C12: [0.01, -0.01] against [-0.005,
0.003]) is the optimizer judgment. Residual gap: no
large-scale training, no data pipeline, no deployment.

#### LLM

Bridge: direct. Attention as a soft lookup (U08-C09: 2x3
weight heatmap, rows sum to 1.0), the transformer block shape
trace (U08-C10), training dynamics (U08-C07: gradient decay
and explosion, U08-C12: Adam against SGD), normalization
(U08-C11: [-1.2247, 0, 1.2247]), and gating (U08-C08) are the
architecture, training, and inference machinery. This
supersedes the RUN 4 residual gap (see the corrected LLM
entry above). Residual gap: tokenization and post-training
(RLHF/DPO) are not covered. No pretraining at scale.

#### MLOps

Bridge: normalization (U08-C11) and the optimizer comparison
(U08-C12) are the stability half of MLOps: what to watch when
training runs. Residual gap: no checkpointing, no distributed
orchestration, no serving, no cost accounting.

#### Agents

Bridge: none direct. The shape contract (U08-C10) transfers
loosely to interface contracts between agent components.
Residual gap: nothing about tools, state, memory,
permissions, approvals, or recovery.

### U09, unsupervised leaves (lesson-09b, lesson-04c SB21-SB26, compute_run5.py)

#### Research

Bridge: the eigendecomposition argument (U09-C03: one direction
keeps 99.72 percent) and the reconstruction identity (U09-C04:
MSE 0.009172 equals the dropped eigenvalue) are
proof-by-computation. The EM climb and its fragility (U04c
SB22/SB23: bound -13.0089 to -3.3320. Two starts giving
-3.3320 against -14.1109) train convergence skepticism.
Residual gap: no novel latent-model design, no identifiability
theory beyond SB24.

#### Research engineering

Bridge: k-means measured honestly (U09-C01: J 0.1333, a bad
start changes the path, not the destination, on the toy) and
the held-out elbow (U09-C12: 14.18 to 0.44 to 0.34 to 0.29,
elbow at k=2, retitled honestly when the U-shape failed to
appear) are the report-what-you-measured habit. Residual gap:
no large-scale clustering, no profiling.

#### FDE

Bridge: the held-out elbow (U09-C12) is the model-selection
conversation with a stakeholder: k=2 is defensible, k=5 buys
little. Residual gap: no real client data, no acceptance
negotiation.

#### ML

Bridge: k-means, PCA, distances and scaling, reconstruction,
factor models, held-out evaluation (U09-C01 through C12, plus
U04c C05-C08, C10, C11). The distance-and-scaling flip
(U09-C02: nearest neighbor flips from B to C under y-axis
scaling) is the preprocessing judgment. Residual gap: no
production pipeline, no streaming clustering.

#### LLM

Bridge: indirect. PCA and reconstruction thinking transfer to
embedding analysis and representation inspection. Residual
gap: nothing about tokenization, attention, training, or
inference. Those are U08 scope.

#### MLOps

Bridge: held-out evaluation (U09-C12) and the honest retitling
of f03 are the model-selection half of MLOps: versioned
evaluation, honest reporting. Residual gap: no artifact
versioning, no CI/CD, no drift monitoring.

#### Agents

Bridge: none direct. Residual gap: nothing about tools, state,
memory, permissions, or recovery.

### U10, generative bridge and synthesis (lesson-10, compute_completion2.py)

#### Research

Bridge: the assumption-first proof audit (U10-C07: four
load-bearing assumptions, gap 0.1438 equals KL(q||posterior))
and the toy counterexamples (U10-C08: three claims, three
killer numbers) are the falsification habit. The VAE ledger
(U10-C02: KL 0.3981, ELBO -1.3283) keeps every term accounted.
Residual gap: no novel generative-model design, no new
theorem. The research-critique (C10) and oral-defense (C11)
sections practice the form, not a real venue.

#### Research engineering

Bridge: implementation evidence with build numbers (U10-C09:
predicted 4.0, measured 4.2887 in seeded trials) and the ELBO
gap measured to 4 digits (U10-C01/C07: -0.7340/-0.8779 bars)
are the verify-against-theory habit. Residual gap: no
large-scale generative training, no GPU work.

#### FDE

Bridge: the attack-with-build-numbers discipline (U10-C10:
three claims restated as honest versions) is the handoff
language: what was shown, on what toy, with what numbers.
Residual gap: no real client system, no acceptance
negotiation.

#### ML

Bridge: generative evaluation (U10-C05: ten samples give 0.3
against 0.5 at x=0), objective comparisons (U10-C03/C04: GAN
value -0.5798 to -0.9163, JSD 0.0462), and the regularization
distinctions (U10-C06: collapse at beta=10) are the
model-assessment judgments. Residual gap: no production
generative deployment, no human evaluation.

#### LLM

Bridge: the VAE and ELBO machinery (U10-C02) and the latent
generative framing (U10-C01) are the probabilistic foundation
under modern generative models, including diffusion and
language-model pretraining objectives. Residual gap: no
autoregressive language modeling, no tokenization, no
post-training. U08 covers attention and transformers.

#### MLOps

Bridge: the source-gap audit (U10-C12: G1-G7 status table) is
the honesty half of MLOps: known unknowns stay visible.
Residual gap: no artifact versioning, no CI/CD, no serving, no
monitoring.

#### Agents

Bridge: none direct. Residual gap: nothing about tools, state,
memory, permissions, approvals, side effects, or recovery.
