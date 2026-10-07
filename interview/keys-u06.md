# keys-u06.md: interview bank U06 answers

Date: 2026-10-06. Closed-book reference answers with
rubrics. Keep separate from the questions file.

## Breadth answers

B1. B = (log10 L2 - log10 L1)/(log10 N2 - log10 N1),
a = log10 L1 - b log10 N1, L(N) = 10^(a + b log10 N).
C = 6 N D: 2N forward + 4N backward per token.

B2. K examples in the prompt, no gradient step. The
weights freeze because the task is specified in
activations (conditioning), not in parameters.

B3. A step metric (pass/fail) on a smooth skill ramp
snaps 0 to 1 at the crossing. The jump is in the
measurement.

B4. Loss = -mean over response tokens of log P(y_t |
x, y_<t). Instruction tokens carry zero weight so
the model learns responses, not instructions.

B5. P(A beats B) = sigmoid(r_A - r_B). Objective:
E[r] - beta * KL(policy || SFT).

B6. The model critiques and revises its draft against
a written constitution. (revised, draft) pairs train
the reward model. Humans write the rules. The AI
applies them.

B7. W = W0 + (alpha/r) B A. Trainable: r(d+k) vs dk.
Toy 4096/16: 131,072 vs 16,777,216 (128x).

B8. Step = max(t_compute, t_fetch + t_assemble).
Compute-bound (loader invisible) vs data-bound
(GPUs idle).

## Deep ladder 1 answers

L1a. A scaling law is a fitted power law relating
loss to parameters, data, or compute: straight line
in log-log space.

L1b. B = -0.0748/decade, a = 1.0534, L(1e11) = 1.70.

L1c. Two points fix a line, so predict(N2) = L2 by
construction. Slope = (0.3054 - 0.3802)/1 decade =
-0.0748.

L1d. Fit_law solves the two equations. Predict
evaluates the line. Each point costs a full
training run.

L1e. Floor form wins when large-N points bend flat. 
two-point line wins with exactly two runs and no
visible floor.

L1f. Recipe changed (data mix/optimizer) between
runs, or seed luck on the single big run.

L1g. Assumption: architecture, data mix, recipe
fixed. Counterexample: double data quality at fixed
N. Loss drops, the N-only law says impossible.

L1h. Three toy scales, fit on two, predict the
third. Accept if within 5% of measured. Reject the
law form otherwise.

Rubric: must compute the slope, not just quote it.
Red flag: treats the law as physics rather than a
fitted trend. Remediation: re-derive from two
points.

## Deep ladder 2 answers

L2a. The reward model scores responses from pairwise
votes. The KL leash penalizes distance from the SFT
policy: E[r] - beta KL.

L2b. P(A wins) = sigmoid(0.4) = 0.5987. Objective =
1.0 - 0.25 = 0.75.

L2c. Objective = reward - beta*KL. At equal reward,
smaller KL gives a larger objective: 0.95 > 0.75.

L2d. Code is the sigmoid and the penalty term.
RLHF pays human label cost. RLAIF pays AI
critique-revise inference cost.

L2e. Explicit reward model when reward needs
auditing and reuse across policies. DPO when
preference data is fixed and fewer moving parts
matter.

L2f. Reward hacking. First fix: raise beta (tighten
the leash) or audit the reward model on the hacked
outputs.

L2g. Assumption: the reward model equals true
quality. Counterexample: long flattering answers
score high and help little.

L2h. Sweep rollout length at fixed beta, plot reward
vs KL. Expect reward to plateau while KL grows. 
the knee marks hacking onset.

Rubric: must separate the reward model from the
policy. Red flag: "RLHF makes the model truthful."
Remediation: the hacking counterexample.

## Analytical answers

A1. C = 6 * 1e9 * 2e10 = 1.2e20 FLOPs. Doubling C
at fixed D/N buys log10(2) = 0.301 decades of N.
Loss drop = 0.0748 * 0.301 = 0.0225 in log10, so
L falls by factor 10^0.0225 = 1.053: about a 5%
relative drop.

A2. Mean 2.31, std 0.0158. B at 2.295: gap 0.015 <
1 std: no evidence. Third recipe at 2.27: gap
0.04 = 2.5 std: evidence of a real win.

## Debug answer

D1. Bug: the function returns bins with no document
mask, so attention leaks across packed documents.
Fix: return (bins, masks) with a per-bin mask that
blocks cross-document attention. Invariant: every
query attends only within its own document.

## Changed-constraint answers

S1. With free instant human labels, RLHF dominates
on quality per label: RLAIF keeps a role only for
rules humans cannot label consistently (the
constitution prices tradeoffs explicitly). DPO
still wins on simplicity when the preference set is
fixed.

S2. Free compute kills the cost-per-token lever and
the loader-tuning lever (idle GPUs cost nothing).
Packing still matters (time-to-loss on real
tokens), LoRA still matters (memory per task, not
dollars), scaling still matters (the loss target is
fixed. Free compute just buys more of it).

## Research critique answer

R1. Five audits: (1) same eval? Invalidate: the 1B
ran an easier suite. (2) same data? Invalidate:
the 1B trained on far more/better tokens. (3) same
metric? Invalidate: threshold metric vs continuous
score. (4) cherry-picked tasks? Invalidate: the
10B wins on the unreported half. (5) one seed?
Invalidate: the gap is inside seed noise.
