# Answer keys, source block 10b (W11-W12 titles, title-level)

Date: 2026-10-06. Ground truth: lesson-10 and compute_run5b.py.
No transcript inspected, all answers stay at the title boundary.

## Q1

W11L47 -> R31, behind C07. W11L48 -> R31, behind
C07. W11L49 -> C05-C07 ramp. W11L50 -> C07.
W11L51 -> C07/C09. W12L52 -> C06. W12L53 -> C08.
W12L54 -> out of scope (C12 records it).
Ambiguous: W11L47 (how deep an overview?),
W11L51 (derivation or contrast?), W12L54 (which
state-space flavor?).

## Q2

State: the token sequence so far. Action: the
next token. Policy: the AR distribution
p(x_i|x_<i). Reward: human preference or the
reward model score. Unusual: (1) action space =
vocab (tens of thousands), (2) transitions
deterministic (append the token), (3) reward
sparse (one score per full response).

## Q3

PPO: clipped surrogate objective, first-order,
the leash is soft. TRPO: hard KL constraint on
the update, second-order machinery. The titles
suggest the course teaches both, the lesson
computes PPO and states TRPO as the alternative.

## Q4

None directly. The audit table shows C10
(reward hacking) as an authored bridge with no
title. W12L52 teaches the reward model, not its
failure mode. Hope is not a mapping.

## Q5

One line: grad E[return] = E[grad log pi(a|s)
A(s,a)]. Condition not guaranteed: the
differentiability and regularity conditions,
and whether the proof or only the statement is
given. The title promises the theorem, not the
fine print.

## Q6

For: it is the only non-transformer
architecture title, the O(n^2) cost (C11)
motivates alternatives, completeness. Against:
no other unit prepares its math, one title
cannot carry a full mechanism, the unit's
scope (inference/quantization/alignment of
transformer LMs) is already full. Verdict:
record it (C12 does), do not teach it here.

## R1

Eight titles prove eight scheduled topics
across Weeks 11-12. They prove the RL-to-
alignment arc was scheduled, nothing about
depth, code, or emphasis.

## R2

Prediction: a table with rows state/action/
policy/transition/reward and the AR-LM
instantiation of each, plus the three
unusual facts.

## R3

R31 holds the MDP definitions, C05 holds the
AR distribution, C06-C07 hold reward and
policy optimization. The lesson contains all
rows of the predicted table. Watching would
add the lecturer's framing and any worked
example.

## R4

W12L52 promises the reward model (learning r
from pairs). W12L53 promises skipping it
(DPO's closed-form substitution). The second
needs the first's BT setup to motivate the
shortcut.

## R5

Attack: the audit denominators say 4 of 12
U10 concepts are title-level (C06-C09), 7 are
authored bridges. Eight titles cover the RL
arc, but inference (C01-C02), quantization
(C03), hardware (C04), SFT demos (C05),
hacking (C10), and eval (C11) have no titles.
"Best-covered" confuses the scheduled arc
with the unit's full scope.
