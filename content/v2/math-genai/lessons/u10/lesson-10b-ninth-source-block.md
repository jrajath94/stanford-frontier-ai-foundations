# Lesson 10b, Ninth source block: RL overview to state-space models

Unit: math-genai-U10. Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

Title-level treatment of eight playlist entries (SRC-04):
W11L47 An overview of Reinforcement Learning, W11L48
Policy Gradient Theorem, W11L49 Expressing an AR-LM as
RL policy, W11L50 Proximal Policy Optimization (PPO),
W11L51 Trust Region Policy Optimization (TRPO), W12L52
Reward-Modelling, W12L53 Direct Preference Optimization
(DPO), W12L54 State-space-Models. No transcript
inspected (source_gaps.md G2). Each section states what
the title promises, the question it answers, where it
lands in the lesson, and what to verify when watching.
Nothing below claims lecture content.

## Scope and objectives

Scope: the eight titles above, mapped to U10 leaf
concepts C05-C09 and C12. Objectives: after this block
the learner can say what each title should teach, point
to the lesson section that prepares them for it, and
list the verification questions to ask while watching.

## SB01, W11L47 An overview of Reinforcement Learning

The title promises: RL foundations. The question:
what are states, actions, rewards, and policies?
Lands in: R31 (the P17 local bridge) and behind C07.
Preparation: the MDP tuple, return, the policy as a
conditional distribution. Verify when watching:
whether the lecture defines the MDP, states the
Markov property, and introduces value functions or
goes straight to policy methods.

## SB02, W11L48 Policy Gradient Theorem

The title promises: the theorem behind policy
optimization. The question: how do you differentiate
expected reward? Lands in: R31 and behind C07.
Preparation: the log-derivative trick,
E[grad log pi * A]. Verify when watching: whether
the theorem is stated with its conditions (the
fine print matters: differentiability, the
stationary distribution), and whether a proof
sketch or only the statement is given.

## SB03, W11L49 Expressing an AR-LM as RL policy

The title promises: the bridge from language
models to RL. The question: what are the states
and actions of a language model? Lands in: C05-C07
(the ramp into alignment). Preparation: U09-C05
(chain rule) and R31. The mapping: state = the
tokens so far, action = the next token, policy =
the AR distribution, reward = human preference
(or the reward model). Verify when watching:
whether the lecture makes this mapping explicit
and names what is unusual about it (huge action
space = vocab, deterministic transitions =
concatenation, sparse reward = one score per
response).

## SB04, W11L50 Proximal Policy Optimization (PPO)

The title promises: the PPO algorithm. The
question: how do you climb reward safely? Lands
in: C07. Preparation: the clipped objective,
the two toy cases (0.6000 with the clip binding,
-0.3200 with pessimism). Verify when watching:
whether the clip is derived or stated, whether
the pessimism of the min is explained, and
whether the value/advantage estimation is
covered or assumed.

## SB05, W11L51 Trust Region Policy Optimization (TRPO)

The title promises: the trust-region predecessor
of PPO. The question: what does a hard KL
constraint buy? Lands in: C07 (alternatives) and
C09 (KL regularization). Preparation: the KL
toy (0.0995 bits) and the soft-vs-hard
distinction. Verify when watching: whether the
lecture contrasts TRPO's constrained update with
PPO's clipped one, and whether the second-order
machinery is shown or cited.

## SB06, W12L52 Reward-Modelling

The title promises: learning the reward from
preferences. The question: how do pairwise
labels become a scalar function? Lands in: C06.
Preparation: the Bradley-Terry model, the toy
(P = 0.7109, loss 0.3412). Verify when watching:
whether BT is named, how label noise is
discussed (the seed of C10), and whether the
reward architecture (a finetuned LM with a
scalar head) is described.

## SB07, W12L53 Direct Preference Optimization (DPO)

The title promises: preference optimization
without RL. The question: what happens when you
solve the RL out? Lands in: C08. Preparation:
the DPO loss (margin 0.07, loss 0.6588), the
implicit rewards (0.0400, -0.0300), and the
assumption audit. Verify when watching: whether
the derivation from the KL-regularized
objective is shown, whether the assumptions
(BT true, pi_ref good, pairs cover the space)
are stated, and whether the absolute
log-ratio pathology is mentioned.

## SB08, W12L54 State-space-Models

The title promises: an alternative sequence
architecture. The question: what comes after
(or beside) transformers? Lands in: none of
C01-C11, recorded in C12 as out-of-scope.
Preparation: U09-C08/C11 (what the transformer
costs: O(n^2), 0.60 GB at n = 1024) is the
motivation, not the content. Verify when
watching: the recurrence structure, the
claimed complexity, and how it differs from
attention. Note: this title is the only
explicit non-transformer architecture title in
the playlist, no math for it is taught in this
unit.

## Source-block exercises (questions. Answers in keys-source-block-10b.md)

Q1. Map each of the eight titles to its U10 leaf
concept(s). Flag any title whose mapping is
ambiguous from the title alone.
Q2. SB03 is the hinge of the unit. Write the
state/action/policy/reward mapping for an AR-LM
and name the three unusual facts about this MDP.
Q3. SB04 vs SB05: state the one-line difference
between PPO and TRPO as the titles suggest it.
Q4. Which titles support C10 (reward hacking)?
Answer with the audit table, not with hope.
Q5. For SB02, state the policy gradient theorem
in one line and name one condition the title
does not guarantee is covered.
Q6. SB08 is out of scope. Argue for and against
teaching it in this unit anyway.

## Deep oral ladder (questions. Answers in keys-source-block-10b.md)

R1. What do eight titles prove about the
course's alignment coverage?
R2. Take W11L49: predict the lecture's central
mapping table.
R3. Check the prediction against the lesson:
which rows does R31 + C05-C07 already contain?
R4. Compare W12L52 and W12L53: what does the
second promise that the first does not?
R5. Critique: "eight titles, so alignment is
the best-covered part of the course." Attack it
with the audit denominators.
