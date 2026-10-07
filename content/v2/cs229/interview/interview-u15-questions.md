# Interview bank: U15, reasoning and reinforcement learning foundations

Date: 2026-10-06. Keys in keys-u15.md.
Provenance: original practice questions
derived from SRC-01 chapters 18-19. No
employer attribution.

## Breadth questions

Q01: What does a chain-of-thought trace
buy an autoregressive model?
Q02: Few-shot CoT versus zero-shot CoT:
what changes in the prompt?
Q03: Write the RLVR objective and name
each symbol.
Q04: What is the KL penalty in (18.2)
for?
Q05: State the MDP tuple.
Q06: Write the Bellman equation for
V^pi and for V*.

## Deep ladders (5 follow-ups each)

L01 (Bellman and iteration):
F1: Define V^pi(s).
F2: Derive the Bellman equation for
V^pi by splitting the first step.
F3: Write the value iteration update
and state the two update styles.
F4: On the two-state toy, compute two
sweeps by hand from V = 0.
F5: Compare value iteration and policy
iteration on per-iteration cost and
convergence behavior.

L02 (RLVR mechanics):
F1: Write the token-level MDP for
RLVR: state, action, transition,
reward.
F2: Why is the reward zero at every
token except the last?
F3: Write the GRPO advantage formula
and compute it for R = [1, 0, 0, 1].
F4: How does GRPO differ from PPO in
where the advantage comes from?
F5: The reward curve rises but task
accuracy is flat. Diagnose.

## Analytical exercises

A01: Derive the Bellman optimality
equation (19.2) from the definition
V*(s) = max_pi V^pi(s).
A02: Show that E_{a~pi(.|s)}[grad
log pi(a|s)] = 0 and explain why a
state-only baseline does not bias the
policy gradient.

## Implementation and debug task

D01: Implement synchronous value
iteration for the two-state toy and
print the max error per sweep for 10
sweeps. Then introduce a bug: set
gamma = 1.0. Describe what the error
sequence does and why.

## Changed-constraint scenarios

S01: The verifier can only check
answer format, not correctness.
How do you change the RLVR setup, and
what do you monitor?
S02: The state space is R^12 and you
may not use a simulator. Which of
the notes' continuous-state methods
survives, and what breaks?

## Research critique

R01: "Fitted value iteration is just
value iteration with regression, so
it inherits the convergence
guarantee." Critique this claim with
the notes' own statements.
