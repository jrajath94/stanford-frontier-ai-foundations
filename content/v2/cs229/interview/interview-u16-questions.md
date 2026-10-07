# Interview bank: U16, control and policy optimization

Date: 2026-10-06. Keys in keys-u16.md.
Provenance: original practice questions
derived from SRC-01 chapters 20-21. No
employer attribution.

## Breadth questions

Q01: Why is the optimal policy of a
finite-horizon MDP non-stationary?
Q02: State the LQR assumptions.
Q03: What is the shape of V*_t in
LQR, and why does the shape persist?
Q04: Why does the LQR policy ignore
the noise?
Q05: State the four DDP steps.
Q06: Write the REINFORCE gradient
estimator.

## Deep ladders (5 follow-ups each)

L01 (LQR):
F1: Write the LQR dynamics and
reward.
F2: Derive the first-order condition
for a*_t.
F3: Compute Phi_1 and L_1 for the 1d
toy by hand.
F4: The notes print the gain without
a minus sign. Prove the sign on the
toy.
F5: When does the Riccati recursion
fail to give a usable policy?

L02 (policy gradients):
F1: Derive the log-derivative
identity.
F2: Why do the transition terms
vanish from the gradient?
F3: Prove the zero-mean score
identity.
F4: Show a state-only baseline keeps
the estimator unbiased.
F5: PPO reuses old-policy data.
What does the likelihood ratio
correct, and what stays uncorrected?

## Analytical exercises

A01: From the corrected gain, derive
the Riccati update for Phi_t and
verify Phi_1 = -1.5 on the toy.
A02: For the bandit toy, compute the
variance of the REINFORCE estimator
with no baseline and with B = 0.5,
and prove the mean is unchanged.

## Implementation and debug task

D01: Implement the 1d Riccati
recursion and assert Phi stays
negative semidefinite. Then flip the
gain sign (use the unsigned formula)
and roll out from s_0 = 5. Report
the trajectory and explain the
difference.

## Changed-constraint scenarios

S01: The dynamics are nonlinear and
the task is a swing-up from far
away. Which methods from the unit
survive, and in what order do you
apply them?
S02: Sampling is 100x more expensive
than gradient steps. How do you set
the PPO update schedule, and what
two numbers guard the trust region?

## Research critique

R01: "PPO is off-policy because it
reuses old trajectories." Critique
with the notes' own local-surrogate
statement.
