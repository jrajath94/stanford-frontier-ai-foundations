# Lab 09: Value iteration, model estimation, GRPO advantages

Date: 2026-10-06. Unit: cs229-U15.
Work the tasks, then check keys-lab-09.md. Run all code.

Setup: numpy only. No other installs.

## Task 1: value iteration by hand and code

MDP: states {A, B}. Actions at A:
move (to B with probability 1) and stay
(in A with probability 1). B is
absorbing (stays in B). R(A) = 0,
R(B) = 1, gamma = 0.9.

(a) Write the closed-form V(B) and V(A)
for the move policy.
(b) Run 5 synchronous value iteration
sweeps from V = 0. Report V(A), V(B)
after each sweep and the max error
against the closed form.
(c) State the error ratio between
consecutive sweeps.

## Task 2: transition estimation from counts

In state s, action a was taken 10 times:
7 transitions to s1, 3 to s2. Then 10
more trials gave 5 to s1 and 5 to s2.

(a) Report P_hat after the first 10
trials and after all 20, by count
accumulation.
(b) A second (s, a) pair was never
visited, |S| = 4. Report the notes'
fallback estimate.
(c) State one sentence on why the
fallback is dangerous for large |S|.

## Task 3: GRPO advantages

One prompt, G = 4 completions, verifier
rewards R = [1, 0, 0, 1], epsilon = 1e-8.

(a) Compute Rbar, s_R, and the four
advantages Ahat_i = (R_i - Rbar) /
(s_R + epsilon).
(b) The clipped token surrogate uses
these advantages with r_t = 1.0 for
all tokens. State C_t for a token in
completion 1 and in completion 2.
(c) One sentence: what happens to the
advantages if all four completions
get reward 1?

## Deliverable

A short log: the three result blocks
with numbers. The numbers must match
keys-lab-09.md within the stated
tolerance.
