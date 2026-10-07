# Lab 02: logistic regression mechanics

Date: 2026-10-06. Unit: cs229-U03.
Work the tasks, then check keys-lab-02.md. Run all code.

Setup: numpy only. Seed 0. Data: 200 examples, 2 features
plus bias, true theta = [-0.5, 1.5, -1.0], labels drawn
from the Bernoulli model.

## Task 1: stable loss

Implement the negative log likelihood with the softplus
form. Evaluate at theta = [0.1, -0.2, 0.3]. Report the
value.

## Task 2: gradient check

Implement the gradient X.T @ (h - y). Compare against
central finite differences at the same theta. Report the
max abs difference. State the verdict.

## Task 3: gradient ascent vs Newton

Fit with gradient ascent (alpha = 0.5, 2000 steps) and
with Newton updates (Hessian = (X.T * w) @ X,
w = h(1-h), max 20 steps). Report iterations, final NLL,
and final theta for both. State which converged faster
and whether the answers agree.

## Task 4: softmax stability

Implement softmax with the max-subtraction trick. Verify
shift invariance on t = [3.0, 1.0, 0.5]. Show what the
naive formula gives on t = [1000, 1001].

## Deliverable

A short log with the numbers. They must match
keys-lab-02.md within tolerance.
