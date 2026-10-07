# keys-lab-02.md

Date: 2026-10-06. Reference outputs for lab 02. Seed 0.

## Task 1

NLL at [0.1, -0.2, 0.3] = 162.98 (tolerance 0.05).

## Task 2

Analytic gradient: [29.13692, -59.25908, 42.895].
Max abs difference vs finite differences: 1e-8. Verdict:
the gradient is correct.

## Task 3

Gradient ascent: 2000 steps, NLL 87.9923,
theta = [-0.639, 1.840, -1.194].
Newton: 7 steps, NLL 87.9923,
theta = [-0.639, 1.840, -1.194].
Newton converges in far fewer steps. Both reach the same
minimum (the objective is convex). Tolerance: NLL within
0.01, theta within 1e-3.

## Task 4

Shift invariance holds: softmax(t) == softmax(t + 7) to
1e-12. Naive formula on [1000, 1001]: [nan, nan]
(overflow). The max-subtraction trick is required.
