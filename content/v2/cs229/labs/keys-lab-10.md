# Keys: Lab 10

## Task 1

(a) Phi_2 = -1. Phi_1 = -1.5. Phi_0
= -1.6. Tolerance 1e-9.

(b) L_1 = -0.5, L_0 = -0.6. From
s_0 = 5: a_0 = -3, s_1 = 2, a_1 =
-1, s_2 = 1.

(c) The unsigned formula gives L_1 =
+0.5. It is wrong because the
first-order condition carries a
leading minus sign: +0.5 pushes the
state away from zero instead of
toward it.

## Task 2

(a) s_{1|0} = 0, Sigma_{1|0} = 2.

(b) K = 2/3. s_{1|1} = 1.0.
Sigma_{1|1} = 2/3.

(c) K = 2 / (2 + 100) = 0.0196: the
filter nearly ignores the noisy
observation and keeps the prediction.

## Task 3

(a) True gradient 0.25. Two-sample
estimate (0.5 + 0) / 2 = 0.25.

(b) No baseline: samples {0.5, 0},
variance 0.0625. Baseline 0.5:
samples {0.25, 0.25}, variance 0.
Mean 0.25 in both cases.

(c) (1, 1.3): 1.2, capped. (1, 0.5):
0.5, not capped. (-1, 1.3): -1.3,
not capped. (-1, 0.5): -0.8, capped.
Tolerance 1e-9.
