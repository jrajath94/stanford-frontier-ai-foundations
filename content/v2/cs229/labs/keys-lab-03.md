# keys-lab-03.md

Date: 2026-10-06. Reference outputs for lab 03.

Computed with numpy on this machine. Tolerances below absorb
platform float differences.

## Task 1

Per degree (bias^2, variance, test MSE):

- deg 1: 0.0072, 0.0156, 0.1128
- deg 2: 0.0010, 0.0218, 0.1128
- deg 3: 0.0018, 0.0393, 0.1311
- deg 4: 0.0028, 0.1495, 0.2423
- deg 5: 0.0447, 1.2735, 1.4081

Tolerance: relative 5 percent. Degree 2 wins (tied
with degree 1 within noise. Both near the noise floor
0.09). At degree 5 variance dominates (1.27 of 1.41).

## Task 2

Test MSE vs lambda: 43.4310, 0.0345, 0.0772, 0.1034,
0.2242, 0.3966, 0.4703. Tolerance: relative 5 percent.
U-shaped: unregularized explodes (43.4), best at
lambda = 0.001 (0.0345). It then rises as shrinkage adds
bias.

## Task 3

Train MSE: 0.4843, 0.1334, 0.0673, 0.0484, 0.0441,
0.0407, 0.0335, 0.0166 (falls monotonically).
Validation MSE: 0.9437, 0.4560, 2.3685, 0.1317,
1.3249, 6.1671, 174.5959, 3589.3432.
Selected degree: 3. Training error alone picks degree
7 (lowest train MSE, catastrophic validation): it
always favors the most complex model.
