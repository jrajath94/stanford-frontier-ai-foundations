# Keys, lab 06 kernels and margins

Date: 2026-10-06. All numbers computed 2026-10-06, numpy 1.26.4,
float64, seed 7.

## Task 1

X^T X = [[3, 6], [6, 14]], det 6. Inverse [[2.333333, -1],
[-1, 0.5]]. X^T y = [5, 11]. w_hat = [2.333333*5 - 11,
-5 + 5.5] = [0.666667, 0.5]. Predictions [1.166667, 1.666667,
2.166667]. Residuals [-0.166667, 0.333333, -0.166667]. RSS
0.166667. Residual sum 4.44e-16 < 1e-12. lstsq agrees to
1e-15.

## Task 2

(a) Eigenvalues [0.817546, 0.981684, 1.20077], all positive:
PSD confirmed.

(b) q3 = (1.8, 1.8): k = [0.039164, 0.193980, 0.193980].
f = -0.039164 + 0.193980 + 0.193980 = 0.348796, sign +1.
q4 = (0.1, 0.1): k = [0.990050, 0.163654, 0.163654].
f = -0.990050 + 0.163654 + 0.163654 = -0.662742, sign -1.

(c) The RBF kernel weights nearby points exponentially more.
q4 sits almost on top of the -1 point (k = 0.99), so its vote
drowns the two distant +1 votes (k = 0.16 each).

## Task 3

(a) 0.5 * 2 + C * 1.4: C = 0.01 gives 1.014, 0.1 gives 1.14,
1 gives 2.4, 10 gives 15.0.

(b) The objective prefers the smallest C (0.01): violations are
cheap, so the answer barely moves. No C in the set gives the
noisy point margin >= 1 with this w. Classifying it
"correctly" would need w to rotate toward it, which the
objective punishes. Tension: the objective prices margin width
against violations. Accuracy counts only verdicts.

(c) Train accuracy on 5 points is a 5-level staircase. C moves
it in jumps, and the winning C memorizes this toy's noise.
U05-C07: selection needs held-out data, not the training
scoreboard.
