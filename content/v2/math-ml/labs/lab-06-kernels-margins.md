# Lab 06, linear models, kernels, and margins

Unit: math-ml-U06. Date: 2026-10-06. Keys in labs/keys-lab-06.md.
All numbers computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## Task 1, OLS audit on a new toy

Data: x = [1, 2, 3], y = [1, 2, 2]. Build X with an intercept
column. Solve the normal equations. Report w_hat, the three
residuals, RSS, and the residual sum. Assert:

- the residual sum is below 1e-12 in absolute value. 
- RSS equals the squared norm of the residuals. 
- solving via np.linalg.lstsq agrees with the normal-equation
  answer to 1e-12.

Measured reference: w_hat = [0.666667, 0.5], residual sum
4.44e-16. Your run must reproduce these to 1e-9.

## Task 2, kernel expansion by hand

Points P = [(0,0), (2,0), (0,2)] with labels y = [-1, +1, +1]
and alphas all 1. RBF kernel gamma = 0.5.

(a) Verify the Gram is PSD: compute its three eigenvalues and
confirm all are positive. Reference: [0.817546, 0.981684,
1.20077].

(b) Classify q3 = (1.8, 1.8) and q4 = (0.1, 0.1) with the kernel
expansion f(q) = sum_i alpha_i y_i k(x_i, q). Compute each
k(x_i, q) by hand, then f, then the sign. Reference: f(q3) =
0.348796 (sign +1), f(q4) = -0.662742 (sign -1).

(c) Explain in two sentences why q4 lands negative although two
of three training labels are positive.

## Task 3, the C sweep and the objective-vs-accuracy tension

Use the C08 toy with the noisy point (0.2, 0.2), y = -1, and
w = [1, 1], b = 0.

(a) Compute the total objective 0.5||w||^2 + C * 1.4 for
C in {0.01, 0.1, 1, 10}. Reference: 1.014, 1.14, 2.4, 15.0.

(b) Which C does the objective prefer? Which C classifies the
noisy point "correctly" (functional margin >= 1)? State the
tension in two sentences: the objective is not accuracy.

(c) A teammate proposes picking C by train accuracy on this
toy. Explain why the proposal fails, citing U05-C07.
