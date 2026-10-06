# Lab 04, entropy, KL, and MLE computations

Unit: math-ml-U04. Date: 2026-10-06. numpy 1.26.4, float64.
Keys in labs/keys-lab-04.md. Test-mode: solve closed-book, then check.

## Task 1, entropy and KL by hand then in code

(a) By hand: compute H([0.75, 0.25]) in nats to 4 digits.
(b) In code: compute D_KL([0.7, 0.3] || [0.5, 0.5]) and the reverse
direction. State both numbers and which direction costs more.
(c) Verify the decomposition: H(p) + D_KL(p||q) equals the direct
cross-entropy -sum p log q within 1e-12.

## Task 2, Gaussian MLE with a score check

Data [2.1, 2.5, 1.9, 2.3].
(a) Compute mu_hat and sigma2_hat.
(b) Compute the score sum (d_i - mu_hat)/sigma2_hat and verify it is
within 1e-12 of zero.
(c) Break it: append 20.0 to the data. Recompute mu_hat and write
one sentence on what the outlier did to the fit.

## Task 3, multivariate Gaussian MLE

Points: [0.5, -0.2], [-0.3, 0.4], [0.1, 0.2], [0.4, 0.5],
[-0.2, -0.4], [0.0, 0.1].
(a) Compute mu_hat and Sigma_hat (MLE, divide by n).
(b) Verify Sigma_hat is symmetric and its eigenvalues are positive.
Verify trace equals the eigenvalue sum.
(c) Evaluate the fitted density at [0.1, 0.2]. State the number.
