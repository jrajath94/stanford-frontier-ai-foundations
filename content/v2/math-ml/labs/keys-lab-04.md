# Lab keys 04, entropy, KL, and MLE computations

Date: 2026-10-06. Computed values, numpy 1.26.4, float64.

## Task 1

(a) H = 0.5623351446188083 nats.
(b) D_KL(p||q) = 0.08228287850505181. D_KL(q||p) =
0.08717669357238889. The reverse direction costs more.
(c) H(p) + D_KL(p||q) = 0.6931471805599452. Direct cross-entropy:
0.6931471805599453. Difference is float noise, within 1e-12.

## Task 2

(a) mu_hat = 2.2. sigma2_hat = 0.05.
(b) Score = -1.7763568394002505e-14, within 1e-12 of zero.
(c) New mu_hat = 5.76. One sentence: "The outlier dragged the mean
to a point where four of the five data points do not live, because
the Gaussian MLE averages everything with no resistance."

## Task 3

(a) mu_hat = [0.0833, 0.1]. Sigma_hat = [[0.0847, 0.005],
[0.005, 0.1]].
(b) Symmetric by construction (off-diagonals both 0.005).
Eigenvalues [0.0832, 0.1015], both positive: PSD. Trace 0.1847
equals the eigenvalue sum 0.1847.
(c) Density at [0.1, 0.2]: 1.6458814703416096.
