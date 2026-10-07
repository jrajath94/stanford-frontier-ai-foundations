# keys-lab-05.md

Date: 2026-10-06. Reference outputs for lab 05.

Computed with numpy on this machine. Tolerances below absorb
platform float differences.

## Task 1

Eigenvalues: 2.129, 1.685, 0.042. Cumulative variance:
0.5522, 0.9890, 1.0000. Tolerance: eigenvalues within
0.05. Reconstruction MSE with top-2: 0.0141. The
per-entry MSE equals the dropped eigenvalue divided by
d (0.042 / 3 = 0.014): the reconstruction error is
exactly the dropped variance.

## Task 2

Whitened covariance diagonal: 1.002, 1.002, 1.002.
Max off-diagonal magnitude: 0.002. Tolerance: 0.01.
Whitening gives identity covariance: it removes all
second-order structure, so the unmixing step only has
to find higher-order (non-Gaussian) structure.

## Task 3

Recovered-source correlations: 1.0000, 1.0000
(tolerance 0.01). The demo does not resolve the two
ICA ambiguities: permutation (which recovered channel
is which source) and scaling (the recovered amplitude
is arbitrary).
