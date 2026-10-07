# Lab 05: PCA and ICA

Date: 2026-10-06. Unit: cs229-U11.
Work the tasks, then check keys-lab-05.md. Run all code.

Setup: numpy only. Seeds as stated per task. No other installs.

## Task 1: PCA finds the subspace

Data: 500 points in 3-D near a plane: coordinates
N(0, 1.5), N(0, 1.3), N(0, 0.2) along random
orthonormal axes, seed 42. Center the data, compute
the covariance, report the three eigenvalues and the
cumulative variance explained. Keep the top 2
components, reconstruct, and report the reconstruction
MSE. State the relationship between the MSE and the
dropped eigenvalue.

## Task 2: whitening

Using the Task 1 data and eigendecomposition:
whiten the data (z = Lambda^{-1/2} U^T x). Report
the covariance of the whitened data (diagonal and
max off-diagonal magnitude). State what whitening
achieves and why ICA pipelines use it.

## Task 3: ICA unmixing (oracle)

Sources: s1 = sin(t), s2 = sign(sin(0.7 t)),
t over [0, 8 pi], 600 points. Mixing matrix
A = [[1.0, 0.8], [0.4, 1.0]]. Form x = A s.
Recover with W = inv(A): s_rec = W x. Report the
best-permutation absolute correlation of each
recovered source with the true sources. State the
two ambiguities this demo does not resolve.

## Deliverable

A short log: the three result blocks with numbers. No
essay. The numbers must match keys-lab-05.md within the
stated tolerance.
