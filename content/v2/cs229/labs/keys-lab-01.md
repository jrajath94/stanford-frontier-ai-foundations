# keys-lab-01.md

Date: 2026-10-06. Reference outputs for lab 01. Seed 42.

Computed with numpy on this machine. Tolerances below absorb
platform float differences.

## Task 1

theta = [2.0785, 1.4882], cost = 0.0499. Tolerance: theta
within 1e-3 of the lstsq answer (it matches to 1e-14).

## Task 2

lstsq theta = [2.0785, 1.4882], matches Task 1 within 1e-3.
norm(X.T @ residuals) = 1.5e-13. This proves the residuals
are orthogonal to every column of X: the prediction is the
projection of y onto the column space.

## Task 3

Explicit inverse: LinAlgError (singular matrix). pinv theta
= [2.0785, 0.7441, 0.7441]: the duplicated weight splits
evenly, the minimum-norm choice. Singular values of X:
[24.895, 2.479, 0.0]: the zero flags the deficiency.

## Task 4

Test MSE vs tau: 0.0389, 0.0241, 0.0632, 0.1314, 0.1727.
Procedure pinned in lab-01.md: linspace train grid,
default_rng(42), train noise sd 0.5 first, then 200
uniform test points, noiseless sin targets.
U-shaped: high at tau = 0.2 (wiggles chase noise), minimum
near tau = 0.5, rising toward the global-line MSE at
tau = 5.0. Best tau ≈ 0.5. Tolerance: each value within
0.01.
