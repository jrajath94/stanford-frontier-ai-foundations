# Lab 11: Baselines, uncertainty, KL identities

Date: 2026-10-06. Unit: cs229-U17.
Work the tasks, then check keys-lab-11.md. Run all code.

Setup: numpy only. No other installs.

## Task 1: three baselines with tests

(a) Ridge: generate n = 200, d = 5
from a known theta with Gaussian
noise, sigma = 0.5, seed 7. Fit
ridge with lambda = 1. Report the
max |theta_hat - theta|.
(b) k-means: two Gaussian blobs in
2d, 100 points each, seed 7. Run
one Lloyd iteration from a fixed
init. Report the objective before
and after.
(c) Value iteration: the two-state
toy from U15. Report the max error
after 5 sweeps and the error ratio.

## Task 2: uncertainty reporting

Ten seeds give returns [8.1, 8.3,
7.9, 8.2, 8.0, 8.4, 8.1, 7.8, 8.2,
8.0].

(a) Report mean, sample std, and
standard error.
(b) A competing method reports mean
8.15 over 3 seeds with no spread.
State in two sentences why you
cannot declare a winner.

## Task 3: KL identity check

P = N([1, 0], I_2), Q = N([0, 0],
I_2).

(a) Closed-form KL from Lemma A.1.3.
(b) Monte Carlo estimate with
200000 samples, seed 7. Report both
and the absolute difference.

## Deliverable

A short log: the three result blocks
with numbers. The numbers must match
keys-lab-11.md within the stated
tolerance.
