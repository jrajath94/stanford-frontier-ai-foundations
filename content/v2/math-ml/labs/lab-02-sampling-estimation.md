# Lab 02, sampling and estimation computations

Unit: math-ml-U04 block (Lec 02-10). Date: 2026-10-06. numpy 1.26.4,
float64, seed 7 where RNG is used. Keys in labs/keys-lab-02.md.
Test-mode: solve closed-book, then check.

## Task 1, the 8-draw rerun

(a) Draw 8 samples from Bernoulli(0.3) with default_rng(7). Record
the draws and the sample mean.
(b) Compute the predicted typical error sqrt(p(1-p)/n). Compare with
the observed |mean - 0.3|.
(c) Rerun with n = 800, same seed. Record the new mean and error.
Did the error shrink by roughly sqrt(800/8) = 10x?

## Task 2, likelihood grid

Flips: H, H, T, H.
(a) Evaluate the log-likelihood on the grid p = 0.1, 0.2, ..., 0.9.
Record the maximizing grid point.
(b) Compare with the analytic MLE 3/4 and its log-likelihood
-2.249340578475233.
(c) Repeat with flips H only (n = 1). What does the grid say, and why
is it dangerous?

## Task 3, histogram honesty

xs = [0.2, 0.5, 0.7, 1.1, 1.4, 1.6, 2.0, 2.3].
(a) Bin into 4 equal bins on [0, 2.5]. Record counts and edges.
(b) Convert to a density (divide by n * width). Check the total area
sums to 1.
(c) Rebin into 8 bins. What changes, and what does that say about
density estimates from n = 8?
