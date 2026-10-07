# Lab answers: U05

Unit: math-genmodels-U05. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

Min ~1.0, max ~2.0, mean ~1.333 (E[z^2] = 1/3). All 10,000 samples
in [1, 2]. The closed form fails because the fold z -> z^2 maps
two z values to each x, and the density needs the sum over both
preimages with the Jacobian of each branch: expressible but not
a clean formula, and for a neural g not expressible at all.

## E2

r(0) = 1.6487, r(2) = 0.2231, D*(0) = 0.6225, D*(2) = 0.1824.
Crossing at x = 0.5: r = 1.0000 there. D* - r/(1+r) is 0 to
1e-12 on the whole grid.

## E3

V(d) peaks at d = 0.5 with -1.3863. Quadrature JSD = 0.1114,
V* = -1.1635. The gap means: a constant discriminator cannot
reach the pointwise optimum. the true laws differ, so the best
discriminator beats the coin flip by 0.2228 nats.

## E4

Table (s, JSD, W1): (0, 0.0000, 0), (1, 0.1114, 1), (2, 0.3368,
2), (3, 0.5268, 3), (4, 0.6327, 4), (5, 0.6759, 5). JSD flattens
toward 0.6931 while W1 grows linearly. Disjoint narrow pair:
JSD = 0.6931 = log 2 within 0.01. Machine-verified by
labs/keys/verify_jsd_table_u05.py: 2,000,001-point trapezoid on
[-20,20] and [-30,30] plus stabilized scipy quad, all three
agree to 4 decimals.

## E5

Radii at steps 20..200: 1.56, 1.73, 1.91, 2.12, 2.34, 2.59,
2.86, 3.17, 3.51, 3.88. Monotone growth. lr = 0.05: growth
continues, roughly half the rate. Lesson: the game Jacobian has
imaginary eigenvalues, so gradient steps spiral outward. smaller
steps slow the spiral but the equilibrium needs algorithmic
fixes, not just a small lr.

## E6

Collapsed: precision 1.0 (all samples within range), recall 0.0
(no sample near the left mode). Blurry N(0,9): precision ~1.0,
recall ~0.24 (some mass lands near -3 by luck). Precision alone
crowns the collapsed model. recall alone favors the blurry one.
The pair exposes both failures.
