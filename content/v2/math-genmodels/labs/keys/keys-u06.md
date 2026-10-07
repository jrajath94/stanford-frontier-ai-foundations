# Lab answers: U06

Unit: math-genmodels-U06. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

s(1) = -1, s(0) = 0, s(-2) = 2 confirmed. Empirical objectives
(seed 11, n = 5000): a = 0.5 gives -0.3751, a = 1.0 gives
-0.5006, a = 2.0 gives -0.0023. Winner a = 1.0, value near
-0.5. The arrows point toward 0 from both sides.

## E2

All three Tweedie estimates are 0.0 exactly. Two-mass case:
Tweedie returns 0.0 (the posterior mean). the true posterior is
bimodal at -2 and 2. Lesson: Tweedie returns the posterior
mean, which is the right estimate and the wrong sample when the
posterior is multimodal.

## E3

Closed-form trajectory (seed 11): 2.0, 2.1818, 2.3054, 2.4023,
0.8194. Stepped trajectory: 2.0, 2.4858, 1.6906, 1.3843,
0.3871. They differ pointwise (different noise draws) but both
are draws from N(sqrt(alpha_bar_t) x 2, 1 - alpha_bar_t) at
each t: the marginals match by construction.

## E4

Reverse mean 1.8983, variance 0.0714 confirmed. DDIM step:
1.7630 confirmed, identical on repeat runs. DDPM with fresh
noise differs across runs by construction.

## E5

Errors: Euler-10: 0.0192, Euler-20: 0.0094, Euler-40: 0.0046,
Heun-10: 0.0007. Euler quarters on doubling. Cheapest under
0.001: Heun at 10 steps (20 score calls) beats Euler at 160
steps.

## E6

Scores: 13.5 at 0.01, 1.1 at 0.1, 0.0 at 0.5. Sigma =
0.01: boundary score 28.8. Lesson: the boundary score
blows up as the smoothing shrinks, so any likelihood built on
the smoothed law depends on an arbitrary noise floor and must
name it.
