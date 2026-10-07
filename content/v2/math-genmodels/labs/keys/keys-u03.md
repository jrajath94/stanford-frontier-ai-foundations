# Lab keys: U03

Unit: math-genmodels-U03. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

MSE at d = 1: 0.0139. MSE at d = 2: 0.0. The d = 2 map is the
identity: it keeps the noise too. Zero error with zero compression
is not a win. Lesson: reconstruction alone cannot rank
bottlenecks. The bottleneck width is the actual model choice.

## E2

Reconstruction: -2.1516. KL: 0.9013. ELBO: -3.0529. Monte Carlo KL
with 200000 samples lands within 0.01 of 0.9013 (seed 8 gives about
0.900). Lesson: the closed form is exact and free. Sampling it is
pure waste.

## E3

Objectives: beta 0.1 -> -2.2417, 0.5 -> -2.6022, 1 -> -3.0529, 2 ->
-3.9542, 4 -> -5.7567, 8 -> -9.3619. Only beta = 1 is a likelihood
bound. The other points are rate-distortion Lagrangian values, not
bounds. Lesson: never tune beta on the weighted objective. It falls
by construction.

## E4

Usage counts: the middle codes split the eight points, the far code
at (100,100) gets zero assignments. Mean quantization error is
finite for the four sane codes. The moved code is dead: zero usage,
zero gradient, never recovers on its own. Lesson: count code usage
on every validation batch. Dead codes are silent capacity loss.

## E5

At mu = 0: true 0.3989, straight-through 1.0, bias 0.6011. At mu =
2.0: true 0.0540, straight-through 1.0, bias 0.9460. The bias grows
as the threshold saturates. Lesson: the estimator's error is
largest exactly where the true gradient is smallest.

## E6

BPD from the ELBO: 2.2022 bits/dim, an upper bound on the true BPD.
The honest number to report is the bound, labeled as a bound. The
lab-report rule: "Report VAE likelihoods as ELBO bounds, never as
exact values. Label bound versus exact on every table."
