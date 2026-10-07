# Lab keys: U04

Unit: math-genmodels-U04. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

Max round-trip error for both maps: 0.0 within float noise
(1e-15). The z^2 pseudo-inverse fails: sqrt(|x|) cannot recover
the sign, so half the round-trips land on the mirror input. Lesson:
test the round-trip on the full domain, not just positives.

## E2

Quadrature area: 0.999999. p_x(1) = 0.1995, p_x(3) = 0.1210. Without
the |det| term the area is 2.0: the "density" is not a density.
Lesson: the Jacobian term is load-bearing. Dropping it doubles
every likelihood here.

## E3

Forward gives (0.5, -0.75). Inverse recovers (0.5, -1.0) to 1e-12.
Finite-difference Jacobian matches [[1, 0], [0, 1.25]] within 1e-6.
Log-det: 0.2231 both from the diagonal product and from
np.linalg.det. Lesson: two independent computations, one number.

## E4

Hand score: -2.4629 - 0.2231 = -2.6860. Flipped sign: -2.2398. The
lab-report sentence: "The log-det subtracts in density eval because
scoring runs the inverse map. The wrong sign inflates every
likelihood by twice the log-det."

## E5

Serial steps equal D in each case: 2, 16, 256. Pure scoring at D =
256: MAF, density in one batched pass. Pure generation: IAF,
sampling in one batched pass. Lesson: the cheap direction decides
the architecture, not the total parameter count.

## E6

Linear-space product: inf. Log-space: 1381.55. Condition number of
diag(1e6, 1e-6): 1e12. The lab-report rule: "Accumulate
log-determinants, never determinants. A condition number above 1e8
means the inverse direction is numerically unreliable in float64."
