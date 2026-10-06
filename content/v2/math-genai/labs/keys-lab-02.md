# keys-lab-02.md, Lab 02 answer keys

Date: 2026-10-06. All numbers computed 2026-10-06, numpy
1.26.4, float64, seed 0 unless stated
(compute_run3.py).

## Task 1

(a) D*(x) = pd(x)/(pd(x)+pg(x)). x = 0: a = 0.3990, b =
0.1079, D* = 0.7870. x = 1: a = 0.2420, b = 0.7979, D*
= 0.2327.
(b) The five marks reproduce to 1e-4 with the gauss
helper.
(c) V = -1.3537194326677915, JS = 0.32314028366610423,
2JS-2 = -1.353719432667792. Identity error 4.4e-16.
(d) Without the clip, 1 - ds underflows to exactly 0
at the grid tails (1-(1-1e-300) == 0 in float64), so
np.log2(0) = -inf with a RuntimeWarning, and the
trapezoid sum is -inf. The clip line is load-bearing
(errors.md E-011).

## Task 2

(a) Predicted: at D = 0.5, both -0.5. At D = 0.001,
minimax -0.001, non-saturating -0.999.
(b) Measured: identical. Collapse ratio: 0.5/0.001 =
500x.
(c) At D = 0.999 the mirror: minimax -0.999 (alive),
non-saturating -0.001 (dead). Each loss has its own
dead zone.
(d) "Your generator gradient vanished: D is
perfect and the minimax loss gives G nothing. Switch
to the non-saturating loss or weaken D."

## Task 3

(a) Predicted: Phi(-2) = 0.0228.
(b) Measured: 0.0232. SE = sqrt(0.0228*0.9772/10000) =
0.0015. |0.0232-0.0228| < 1 SE: prediction holds.
(c) Healthy: 0.4952, expected 0.5, SE = 0.0050: holds.
(d) D*(-2) = 0.9993: the optimal critic is near-certain
that left-hump points are real, so it sees the missing
mode even though no fake sample lands there.

## Task 4

(a) -log D* = log((pd+pg)/pd) = log(2m/pd). E_pg:
E_pg[log(pg/pd)] - E_pg[log(pg/(2m))] = KL(pg||pd) -
KL(pg||m) + log2 2 = KL(pg||pd) - KL(pg||m) + 1.
(b) LHS = 1.7910143066844335, RHS =
1.7910143066844333. Difference 2e-16.
(c) The minus sign rewards p_g for concentrating where
m is small relative to p_g: sharpness, mode-seeking.
It encourages mode collapse.
(d) Both sides equal 1.0: LHS = E_pd[-log2(1/2)] = 1,
RHS = 0 - 0 + 1 = 1. At equilibrium the loss is 1 bit
and both gradients are -0.5: the fix changes dynamics,
not the fixed point.

## Task 5

(b) Finite differences: 0.6819786672607187.
(c) Analytic via grid: 0.681979859549136.
(d) Agreement to 1.2e-6, within 1e-5. A disagreement
would indicate a bug in the D' grid (resolution or
boundaries) or in the loss sampling (seed, N, or the
interp), not in the game theory: the two routes
differentiate the same function.
