# Interview keys: U04

Unit: math-genmodels-U04. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers, red flags, and rubrics. Original practice material.

## Q1

Strong: invertibility gives the inverse map that density eval
needs. Without it, the change-of-variables formula needs a sum
over preimages the code does not compute, so the first thing to
break is the likelihood: silently wrong, no error raised. Red
flag: "sampling still works, so it is fine". Rubric: the density
direction named as the casualty.

## Q2

Strong: it corrects for volume change. Stretching space spreads
the same mass over more volume, so the density drops. The |det|
factor keeps the total mass at 1. Red flag: calling it a
normalization constant. Rubric: the conservation argument.

## Q3

Strong: y1 depends only on x1 and y2 on (x1, x2), so the Jacobian
is lower triangular by construction. That makes the determinant
the diagonal product: O(D) instead of O(D^3). Red flag:
memorizing "triangular" without the dependency reason. Rubric:
the dependency structure plus the cost.

## Q4

Strong: log p(y) = log p_z(f^-1(y)) - log|det J|. The minus comes
from scoring through the inverse map: forward expansion is inverse
contraction. Red flag: the sign error. Rubric: the formula with
the sign justified.

## Q5

Strong: MAF: density 1 parallel pass, sampling D serial steps. IAF:
the reverse. The autoregressive dependency y_i on y_<i> chains one
direction and frees the other. Red flag: mixing up which is which.
Rubric: both models with the mechanism.

## Q6

Strong: determinants multiply across layers and dimensions and
overflow float64 on deep stacks (1e600 vs 1e308). Log-determinants
add and stay finite. The likelihood needs the log anyway. Red
flag: "float128 fixes it". Rubric: the overflow number and the
log argument.

## L1

1. p_x(x) = p_z(f^-1(x)) |det J_{f^-1}(x)|.
2. J = [[1, 0], [0, 1.25]]: dy1/dx1 = 1, dy2/dx2 = 1 + x1^2 = 1.25,
   off-diagonals 0 and 0.
3. Triangular det is the diagonal product by row expansion. Log of
   it: log 1 + log 1.25 = 0.2231.
4. -2.4629 - 0.2231 = -2.6860 nats.
5. The error is 2 x 0.2231: the sign is flipped. The code adds the
   log-det instead of subtracting.
Red flags: computing the Jacobian in the wrong direction. Rubric:
all five rungs with the 2x diagnosis.

## L2

1. Coupling: half conditions, half transforms, parallel both
   ways. Autoregressive: each coord depends on previous ones, one
   direction serial.
2. At D = 256: MAF sampling 256 serial steps, density 1 pass. IAF
   flipped. Coupling: 1 pass both ways.
3. Half the coordinates never transform, so cross-split joint
   structure is unlearnable. Marginals look fine, joints fail.
4. Affine images of Gaussians are Gaussian (characteristic
   function). Two modes are not Gaussian.
5. Causes: log-det overflow to +inf (accumulating determinants, not
   log-dets) and a diagonal hitting zero (log-det to -inf).
   Guards: always accumulate in log space. Clamp conditioner
   outputs away from zero.
Red flags: blaming the learning rate for NaNs. Rubric: both
causes with guards.

## A1

Strong: start from P(x in A) = P(z in f^-1(A)). Write both sides
as integrals. Substitute y = f(z): dy = |det J| dz. The integrands
must agree for all A, giving p_x(f(z)) |det J(z)| = p_z(z).
Rearrange with x = f(z). Red flag: skipping the "for all A"
step. Rubric: the substitution and the rearrangement.

## A2

Strong: log p(y) = log p_z(z) - sum_l l_d summed over dims and
layers. Each layer contributes D diagonal logs: O(D) work, L
layers give O(L x D). Red flag: writing the product instead of
the sum. Rubric: the sum form with the cost.

## D1

Strong: check in order: (1) expressivity: is one affine-ish layer
enough for this data, or does the target need depth/nonlinearity
(C11). (2) the base support versus data support: a mismatch sends
likelihoods to -inf on real points. (3) the sign and reduction of
the log-det sum: a flipped sign or a mean-instead-of-sum silently
shifts every likelihood. Red flag: adding layers before checking
the sign. Rubric: the ordered checks.

## T1

Strong: MAF at 2k/s cannot hit 10k/s on one core. Switch to
coupling layers: parallel both directions, fewer serial steps.
Tradeoff: weaker per layer, so either accept lower likelihood or
add more cheap layers until the latency budget binds. What you
lose: the autoregressive expressivity per layer. Measure: points
per second versus held-out likelihood at each layer count. Red
flag: proposing a bigger MAF. Rubric: the architecture switch
with the measured tradeoff.

## T2

Strong: minimal change: stack more coupling layers with
alternating splits and nonlinear conditioners, or switch the base
to a mixture. The experiment: held-out likelihood versus layer
count, plus a mode-count check on samples (e.g. cluster the
samples and count occupied modes). Proved when all 5 modes appear
in samples and the likelihood plateaus. Red flag: widening a
single layer. Rubric: the change plus the two-part proof.

## R1

Strong: reason 1: exactness is about the number, not the ranking.
A misspecified flow gives exact likelihoods under the wrong
model. Anomalies then mean "unlikely under my wrong model". Reason
2: likelihood rewards covering mass, so a flow that smears mass
widely scores anomalies as merely unlikely while a tighter model
flags them. The experiment: inject labeled anomalies into held-out
data, compare flow likelihood ranking against a baseline
detector's ranking by AUROC, with matched compute. Exactness wins
only if the ranking wins. Red flag: "exact is always better".
Rubric: two valid reasons plus the ranking experiment.
