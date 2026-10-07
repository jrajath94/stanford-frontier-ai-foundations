# Answer keys: U04

Unit: math-genmodels-U04. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: invertible means one-to-one and onto with a closed-form
undo. f(z) = 2z + 1 inverts as (x-1)/2. g(z) = z^2 fails: g(1) =
g(-1) = 1, two preimages, no inverse. Red flag: calling z^2
invertible "on positives" without stating the domain restriction.
Rubric: both verdicts with the witness pair.

## B2

Strong: the determinant is the volume scaling factor. det = 2 x 0.5
= 1.0. The unit square becomes a 2-by-0.5 rectangle with area still
1. Red flag: reporting the determinant as a length. Rubric: the
number plus the area reading.

## B3

Strong: p_x(x) = p_z(f^-1(x)) x |det J_{f^-1}(x)|. p_x(1) =
0.3989 x 0.5 = 0.1995. Red flag: dropping the Jacobian term.
Rubric: the formula and the number.

## B4

Strong: forward: y1 = 0.5, y2 = -1 x 1.25 + 0.5 = -0.75. Inverse:
x1 = 0.5, x2 = (-0.75 - 0.5)/1.25 = -1.0. Red flag: inverting s and
t instead of using them as coefficients. Rubric: both directions
computed.

## B5

Strong: log p(y) = log p_z(f^-1(y)) - log|det J|. At the toy
point: -2.4629 - 0.2231 = -2.6860 nats. Red flag: the sign error
(+ instead of -). Rubric: the formula with the sign and the number.

## B6

Strong: MAF: density eval is 1 parallel pass, sampling is D serial
steps. IAF: sampling is 1 parallel pass, density eval is D serial
steps. Red flag: claiming one of them is cheap both ways. Rubric:
both models with both directions.

## L1 ladder

1. Define: p_x(x) = p_z(f^-1(x)) |det J_{f^-1}(x)|. Mass is
   conserved. The Jacobian corrects the volume.
2. Compute: J = [[1, 0], [0, 1.25]] at (0.5, -1).
3. Derive: triangular determinant is the diagonal product by row
   expansion. Log-det = sum of log diagonal = log 1.25 = 0.2231.
4. Evaluate: -2.4629 - 0.2231 = -2.6860 nats.
5. The sign: density eval runs the inverse map. Forward expansion
   is inverse contraction, so the correction subtracts.
Red flags: evaluating the Jacobian in the forward direction for a
density. Rubric: all five rungs with the sign explained.

## L2 ladder

1. Define: coupling transforms half the coords with coefficients
   from the other half, parallel both ways. Autoregressive makes
   each coord depend on all previous ones.
2. Costs: coupling O(D) both directions. MAF: density 1 batched
   pass, sampling D serial steps. IAF flipped.
3. Predict: a never-alternated split leaves half the coordinates
   untransformed through the whole stack. Joint structure across
   the split is unlearnable.
4. Prove: the characteristic function of an affine image of a
   Gaussian is Gaussian. The two-mode target is not Gaussian, so
   no affine layer fits it.
5. Choose MAF for scoring: its cheap direction is density eval.
Red flags: choosing IAF for a scoring workload. Rubric: the proof
sketch and the workload match.

## A1

Strong: expand det along the first row of a lower triangular
matrix: only the (1,1) entry times the minor survives. Induct on
D. The coupling Jacobian is lower triangular with diagonal (1,
s(x1)), so log|det| = log 1 + log s(x1) = 0.2231 on the toy. Red
flag: assuming the result without the induction. Rubric: the row
expansion and the application.

## A2

Strong: if z ~ N(mu, Sigma) and x = Az + b, the characteristic
function phi_x(t) = exp(i t^T b) phi_z(A^T t) is Gaussian in t.
Hence x is Gaussian. The two-mode mixture is not Gaussian, so no
affine (A, b) produces it. Red flag: arguing from samples instead
of the law. Rubric: the characteristic-function step and the
conclusion.

## D1

Strong: the error is 0.4462 = 2 x 0.2231, exactly twice the
log-det. The bug is the sign: the code adds log|det| instead of
subtracting it. Fix: log_py = log_pz - logdet. Test that catches
it: evaluate on the toy point with known answer -2.6860, or check
the density integrates to 1 by quadrature. Either fails with the
wrong sign. Red flag: "fixing" by retraining. Rubric: the
2x-log-det diagnosis with the test.

## T1

Strong: replace exp/softplus with s(v) = 1 + v^2 or s(v) = 1 + |v|:
positive by construction, no exp needed. New failure mode:
unbounded growth. Large |v| gives huge log-det contributions and
the likelihood can overflow in the other direction. Mitigation:
clamp the conditioner output range. Red flag: s(v) = v^2, which
hits zero. Rubric: a valid positivity design plus the new failure.

## T2

Strong: dense log-det costs O(D^3): at D = 1024 about 1e9 ops per
layer per point, dead. Triangular costs O(D) = 1024, fine.
Surviving architectures: coupling layers, autoregressive flows,
any structured triangular Jacobian. Dense Jacobians do not
survive. Red flag: proposing to "just use a GPU". Rubric: both
costs with the survivor list.

## R1

Strong: argument 1: exact versus bound. The flow reports an exact
likelihood. The VAE reports a lower bound. The gap alone can
explain the difference. The comparison is rigged until both are
exact or both are bounds. Argument 2: the flow buys exactness with
constraints: invertibility and triangular structure limit
expressivity per layer, and the winning flow may sample orders of
magnitude slower (the asymmetry). "Strictly better" ignores the
workload. Fair experiment: matched parameter budgets, bound-vs-
bound or exact-vs-exact likelihoods, plus sample-quality checks
and sampling throughput on the target workload. Red flag:
declaring victory from one likelihood number. Rubric: two valid
arguments plus a matched experiment.
