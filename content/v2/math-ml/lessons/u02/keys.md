# Answer keys, U02 lesson

Date: 2026-10-06. Kept separate from the lesson per the assessment rule.
Test-mode: read the lesson first, answer closed-book, then check here.
All numbers computed 2026-10-06, numpy 1.26.4, float64, unless stated.

## E01

span([1, 0], [2, 0]): every combination is (a + 2b)*[1, 0], so the span
is the x-axis line t*[1, 0]. Dimension 1. The second arrow adds no new
direction.

## E02

The unit disk is not a subspace. Take [0.7, 0] in the disk. Doubling
gives [1.4, 0], outside the disk. Closure under scaling fails.
Minimum sufficient: one arrow whose double leaves the set disqualifies
it.

## E03

[1, 0, -1] = [1, 1, 0] - [0, 1, 1], so the three arrows are dependent.
Dimension is 2. The first two are independent: neither is a multiple of
the other.

## E04

Rank 2. Row 3 = 2 * row 2 - row 1 (check: 2*4-1 = 7, 2*5-2 = 8,
2*6-3 = 9). Two independent rows, one dependent. numpy matrix_rank
confirms 2.

## E05

Nullspace of [[1, 1], [1, 1]]: arrows [x, y] with x + y = 0, so the
span of [-1, 1]. Nullity 1, rank 1, and 1 + 1 = 2.

## E06

Failure diagnosis: treating the rank-1 matrix as invertible. The
inverse does not exist. Np.linalg.inv raises LinAlgError, and any
hand-rolled adjugate divides by det = 0. Fix: check the sigma gap
(sigma_2 = 1.986e-16 is dust) and use the pseudoinverse, which
returns the minimum-norm least-squares answer instead of crashing.

## E07

[2, 3] dot [-3, 2] = -6 + 6 = 0. The arrows are orthogonal: zero
agreement, 90 degrees apart.

## E08

||[6, 8]|| = sqrt(36 + 64) = 10. Unit vector [0.6, 0.8]. Check:
0.6^2 + 0.8^2 = 1.0.

## E09

Project [5, 0] onto [1, 1]: dot = 5, a dot a = 2, p = 2.5 * [1, 1] =
[2.5, 2.5]. Residual [2.5, -2.5], dot with [1, 1] = 0.0. The shadow
is correct.

## E10

Projection onto the zero arrow divides by a dot a = 0. No answer
exists. Guard: assert a dot a > 0 before the formula, or return an
explicit error. Silent NaN is the failure mode to avoid.

## E11

R @ [2, 0] = [0, 2]. The 90-degree rotation moves the x-axis arrow to
the y-axis, length preserved.

## E12

Any rank-1 matrix maps the square to a line segment. Example:
[[1, 2], [2, 4]] sends every corner onto span([1, 2]). The area is 0
and no inverse exists.

## E13

det([[3, 0], [0, 4]]) = 3*4 - 0*0 = 12. The matrix scales every area
by 12.

## E14

Tiny determinant does not mean bad conditioning. Counterexample:
1e-6 * I has det = 1e-12 but kappa = 1, perfectly conditioned.
The determinant measures volume. Kappa measures error amplification.
Never judge invertibility quality by the determinant.

## E15

x = [5/3, 5/3] = [1.66666667, 1.66666667] measured. Residual
norm(D @ x - b) = 0.0.

## E16

pinv([[1, 0], [0, 0]]) = [[1, 0], [0, 0]] itself: it inverts the kept
direction and returns zero on the killed direction. Measured
[1. 0. 0. 0.].

## E17

Debug: np.linalg.inv on the singular Z raises LinAlgError: Singular
matrix. Fix: replace inv(Z) @ b with lstsq/pinv, which return the
minimum-norm least-squares solution. Root cause: code assumed
invertibility that the data did not have.

## E18

Eigenvalues 4 and 9, eigenvectors [1, 0] and [0, 1]. A diagonal
matrix shows its eigenvalues on the diagonal and its eigenvectors as
the axes.

## E19

The 90-degree rotation has no real eigenvectors: every nonzero arrow
changes direction. Its eigenvalues are +i and -i (complex). Over R
the eigen equation has no solution. Over C it does.

## E20

SVD of diag(3, 4): singular values [4.0, 3.0] measured, sorted
descending. U and V are identity up to sign flips. The matrix already
is in its stretch axes.

## E21

Rank-1 approximation error for sigma = [5, 3, 1]: the dropped tail is
[3, 1], error = sqrt(3^2 + 1^2) = sqrt(10) = 3.1622776601683795
measured. General rule: error = norm of the dropped singular values.

## E22

Counterfactual: if sigma_2 were 4.9, the rank-1 error would be 4.9
and the kept energy fraction 25 / (25 + 24.01) = 51 percent. Truncation
would be worthless: nearly half the energy is dropped. Selection
boundary: truncate only when the dropped tail is small relative to the
kept head.

## E23

[[3, 1], [1, 3]] has eigenvalues 4 and 2, both positive, so PSD: yes.
Equivalently Cholesky succeeds.

## E24

A computed minimum eigenvalue of -1e-16 is dust from rounding, not a real
negative curvature. Use min_eig >= -tol with tol ~ 1e-12, not strict
> 0. Strict inequality on floats rejects valid PSD matrices.

## E25

Mean 5. Squared deviations sum: 9+1+1+1+0+0+4+16 = 32. Sample
variance (n - 1 = 7): 32/7 = 4.571428571428571 measured. Population
variance (n = 8): 4.0. The lesson uses the n - 1 convention.

## E26

Correlation = 2.33333333 / sqrt(1.66666667 * 3.33333333) =
0.9899494917519782 measured. Nearly locked: the two features move
together almost perfectly.

## E27

Points on the unit circle have covariance ~0 because positive and
negative co-movements cancel, yet x^2 + y^2 = 1 is a perfect
relation. Zero covariance rules out linear co-movement only, never
dependence. Diagnosis of the failure: the tool assumes linearity.

## E28

kappa([[1, 0], [0, 2]]) = 2/1 = 2. Mild: input wobbles at most double.

## E29

Measured: np.linalg.solve on float16 arrays raises TypeError (array
type float16 is unsupported in linalg). numpy refuses the operation
entirely. Conceptual answer: kappa = 4002 with ~3 decimal digits of
float16 leaves no reliable digits for the sensitive solution
component. The solver must run in float32 or float64. The honest
measured behavior is the refusal, not a wrong answer.

## E30

Research design (open, graded on falsifiability): precondition K by
diagonal scaling D, solve the scaled system, and measure kappa before
and after. Hypothesis: kappa drops below 100. Baseline: kappa =
4002.000750124841. Metric: measured kappa of the scaled matrix and
the input/output shake ratio on the [2, 2] -> [2, 2.000001] test.
Failure criterion: hypothesis rejected if measured kappa stays above
1000 or the shake ratio does not improve.

## Ladder answer sketches L01-L10

Each ladder is graded define -> toy -> derive -> implement ->
complexity -> compare -> debug -> critique -> design. Strong answers
name the assumption being tested at each rung and quote one measured
number from this lesson (for example kappa = 4002.0, sigma_2 =
1.986e-16, correlation 0.9899). Common red flag: reciting the
definition without the toy numbers. Rubric: 2 points per rung, 18
points per ladder. Remediation for a failed rung is the matching
R-section in prerequisites.md.
