# Interview keys, U02

Date: 2026-10-06. Each answer gives the minimum sufficient
explanation, a strong answer, common red flags, a rubric, and
remediation. Computed values: numpy 1.26.4, float64.

## B1

Minimum: rank is the count of linearly independent rows/columns,
equivalently the count of nonzero singular values. Example:
[[1, 2], [2, 4]] has rank 1.
Strong: adds the evidence (sigma_2 = 1.986e-16 is dust) and the
consequence (no inverse, one null direction).
Red flags: "rank is the number of nonzero entries". Confusing rank
with dimension of the ambient space.
Rubric: 1 pt definition, 1 pt correct example, 1 pt evidence.
Remediation: lesson C01-C02, R11.

## B2

Minimum: all arrows x with M x = 0, the directions the matrix kills.
Strong: rank-nullity (rank + nullity = n) and the least-squares
reading (non-uniqueness of solutions).
Red flags: "the zero vector only" as the whole story. No mention of
dimension.
Rubric: 1 pt set description, 1 pt rank-nullity link.
Remediation: lesson C02.

## B3

Minimum: proj_a(b) = (b dot a)/(a dot a) * a. Check (b - p) dot a = 0.
Strong: derives it from minimizing ||b - t a||^2 over t.
Red flags: formula without the check. Projecting without guarding
a = 0.
Rubric: 1 pt formula, 1 pt orthogonality check, 1 pt zero-guard.
Remediation: lesson C04, R13.

## B4

Minimum: the signed area scale factor. Det = 3 means every area
triples.
Strong: zero means the matrix crushes area (singular). Sign means
orientation flip. Tiny det is not the conditioning test.
Red flags: "determinant tells if the matrix is good for solving".
Rubric: 1 pt area meaning, 1 pt zero case.
Remediation: lesson C06.

## B5

Minimum: when some eigenvalue is negative, equivalently when some x
gives x^T A x < 0. Example [[1, 2], [2, 1]] at [1, -1] gives -2.0.
Strong: Cholesky fails as the independent test. Notes the -1e-16
rounding tolerance.
Red flags: "when the determinant is negative" (wrong test in 3D+).
Rubric: 1 pt eigenvalue test, 1 pt counterexample.
Remediation: lesson C10.

## B6

Minimum: kappa = sigma_max/sigma_min, the worst-case error
amplifier. kappa = 4000 means input noise can grow ~4000x in the
answer.
Strong: quotes the measured shake (3.54e-07 in -> 7.07e-04 out,
bound 4002 holds) and the float-precision consequence.
Red flags: "kappa measures speed". Trusting a zero residual.
Rubric: 1 pt definition, 1 pt interpretation, 1 pt measured link.
Remediation: lesson C12, R20-R21.

## D1

D1.1. Minimum: M = U Sigma V^T. M (m,n), U (m,m) orthogonal, Sigma
(m,n) diagonal nonneg, V^T (n,n) orthogonal.
D1.2. Minimum: singular values [5, 3]. Rank-1 error 3.0 = sigma_2.
Strong: states M_1 = 5 u_1 v_1^T and checks ||M - M_1||_F = 3.0.
D1.3. Minimum: the Eckart-Young argument in one line: the tail
energy is sum of dropped sigma^2, and no rank-k matrix can do
better. Strong: sketches the proof via the SVD basis.
D1.4. Minimum: forming M^T M squares kappa (here kappa 1.67 ->
2.78 in squares) and destroys small singular values. Fix: SVD of M
directly. Red flag: "it works fine for small matrices" without the
kappa-squared argument.
D1.5. Minimum: no. Kept energy 25/49.01 = 51 percent. Truncation
drops nearly half. Strong: computes both fractions and states the
selection boundary (truncate only when the tail is small).
Rubric: 2 pts per rung, 10 total. Remediation per failed rung:
C09, R19, R22.

## D2

D2.1. Minimum: kappa = sigma_max/sigma_min >= 1, worst-case relative
error amplification of a linear solve.
D2.2. Minimum: kappa = 4002.000750124841. Measured shake 3.54e-07
-> 7.07e-04, about 2000x, under the bound.
D2.3. Minimum: from M(x + dx) = b + db, dx = M^{-1} db, take norms
and use ||M^{-1}|| ||M|| = kappa. Strong: notes it is worst-case,
typical error smaller.
D2.4. Minimum: diagnosis: kappa(X^T X) = kappa(X)^2 ~ 1.6e7 means
the normal equations squared the conditioning. A zero residual does
not imply an accurate beta. Fix: QR or SVD solve on X directly.
Red flag: "the solver converged, so the answer is right".
D2.5. Minimum: attack: a bad preconditioner (wrong scale, dense
inverse of a sparse matrix) can raise cost or break structure.
preconditioning changes the problem's geometry, and a poor choice
amplifies the wrong modes. Strong: demands the experiment (kappa
before/after on the actual matrix).
Rubric: 2 pts per rung. Remediation: C12, R20-R21.

## Q1

Rank 2: row 3 = 2*row 2 - row 1. Nullspace: solve. One vector is
[1, -2, 1] (check: row dot = 1-4+3 = 0, 4-10+6 = 0, 7-16+9 = 0).
Minimum sufficient: the dependence relation plus one verified
null vector. Red flag: claiming rank 3 "because 3 by 3".

## Q2

Correlation 0.9899494917519782 measured. The near-zero eigenvalue
0.0223 says the data is nearly one-dimensional: almost all spread
lies along one direction, the second feature is nearly a linear
function of the first.

## T1

Fix 1: x = np.linalg.lstsq(Z, b, rcond=None)[0] (minimum-norm
least-squares). Fix 2: x = np.linalg.pinv(Z) @ b. Verify:
norm(Z @ x - b) ~ 0 and, for the minimum-norm claim, compare
norm(x) against another particular solution. Ship lstsq: one call,
no explicit matrix, standard. Strong: explains why inv is wrong
(singular => no inverse exists) rather than just catching the
exception. Red flag: wrapping inv in try/except and returning zeros.

## S1

Surviving: rank, nullspace, SVD, subspaces, projection geometry.
First formula to change: the dot product needs the conjugate
transpose (Hermitian inner product). Norms use |z|^2. Symmetric
becomes Hermitian. Eigenvalues stay real for Hermitian matrices.

## S2

(a) Randomized/truncated SVD or power iteration for the top-k
singular values. Needs a fast matrix-vector product and a spectral
gap for quick convergence. (b) Iterative solvers (conjugate gradient
for SPD, GMRES/MINRES otherwise). Needs a good preconditioner and a
tolerance, and gives an approximate answer, not an exact one.

## R1

Critique: it is a modeling decision, not a bug, but it must be
justified. Covariance PCA finds max-variance directions in original
units. Correlation PCA standardizes first, so a high-variance
feature no longer dominates. Minimum: state which question each
answers. Strong: the distinguishing experiment: run both, then check
which set of components predicts the held-out target better (or
which matches the domain expert's notion of "important"). Also check
whether feature units are commensurable. Red flag: "unit-free is
always more correct" without a task metric.
