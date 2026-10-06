# Interview bank, U02 linear geometry and decompositions

Date: 2026-10-06. Questions only. Keys in interview/keys-u02.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Define rank in one sentence, then give a 2 by 2 matrix of rank 1.
B2. What does the nullspace of a matrix contain, in plain words?
B3. State the projection formula and the one check that proves it
right.
B4. What does the determinant measure for a 2 by 2 matrix?
B5. When does a symmetric matrix fail to be PSD?
B6. What is the condition number, and what does a value of 4000 mean
for a linear solve?

## Deep ladder D1, SVD (5 follow-ups)

D1.1. Define the SVD of a real matrix: shapes of all three factors.
D1.2. Toy: for M = [[3, 2, 2], [2, 3, -2]], what are the singular
values, and what is the rank-1 reconstruction error?
D1.3. Derive: why does the rank-k truncation error equal the norm of
the dropped singular values?
D1.4. Implement/debug: a colleague forms M^T M and takes square roots
of its eigenvalues to get singular values of an ill-conditioned M.
What breaks, and what is the fix?
D1.5. Changed constraint: the spectrum is [5, 4.9] instead of
[5, 3]. Is rank-1 truncation still sensible? Justify with energy
fractions.

## Deep ladder D2, conditioning (5 follow-ups)

D2.1. Define the condition number.
D2.2. Toy: K = [[1, 1], [1, 1.001]]. Compute kappa and describe the
measured input/output shake from the lesson.
D2.3. Derive or justify: why is the relative output error bounded by
kappa times the relative input error?
D2.4. Implement/debug: a teammate solves normal equations
(X^T X) beta = X^T y with kappa(X^T X) = 1.6e7 and reports "residual
zero, answer trusted". Diagnose.
D2.5. Research critique: "Preconditioning always fixes conditioning."
Attack the claim: name one case where it fails or backfires.

## Analytical/quantitative (2)

Q1. A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]. Without a computer: find
the rank, exhibit the dependence, and give one nullspace vector.
Q2. The C11 covariance has eigenvalues 4.9777 and 0.0223. Compute the
correlation from the matrix entries and explain in one sentence what
the near-zero eigenvalue says about the data.

## Implementation/debug (1)

T1. This code intends the minimum-norm solution of Z x = b for the
singular Z = [[1, 2], [2, 4]]:

```python
import numpy as np
Z = np.array([[1., 2.], [2., 4.]])
b = np.array([1., 2.])
x = np.linalg.inv(Z) @ b
```

It raises LinAlgError. Fix it two ways, verify each fix with a
residual check, and state which fix you would ship and why.

## Changed-constraint scenarios (2)

S1. Your vectors are complex (MRI k-space data). Which U02 facts
survive unchanged, and which formula must change first?
S2. Your matrix is 1e6 by 1e6 sparse. The full SVD is impossible.
Name the practical replacement for (a) top singular values,
(b) solving a linear system, and the assumption each replacement
needs.

## Research critique (1)

R1. "We replaced the covariance matrix with the correlation matrix
in our PCA pipeline and the top components changed completely.
Correlation is unit-free, so it must be the more correct choice."
Critique: is the change a bug or a modeling decision? What
experiment distinguishes the two?
