---
page_id: math-ml-l03
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 3
nav: "L03 · Eigen-decomposition"
title: "Lecture 3: Eigen-decomposition and PCA"
summary: "Some vectors survive a matrix unchanged in direction: eigenvectors. A 2x2 eigen-decomposition by hand, then PCA on four data points with real variance numbers."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [eigenvalue, eigenvector, eigendecomposition, spectral-decomposition, covariance, pca]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 09 (Eigenvalues and Eigenvectors), 10 (Special matrices), 11 (Spectral Decomposition), 16-17 (PCA)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 4"
    url: https://mml-book.github.io
  - tag: supplement
    label: "Strang, Introduction to Linear Algebra, ch. 6"
---

## The task: find the directions a matrix leaves alone

Apply a matrix to a vector and the vector usually rotates to a new
direction. But some matrices have special vectors that refuse to
rotate: the matrix only stretches or shrinks them. Those vectors
are **eigenvectors**, and the stretch factors are **eigenvalues**.

Why hunt for them? Because a matrix's action on its eigenvectors
reveals its whole character. In ML the starring example is the
**covariance matrix** of a dataset. Its eigenvectors are the
directions of greatest spread in the data, and its eigenvalues say
how much spread each direction holds. That is **PCA**, principal
component analysis: the most used dimensionality reduction in ML
(CS229 L10).

## First attempt: watch what a diagonal matrix does

Take the simplest matrix, a diagonal one:

```ascii
A = [ 2  0 ]    v = [ 1 ]        A v = [ 2*1 + 0*0 ] = [ 2 ] = 2 * [ 1 ]
    [ 0  3 ]        [ 0 ]              [ 0*1 + 3*0 ]   [ 0 ]         [ 0 ]
```

The x-axis vector [1, 0] stays on the x-axis, stretched by 2. The
y-axis vector [0, 1] stays on the y-axis, stretched by 3. Both are
eigenvectors; the eigenvalues are 2 and 3. For a diagonal matrix,
life is easy: the axes are the eigenvectors and the diagonal holds
the eigenvalues.

But real covariance matrices are not diagonal. The data's spread
runs at an angle. So we need the general method.

## The key question

Given any square matrix, how do you find the vectors it only
stretches, and by how much?

## The new idea: solve Av = lambda v

An eigenvector v and eigenvalue lambda (the Greek letter lambda)
satisfy one equation:

```ascii
A v = lambda * v
```

The matrix acts, and the vector comes back pointing the same way,
scaled by lambda. Rearranged: (A - lambda*I) v = 0, where I is the
identity. A nonzero v solves this only when (A - lambda*I) squashes
some direction flat, which happens exactly when its determinant is
zero. That condition, det(A - lambda*I) = 0, is the **characteristic
equation**. Its roots are the eigenvalues.

Watch it on a real 2x2, by hand. The matrix stretches x by 2 and
shears upward:

```ascii
A = [ 2  1 ]
    [ 0  3 ]

step 1, characteristic equation:
  det([ 2-lambda,  1        ]) = (2-lambda)(3-lambda) - 0 = 0
     ([ 0,         3-lambda ])
  eigenvalues: lambda1 = 3, lambda2 = 2

step 2, eigenvector for lambda1 = 3:
  [ 2-3  1 ] [ x ] = [ -x + y ] = [ 0 ]   -->  y = x
  [ 0   0 ] [ y ]   [   0    ]   [ 0 ]
  eigenvector: v1 = [ 1, 1 ]     check: A v1 = [ 3, 3 ] = 3 * v1

step 3, eigenvector for lambda2 = 2:
  [ 0  1 ] [ x ] = [ y ] = [ 0 ]   -->  y = 0
  [ 0  1 ] [ y ]   [ y ]   [ 0 ]
  eigenvector: v2 = [ 1, 0 ]     check: A v2 = [ 2, 0 ] = 2 * v2
```

Read the result. Along [1, 1] the matrix triples everything. Along
[1, 0] it doubles everything. Every other vector is a mix of these
two, so the matrix's whole behavior is captured by two directions
and two numbers. That capture has a name: the **eigen-decomposition**
(or **spectral decomposition** for symmetric matrices),

```ascii
A = V Lambda V^-1
```

V holds the eigenvectors as columns, Lambda is diagonal with the
eigenvalues. The matrix equals: change into the eigenvector basis,
stretch each axis by its eigenvalue, change back.

## Where it breaks: not every matrix cooperates

Two honest limits. First, only square matrices have eigenvalues in
this sense. A 3x2 data matrix has no eigenvectors; that is why L04
builds the SVD instead. Second, some square matrices are
**defective**: they do not have enough independent eigenvectors to
fill V. The playlist covers this in Lectures 29-30 (Jordan form)
[uncertain: exact scope unknown]. For symmetric matrices, which
include every covariance matrix, neither problem occurs: the
eigenvectors are perpendicular and complete. That is the spectral
theorem, and it is why PCA always works.

## PCA by hand: four points, real variance numbers

Now the payoff. Four 2-D data points, already centered (mean
subtracted):

```ascii
points:  [ 1, 1 ], [ 2, 2 ], [ 3, 3 ], [ -6, -6 ]?  -- no, use a spread
```

Use a cleaner set with spread in two directions:

```ascii
points: p1 = [ 2, 0 ], p2 = [ -2, 0 ], p3 = [ 0, 1 ], p4 = [ 0, -1 ]
```

Step 1, covariance matrix. Covariance measures how two coordinates
vary together: average of (x_i * y_i) over the points.

```ascii
var(x) = (4 + 4 + 0 + 0) / 4 = 2
var(y) = (0 + 0 + 1 + 1) / 4 = 0.5
cov(x,y) = (0 + 0 + 0 + 0) / 4 = 0

C = [ 2    0  ]
    [ 0   0.5 ]
```

Step 2, eigen-decompose C. It is diagonal: eigenvalues 2 and 0.5,
eigenvectors [1, 0] and [0, 1].

Step 3, read the answer. The first principal component is the
x-axis: it carries variance 2 out of 2.5 total, i.e. 80% of the
data's spread. Drop the y-axis and each point keeps 80% of the
information while the data shrinks from 2-D to 1-D. That is PCA:
keep the top eigenvectors of the covariance matrix, project the
data onto them, discard the rest.

In real PCA the covariance is not diagonal, so the components run
at angles, but the recipe is identical: eigen-decompose the
covariance, sort by eigenvalue, keep the top k.

| Concept | Definition | In the toy |
|---|---|---|
| Eigenvector | v with Av = lambda v, direction unchanged | [1, 1] and [1, 0] |
| Eigenvalue | the stretch factor lambda | 3 and 2 |
| Eigen-decomposition | A = V Lambda V^-1 | change basis, stretch, change back |
| Covariance matrix | C_ij = average of (coord_i * coord_j) | [2 0; 0 0.5] |
| PCA | keep top eigenvectors of C | x-axis keeps 80% of variance |

> [!QA]
> Q: What is an eigenvector, in one concrete picture?
> A: A vector a matrix only stretches, never rotates. In the toy, A = [2 1; 0 3] sends [1, 1] to [3, 3]: same direction, triple the length. So [1, 1] is an eigenvector with eigenvalue 3. Every matrix action decomposes into stretches along its eigenvectors, which is why A = V Lambda V^-1 tells you everything about A.
> Follow-up: Can eigenvalues be negative or zero?
> A: Yes. Negative means the matrix flips the direction (a 180-degree turn plus stretch). Zero means the matrix crushes that direction flat: the eigenvector lands on the zero vector. A zero eigenvalue of a covariance matrix means the data has no spread at all in that direction, so PCA drops it for free.

> [!QA]
> Q: How does PCA actually reduce dimensions?
> A: It finds the directions of greatest variance and keeps only those. In the four-point toy, the covariance matrix [2 0; 0 0.5] has eigenvalues 2 and 0.5: the x-axis holds 80% of the variance, the y-axis 20%. Keeping only the x-component shrinks each point from 2 numbers to 1 while preserving 80% of the spread. Real PCA does the same with a non-diagonal covariance.
> Follow-up: Why the covariance matrix and not the data matrix itself?
> A: Because variance is the question. PCA asks "along which directions does the data spread most?" and the covariance matrix stores exactly the spread in every direction pair. Its eigenvectors are the spread directions, its eigenvalues the spread amounts. The data matrix is not square, so it has no eigenvectors at all: that gap is what the SVD in L04 fills.

> [!QA]
> Q: Why do symmetric matrices get special treatment?
> A: Because their eigenvectors are perpendicular and complete: every symmetric matrix decomposes cleanly with no defective cases. Covariance matrices are symmetric by construction (C = X^T X up to scaling), so PCA's eigen-decomposition always exists and the components come out orthogonal. That orthogonality is why PCA features are uncorrelated, which later models appreciate.
> Follow-up: What breaks if you skip centering the data before PCA?
> A: The first component chases the mean instead of the spread. Without centering, the direction from the origin to the data cloud dominates the variance, and PCA reports "the data sits over there" rather than "the data spreads along this axis". Always subtract the mean first.

## Recap: the whole lesson on one screen

1. **The task.** Find the directions a matrix only stretches: they reveal its character.
2. **First attempt.** Diagonal matrices make it trivial: axes are eigenvectors, diagonal holds eigenvalues.
3. **The key question.** How to find them for any square matrix?
4. **The new idea.** Solve Av = lambda v via det(A - lambda I) = 0. Toy: eigenvalues 3 and 2, eigenvectors [1, 1] and [1, 0], each verified by multiplication.
5. **The capture.** A = V Lambda V^-1: change basis, stretch, change back.
6. **Where it breaks.** Non-square matrices have no eigenvectors (L04's SVD answers); defective matrices lack a full set (symmetric matrices never do).
7. **PCA by hand.** Four points, covariance [2 0; 0 0.5], top component holds 80% of variance. Drop the rest.
8. **The price and the bridge.** Eigen-decomposition needs square matrices. Real data matrices are rectangular. L04 generalizes the idea to every matrix: the SVD.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 09-11, 16-17:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 4 (free):
  https://mml-book.github.io — eigendecomposition, PCA derivation.
- Strang, "Introduction to Linear Algebra", ch. 6 — eigenvalues, symmetric matrices, PCA.

**Caveats.** The toy matrices and the four-point PCA are the lesson's own worked examples. The lecture's exact examples are [uncertain] (transcripts for L09-L11, L16-L17 were not recovered). The PCA recipe (eigen-decompose covariance, keep top components) matches the playlist's lecture titles and standard treatment.

## Connections to the other courses

- **CS229 L10:** PCA and factor analysis, the full ML treatment; this lesson is its math prerequisite.
- **CS229S L06:** low-rank approximation via SVD generalizes this lesson's truncation idea to non-square matrices.
- **CS336:** embedding matrices are often compressed with the same eigen/SVD tools.
