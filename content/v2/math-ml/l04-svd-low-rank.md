---
page_id: math-ml-l04
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 4
nav: "L04 · SVD and Low Rank"
title: "Lecture 4: SVD and Low-Rank Approximation"
summary: "Every matrix, square or not, splits into rotation, stretch, rotation. A full SVD worked by hand on a 2x2, and why dropping a singular value is the cheapest compression in ML."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [svd, singular-value, low-rank, eckart-young, frobenius-norm, rank]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 12 (SVD), 13 (SVD Properties and Applications), 14 (Low Rank Approximations)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 4.5"
    url: https://mml-book.github.io
  - tag: supplement
    label: "Strang, Introduction to Linear Algebra, ch. 7"
---

## The task: decompose the matrices eigen-decomposition cannot touch

L03's eigen-decomposition needs a square matrix. But the matrices
that matter in ML are rarely square. A dataset with 1,000 examples
and 50 features is a 1000x50 matrix. A weight layer mapping 512
inputs to 128 outputs is 128x512. None of them have eigenvectors.

The **singular value decomposition**, the SVD, is the
generalization that works on every matrix: every m x n matrix
splits into three simple pieces,

```ascii
A = U Sigma V^T

U:      m x m, a rotation (columns perpendicular, unit length)
Sigma:  m x n, a stretch (diagonal, nonnegative numbers)
V^T:    n x n, a rotation
```

Read it right to left as an action on a vector: rotate into a
special basis (V^T), stretch each axis by a **singular value**
(Sigma), rotate into the output space (U). Every linear map is a
rotation, then a stretch, then a rotation. Nothing else.

## First attempt: force eigenvectors on a rectangle

You might try the eigen trick on A^T A instead. For our data
matrix A, the matrix A^T A is square, symmetric, and has
eigenvectors. That gives V and the singular values (their square
roots). But it loses U: the output-side directions. Half a
decomposition is not enough to rebuild A.

## The key question

What are the input directions and output directions such that A
maps each input direction to exactly one output direction, scaled
by a single number?

## The new idea: build all three pieces from A^T A

The recipe, worked by hand on a small matrix. Take

```ascii
A = [ 4  0 ]
    [ 3  0 ]
```

Two columns: [4, 3] and [0, 0]. One column is dead. Intuition says
this matrix is really one-dimensional. The SVD will prove it.

Step 1, form A^T A (square, symmetric, eigenvectors guaranteed):

```ascii
A^T A = [ 4 3 ] [ 4 0 ] = [ 25  0 ]
        [ 0 0 ] [ 3 0 ]   [  0  0 ]
```

Step 2, eigen-decompose it. Diagonal already: eigenvalues 25 and
0. Singular values are the square roots: sigma1 = 5, sigma2 = 0.
Eigenvectors: v1 = [1, 0], v2 = [0, 1]. These are the columns of V.

Step 3, build U's columns: u_i = A v_i / sigma_i.

```ascii
u1 = A [1,0] / 5 = [4, 3] / 5 = [0.8, 0.6]
u2: sigma2 = 0, division impossible -- pick any unit vector
     perpendicular to u1: u2 = [-0.6, 0.8]
```

Step 4, assemble:

```ascii
A = U Sigma V^T

    [ 0.8  -0.6 ] [ 5  0 ] [ 1  0 ]
  = [ 0.6   0.8 ] [ 0  0 ] [ 0  1 ]

check the (1,1) entry: 0.8*5*1 + (-0.6)*0*0 = 4.  correct.
check the (2,1) entry: 0.6*5*1 + 0.8*0*0 = 3.    correct.
```

The matrix is exactly rotation-stretch-rotation. The zero singular
value exposes the dead column: nothing flows through that
direction. The **rank** of a matrix is its number of nonzero
singular values. This matrix has rank 1.

## Where the SVD earns its keep: low-rank approximation

Here is the turn that matters for ML. Suppose the singular values
are 5 and 1 instead of 5 and 0:

```ascii
B = [ 0.8  -0.6 ] [ 5  0 ] [ 1  0 ]
    [ 0.6   0.8 ] [ 0  1 ] [ 0  1 ]
```

B is full rank. But the second direction carries little: sigma2 =
1 against sigma1 = 5. Drop it. Keep only the rank-1 piece:

```ascii
B_1 = [ 0.8 ] [ 5 ] [ 1  0 ] = [ 4  0 ]
      [ 0.6 ]                     [ 3  0 ]
```

B_1 stores 2 numbers per side plus 1 stretch instead of 4 full
entries. The error, measured by the **Frobenius norm** (square root
of the sum of squared entries, the matrix version of L01's vector
norm), is exactly the dropped singular value:

```ascii
||B - B_1|| = sigma2 = 1
```

This is the **Eckart-Young** result: the best rank-k approximation
of a matrix is its top k SVD pieces, and the error equals the first
dropped singular value. No other rank-k matrix is closer. In real
ML the savings are dramatic: a 4096x4096 weight matrix with the
top 64 singular values kept stores 64*(4096+4096+1) numbers
instead of 16.7 million. That is the low-rank compression behind
LoRA fine-tuning and behind CS229S L06.

## The honest price

The SVD of a big matrix is expensive to compute: roughly cubic in
the dimensions. You never SVD a billion-entry matrix directly.
Practical code uses randomized or iterative methods that find only
the top k pieces. And truncation is lossy by design: if the
singular values decay slowly, dropping pieces costs real accuracy.
The spectrum decides whether compression is free or fatal.

| Idea | Formula | Meaning |
|---|---|---|
| SVD | A = U Sigma V^T | rotation, stretch, rotation; works on any matrix |
| Singular values | sqrt of eigenvalues of A^T A | stretch per direction; sorted largest first |
| Rank | count of nonzero singular values | true dimensionality of the map |
| Best rank-k approx | keep top k pieces | error = first dropped singular value |
| Frobenius norm | sqrt of sum of squared entries | the ruler that measures the error |

> [!QA]
> Q: What does the SVD actually say, in plain words?
> A: Every linear map is a rotation, then a per-axis stretch, then another rotation. For A = [4 0; 3 0], the pieces are U = [0.8 -0.6; 0.6 0.8], Sigma = [5 0; 0 0], V = identity: stretch the x-axis by 5, rotate it to [0.8, 0.6], kill the y-axis entirely. The zeros in Sigma expose dead directions; their count is the rank.
> Follow-up: Why not just use eigenvalues of A directly?
> A: Because A is usually not square. Eigenvalues need square matrices. The SVD's trick is to eigen-decompose A^T A, which is always square and symmetric, then recover both rotations. For square symmetric A the two coincide: singular values are the absolute eigenvalues.

> [!QA]
> Q: Why is truncating the SVD the best compression, not just a decent one?
> A: Because of Eckart-Young: among all rank-k matrices, the top-k SVD pieces minimize the Frobenius error, and the error equals the first dropped singular value. In the toy, dropping sigma2 = 1 from B gives error exactly 1; no other 2-number-per-side approximation beats it. Compression becomes a solved optimization, not a heuristic.
> Follow-up: When does low-rank compression fail?
> A: When the singular values decay slowly. If sigma1 = 5 and sigma2 = 4.9, dropping one direction loses nearly half the energy. Image and weight matrices in practice have fast-decaying spectra, which is why LoRA works. Always look at the spectrum before truncating.

> [!QA]
> Q: What is rank, really?
> A: The number of independent directions a matrix actually uses: the count of nonzero singular values. The toy A has rank 1 despite being 2x2, because its second column is all zeros. Rank is the true dimensionality hiding inside the nominal dimensions, and low-rank approximation exploits the gap between them.
> Follow-up: How is rank different from dimension?
> A: Dimension is the size of the room (2x2 lives in a 4-dimensional space of matrices). Rank is how much of the room the matrix occupies (1 direction). A 1000x50 data matrix has dimension 50,000 entries but might have rank 10: ten true directions, the rest noise or redundancy.

## Recap: the whole lesson on one screen

1. **The task.** Decompose non-square matrices: data tables, weight layers.
2. **First attempt.** Eigen-decompose A^T A. It gives V and the stretches but loses U.
3. **The key question.** Which input directions map to which output directions, each scaled by one number?
4. **The new idea.** A = U Sigma V^T, built from the eigen-decomposition of A^T A plus u_i = A v_i / sigma_i. Toy verified entry by entry: 4 and 3 recovered.
5. **Rank.** Nonzero singular values: the toy has rank 1; the zero exposes the dead column.
6. **Low-rank approximation.** Drop small singular values. Eckart-Young: the top-k pieces are the best rank-k approximation; error = first dropped value (1 in the toy).
7. **The ML payoff.** A 4096x4096 weight matrix compresses to 64 directions: the engine of LoRA and CS229S L06.
8. **The price.** Full SVD is cubic; use iterative top-k methods. Slow-decaying spectra resist compression.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 12-14:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 4.5 (free):
  https://mml-book.github.io — SVD, low-rank approximation.
- Strang, "Introduction to Linear Algebra", ch. 7 — the SVD chapter.

**Caveats.** The toy matrices, the hand computation, and the LoRA/4096 numbers are the lesson's own. The lecture's exact worked examples are [uncertain] (transcripts for L12-L14 were not recovered). Eckart-Young is the standard result behind the playlist's "Low Rank Approximations" lecture.

## Connections to the other courses

- **CS229S L06:** low-rank weight compression and efficient inference; this lesson is the math it rests on.
- **CS229 L10:** PCA via SVD: for centered data, the SVD of the data matrix gives the principal components directly.
- **CS336:** LoRA fine-tuning keeps pretrained weights frozen and trains low-rank updates.
