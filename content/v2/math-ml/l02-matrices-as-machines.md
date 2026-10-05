---
page_id: math-ml-l02
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 2
nav: "L02 · Matrices as Machines"
title: "Lecture 2: Matrices as Machines"
summary: "A matrix is a machine that transforms vectors: one multiplication rotates, scales, and projects. Projection by hand, and why least squares is just a shadow."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [matrix, matrix-multiplication, linear-transformation, projection, orthogonal-complement, transpose]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 02 (Matrix Algebra), 06 (Linear Transformations), 08 (Orthogonal Complement and Projection Mapping)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 2"
    url: https://mml-book.github.io
  - tag: supplement
    label: "Strang, Introduction to Linear Algebra, ch. 2-3"
---

## The task: transform a whole dataset at once

L01 turned objects into vectors. Now a harder job: transform every
vector in a dataset the same way. Rotate all photos by 90 degrees.
Compress all embeddings from 512 numbers to 50. Score all examples
with one classifier.

Doing this vector by vector is slow and unreadable. The lecture's
answer is a **matrix**: a rectangular grid of numbers that acts as
a machine. Feed it a vector, it returns a transformed vector. Feed
it a million vectors, it transforms all of them.

```ascii
matrix M (2 x 2)      vector v        result
[ 0  -1 ]             [ 1 ]           [ 0*1 + -1*2 ]   [ -2 ]
[ 1   0 ]    x        [ 2 ]    =      [ 1*1 +  0*2 ] = [  1 ]
```

That matrix rotated [1, 2] by 90 degrees. The rule for the
multiplication: each row of the matrix takes a dot product with the
vector. Row 1 dot v is the first answer, row 2 dot v is the second.
A 2x2 matrix maps R^2 to R^2. An m x n matrix maps R^n to R^m:
n numbers in, m numbers out.

A matrix is also a **linear transformation**: it keeps straight
lines straight and the origin fixed. Doubling the input doubles the
output. Adding two inputs then transforming equals transforming
then adding. That is all "linear" means here.

## First attempt: write the transformation as code

Before matrices, you might write the rotation as a function:

```ascii
rotate(v): return [-v[1], v[0]]
```

This works for one transformation. But now compose two: rotate,
then scale by 2. The code becomes nested calls, and for a
10,000-dimensional embedding it is unreadable. Worse, a GPU cannot
see the structure: it needs the transformation as data, not as
code, so it can run it on thousands of vectors in parallel.

## Where code breaks: composition and batches

Two problems. First, chaining transformations by hand is error
prone: three nested functions for rotate-scale-project, each with
its own index bugs. Second, hardware wants rectangles. A GPU
multiplies matrices millions of times per second; it cannot
accelerate your custom Python loop.

The matrix fixes both. Composition becomes multiplication: to apply
A then B, multiply the matrices once (BA) and apply the result.
Batching becomes free: stack vectors as columns of a matrix X, and
MX transforms the whole batch in one call.

## The key question

What is the one multiplication rule from which rotation, scaling,
projection, and every neural network layer all follow?

## The new idea: rows dot the vector, columns are the images of the axes

Watch the rule once, slowly, on a 2x2 matrix:

```ascii
M = [ 2  1 ]    v = [ 3 ]       M v = [ 2*3 + 1*4 ] = [ 10 ]
    [ 0  3 ]        [ 4 ]             [ 0*3 + 3*4 ]   [ 12 ]
```

Read it two ways. Row view: each row dots with v. Column view:
M v = 3 * [2, 0] + 4 * [1, 3]. The answer is a linear combination
of the matrix's columns, weighted by v's components. So the
columns of M are where the axes land: the x-axis [1, 0] lands on
column 1, the y-axis [0, 1] lands on column 2. A matrix is fully
described by where it sends the axes.

Special cases you will meet everywhere:

```ascii
identity: [ 1 0 ]          scale x by 2: [ 2 0 ]
          [ 0 1 ]                         [ 0 1 ]

rotation 90 deg: [ 0 -1 ]  projection onto x-axis: [ 1 0 ]
                 [ 1  0 ]                          [ 0 0 ]
```

The identity does nothing. The projection matrix kills the
y-component: [3, 4] becomes [3, 0]. The vector's shadow on the
x-axis.

## Projection by hand: the shadow formula

**Projection** answers: what is the closest point on a line to a
given vector? Drop the perpendicular from the vector to the line.
The foot of that perpendicular is the projection.

Project b = [3, 4] onto the direction a = [1, 0] (the x-axis).
The formula: scale a by the amount of b that lies along a.

```ascii
projection = ((a . b) / (a . a)) * a

a . b = 1*3 + 0*4 = 3
a . a = 1*1 + 0*0 = 1
projection = (3 / 1) * [1, 0] = [3, 0]
```

The leftover, b minus the projection = [0, 4], is perpendicular to
the line. That leftover has a name: it lives in the **orthogonal
complement**, the set of all vectors perpendicular to the line.
Every vector splits into a part along the line and a part
perpendicular to it. Nothing is lost: [3, 0] + [0, 4] = [3, 4].

Now the payoff. Least squares regression (L09) is exactly this
picture in high dimensions. The data vector y usually does not lie
in the column space of the feature matrix X. So you project y onto
that space. The projection is the prediction; the leftover is the
error, perpendicular to everything the model can express. The
normal equation X^T X w = X^T y is the projection formula written
in matrix form.

## The transpose: flip the rectangle

The **transpose** M^T flips a matrix across its diagonal: rows
become columns. [2, 1; 0, 3]^T = [2, 0; 1, 3]. Two facts matter:

- (AB)^T = B^T A^T. Order flips, like taking off shoes then socks.
- M^T M is always symmetric and never has negative eigenvalues.
  Covariance matrices are built this way: X^T X.

The lecture introduces the transpose with matrix algebra in
Lecture 02; it returns as the hero of least squares (L09) and PCA
(L03), where X^T X appears in both.

| Matrix | What it does to [1, 2] | Columns = images of axes |
|---|---|---|
| Identity | [1, 2], unchanged | axes stay put |
| [2 0; 0 1] | [2, 2], x stretched | x-axis doubles |
| [0 -1; 1 0] | [-2, 1], rotated 90 deg | axes swap and flip |
| [1 0; 0 0] | [1, 0], shadow on x-axis | y-axis collapses to zero |

> [!QA]
> Q: What does it mean to say a matrix is a linear transformation?
> A: It means the matrix keeps the origin fixed and straight lines straight. Doubling the input doubles the output; transforming a sum equals the sum of the transforms. In the toy, M = [2 1; 0 3] sent [3, 4] to [10, 12], and it would send [6, 8] to [20, 24]: exactly double. Neural network layers are matrices plus a nonlinearity, so this linearity is the skeleton under every layer.
> Follow-up: Is every function from R^n to R^m a matrix?
> A: No. Only the linear ones. The moment you add a constant shift or a curve (like a sigmoid), no single matrix can express it. That is why neural nets alternate: matrix, nonlinearity, matrix, nonlinearity.

> [!QA]
> Q: Why is matrix multiplication defined so strangely, rows times columns?
> A: Because it means composition. If M rotates and N scales, then NM applied to v means "scale, then rotate", and the row-times-column rule makes (NM)v = N(Mv) come out right. The column view is the intuitive one: Mv is a linear combination of M's columns, weighted by v. The lecture's vector-space block makes this the bridge to basis and dimension.
> Follow-up: Does AB equal BA?
> A: Almost never. Rotate then scale differs from scale then rotate. Matrix multiplication is associative, (AB)C = A(BC), but not commutative. Interviewers probe this because assuming commutativity is the classic bug in derivations.

> [!QA]
> Q: What is a projection, and why does least squares need it?
> A: A projection drops a perpendicular from a vector onto a line or plane and takes the landing point. In the toy, [3, 4] projected onto the x-axis is [3, 0], with leftover [0, 4] perpendicular to the axis. Least squares is this in high dimensions: the target vector y rarely lies in the feature matrix's column space, so you project it there. The projection is the prediction; the perpendicular leftover is the error.
> Follow-up: What is the orthogonal complement?
> A: Everything perpendicular to the given space. For the x-axis in R^2, it is the y-axis. Every vector splits uniquely into a part in the space and a part in its complement, and the two parts add back to the original. No information is lost in the split.

## Recap: the whole lesson on one screen

1. **The task.** Transform whole datasets at once: rotate, scale, project, score.
2. **First attempt.** Write each transformation as code. It works once.
3. **Where it breaks.** Chaining by hand breeds index bugs; GPUs need rectangles, not loops.
4. **The key question.** One multiplication rule for rotation, scaling, projection, and network layers?
5. **The new idea.** Rows dot the vector; equivalently, the answer is a mix of the columns weighted by the input. Columns are where the axes land.
6. **Projection by hand.** [3, 4] onto the x-axis: ((a.b)/(a.a)) * a = [3, 0]. Leftover [0, 4] is perpendicular.
7. **The payoff.** Least squares is projection in high dimensions: prediction is the shadow, error is perpendicular to it.
8. **The price and the bridge.** Matrices only do linear maps; curves need nonlinearities between layers. L03 asks what happens when a matrix acts on its own favorite vectors: eigenvalues.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 02, 06, 08:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 2.3-2.4 (free):
  https://mml-book.github.io — matrix multiplication, transpose, linear maps.
- Strang, "Introduction to Linear Algebra", ch. 2 (elimination), ch. 4 (orthogonality, projections).

**Caveats.** Projection formula and transpose identities are standard; the toy numbers are the lesson's own. The lecture's exact worked examples are [uncertain] (transcripts for L02/L06/L08 were not recovered).

## Connections to the other courses

- **CS229 L02/L03:** least squares is projection; the normal equation is the projection formula in matrix form. L09 works it.
- **CS229S L02:** the attention score matrix QK^T is a matrix product; softmax rows are linear combinations of value rows.
- **CS336:** every linear layer is a matrix multiply; the embedding table is a matrix whose rows are token vectors.
