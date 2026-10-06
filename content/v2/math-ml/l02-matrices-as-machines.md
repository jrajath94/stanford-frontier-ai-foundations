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
video_id: VviJTNbznjE
video_title: "Lecture 02: Basics of Matrix Algebra (NPTEL)"
video_caption: "The NPTEL lecture this chapter follows: matrix algebra, the multiplication rule, and the transpose."
concepts: [matrix, matrix-multiplication, linear-transformation, projection, orthogonal-complement, transpose, composition]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning: Lecture 02 (Basics of Matrix Algebra)"
    url: https://www.youtube.com/watch?v=VviJTNbznjE
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
multiplies matrices millions of times per second. It cannot
accelerate your custom Python loop.

The matrix fixes both. Composition becomes multiplication: to apply
A then B, multiply the matrices once (BA) and apply the result.
Batching becomes free: stack vectors as columns of a matrix X, and
MX transforms the whole batch in one call.

## The key question

What is the one multiplication rule from which rotation, scaling,
projection, and every neural network layer all follow?

### Subchapter: the multiplication rule, by hand

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

Which view when? The row view computes: it is how code does it.
The column view understands: it shows what the transformation
does to space. Interviewers switch views mid-question. Practice
both until the switch is instant.

![A matrix rotates the vector](assets/plate-l02-rotation.svg "Rows dot the vector: [0 -1. 1 0] turns [1,2] into [-2,1]. Shell 2. Source: original toy. Project: Stanford Frontier AI.")

![Columns are where the axes land](assets/plate-l02-columns.svg "Mv = 3*[2,0] + 4*[1,3] = [10,12]. Same answer as rows dot v. Shell 2. Source: original toy. Project: Stanford Frontier AI.")

### Subchapter: the four matrices you will meet everywhere

```ascii
identity: [ 1 0 ]          scale x by 2: [ 2 0 ]
          [ 0 1 ]                         [ 0 1 ]

rotation 90 deg: [ 0 -1 ]  projection onto x-axis: [ 1 0 ]
                 [ 1  0 ]                          [ 0 0 ]
```

The identity does nothing: Iv = v. The scaler stretches one axis.
The rotation turns the plane. The projection kills the y-component:
[3, 4] becomes [3, 0]. The vector's shadow on the x-axis.

Memorize these four the way you memorize multiplication tables.
Every matrix you meet in ML is a combination of these moves, and
recognizing them by sight is the difference between reading a
formula and understanding it.

### Subchapter: composition is multiplication

Apply A (rotate 90) then B (scale by 2) to v = [1, 2]:

```ascii
A v = [-2, 1]          (rotate)
B(Av) = [-4, 2]        (scale)
```

Now multiply the matrices once: BA = [0 -2. 2 0]. Apply to v:
(BA)v = [-4, 2]. Same answer. The product BA is the single matrix
that does both steps. This is why matrix multiplication is defined
the "strange" way: rows times columns is exactly the rule that
makes (BA)v = B(Av) come out right.

Order matters. AB means "apply B first, then A": the rightmost
matrix acts first. And AB almost never equals BA: rotate-then-scale
differs from scale-then-rotate. Assuming commutativity is the
classic bug in derivations.

![Composition is multiplication](assets/plate-l02-compose.svg "Rotate then scale-2: one matrix BA does both. Shell 3. Source: original toy. Project: Stanford Frontier AI.")

### Subchapter: batching is free

Stack 1,000 vectors as columns of a 2x1000 matrix X. One call, MX,
transforms all of them. The GPU sees a rectangle and parallelizes
across columns. This is why neural networks process batches, not
examples: the hardware's native language is matrix-matrix
multiplication, and a batch is just a wide matrix.

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
that space. The projection is the prediction. The leftover is the
error, perpendicular to everything the model can express. The
normal equation X^T X w = X^T y is the projection formula written
in matrix form.

![Projection: the shadow formula](assets/plate-l02-projection.svg "[3,4] onto the x-axis is [3,0]. The leftover [0,4] is perpendicular. Shell 3. Source: original toy. Project: Stanford Frontier AI.")

### Subchapter: the orthogonal complement

The orthogonal complement of a line in R^2 is the perpendicular
line. Of a plane in R^3, the perpendicular line. In general: the
set of all vectors perpendicular to every vector in the space.
Dimension check: a k-dimensional space in R^n has an (n-k)
dimensional complement. Nothing overlaps, nothing is missing:
every vector splits uniquely into a part in the space and a part
in its complement, and the two parts add back to the original.

Why care? The complement is where the error lives. In least
squares, the residual is perpendicular to the column space: it
sits entirely in the complement. "The error is orthogonal to the
model's reach" is the geometric sentence the normal equation
encodes.

### Subchapter: the transpose

The **transpose** M^T flips a matrix across its diagonal: rows
become columns. [2, 1. 0, 3]^T = [2, 0. 1, 3]. Two facts matter:

- (AB)^T = B^T A^T. Order flips, like taking off shoes then socks.
- M^T M is always symmetric and never has negative eigenvalues.
  Covariance matrices are built this way: X^T X.

The transpose is the quiet hero of the next lessons: it appears in
the normal equation (L09), in covariance (L03), and in the SVD
(L04). Whenever you see M^T M, read "M measuring itself": the
(i,j) entry is column i dot column j, the agreement between two
features.

![Transpose flips rows and columns](assets/plate-l02-transpose.svg "(AB)^T = B^T A^T: both equal [5 1. 2 1]. Order flips. Shell 2. Source: original toy. Project: Stanford Frontier AI.")

| Matrix | What it does to [1, 2] | Columns = images of axes |
|---|---|---|
| Identity | [1, 2], unchanged | axes stay put |
| [2 0; 0 1] | [2, 2], x stretched | x-axis doubles |
| [0 -1; 1 0] | [-2, 1], rotated 90 deg | axes swap and flip |
| [1 0; 0 0] | [1, 0], shadow on x-axis | y-axis collapses to zero |

## What is used where: the real systems

| Math idea | Where it appears | Why there |
|---|---|---|
| Matrix multiply | Linear layers (every network) | Wx + b is the layer; batches make it matrix-matrix |
| Wq, Wk, Wv | Attention (CS229S L02) | three learned projections per layer |
| Embedding table | Token lookup (CS336) | a matrix whose rows are token vectors |
| Composition BA | Deep stacks | layers compose by multiplication |
| Projection | Least squares (L09) | prediction is the shadow on the column space |
| M^T M | Covariance, normal equation | symmetric, nonnegative eigenvalues |

![Matrices: what is used where](assets/plate-l02-used-where.svg "Every neural layer is one of these multiplications. Shell 5. Source: public model docs. Project: Stanford Frontier AI.")

> [!QA]
> Q: What does it mean to say a matrix is a linear transformation?
> A: It means the matrix keeps the origin fixed and straight lines straight. Doubling the input doubles the output. Transforming a sum equals the sum of the transforms. In the toy, M = [2 1. 0 3] sent [3, 4] to [10, 12], and it would send [6, 8] to [20, 24]: exactly double. Neural network layers are matrices plus a nonlinearity, so this linearity is the skeleton under every layer.
> Follow-up: Is every function from R^n to R^m a matrix?
> A: No. Only the linear ones. The moment you add a constant shift or a curve (like a sigmoid), no single matrix can express it. That is why neural nets alternate: matrix, nonlinearity, matrix, nonlinearity.

> [!QA]
> Q: Walk me through matrix-vector multiplication two ways.
> A: Take M = [2 1. 0 3], v = [3, 4]. Row view: row 1 dots v: 2*3 + 1*4 = 10. Row 2 dots v: 0*3 + 3*4 = 12. Answer [10, 12]. Column view: 3 * [2, 0] + 4 * [1, 3] = [6, 0] + [4, 12] = [10, 12]. Same answer. The row view is how you compute. The column view shows the answer is a mix of the columns, weighted by v.
> Follow-up: What does that tell you about the columns?
> A: They are where the axes land. Column 1 is M[1, 0] = [2, 0]: the x-axis lands there. Column 2 is M[0, 1] = [1, 3]: the y-axis lands there. A matrix is fully described by where it sends the axes.

> [!QA]
> Q: Why is matrix multiplication defined so strangely, rows times columns?
> A: Because it means composition. If M rotates and N scales, then NM applied to v means "scale, then rotate", and the row-times-column rule makes (NM)v = N(Mv) come out right. The column view is the intuitive one: Mv is a linear combination of M's columns, weighted by v.
> Follow-up: Does AB equal BA?
> A: Almost never. Rotate then scale differs from scale then rotate. Matrix multiplication is associative, (AB)C = A(BC), but not commutative. Interviewers probe this because assuming commutativity is the classic bug in derivations.

> [!QA]
> Q: What is a projection, and why does least squares need it?
> A: A projection drops a perpendicular from a vector onto a line or plane and takes the landing point. In the toy, [3, 4] projected onto the x-axis is [3, 0], with leftover [0, 4] perpendicular to the axis. Least squares is this in high dimensions: the target vector y rarely lies in the feature matrix's column space, so you project it there. The projection is the prediction. The perpendicular leftover is the error.
> Follow-up: What is the orthogonal complement?
> A: Everything perpendicular to the given space. For the x-axis in R^2, it is the y-axis. Every vector splits uniquely into a part in the space and a part in its complement, and the two parts add back to the original. No information is lost in the split.

> [!QA]
> Q: Prove (AB)^T = B^T A^T on a concrete pair.
> A: Take A = [1 2. 0 1], B = [3 0. 1 1]. AB = [5 2. 1 1], so (AB)^T = [5 1. 2 1]. Now B^T A^T = [3 1. 0 1] [1 0. 2 1] = [5 1. 2 1]. Equal. The intuition: transposing reverses the order of operations, like undoing "socks then shoes" as "shoes then socks."
> Follow-up: Why is M^T M special?
> A: It is always symmetric, and its eigenvalues are never negative. The (i,j) entry is column i dot column j: the matrix measuring its own columns' agreements. Covariance matrices and the normal equation's X^T X are built exactly this way, which is why both inherit those nice properties.

> [!QA]
> Q: Why do neural networks process batches instead of single examples?
> A: Because the hardware's native language is matrix-matrix multiplication. Stack 1,000 vectors as columns of X. One call MX transforms all of them, and the GPU parallelizes across columns. Single-example loops leave thousands of cores idle. Batching is not a convenience: it is how you feed the machine rectangles.
> Follow-up: Does batching change the math?
> A: No. Each column is transformed independently: column j of MX is M times column j of X. The batch is just bookkeeping that the hardware can parallelize. Gradients average over the batch the same way.

> [!QA]
> Q: You need to apply "rotate 90, then scale x by 2, then project onto the x-axis" to a million vectors. How?
> A: Multiply the three matrices once, then apply the single product to the batch. R = [0 -1. 1 0], S = [2 0. 0 1], P = [1 0. 0 0]. The composed matrix PSR = [0 -2. 0 0]: one multiplication per vector instead of three, and one batched call for all million. Composition first, application second: that order is the whole point of the multiplication rule.
> Follow-up: The pipeline changes per request. Now what?
> A: Then precomputation buys nothing and you apply the matrices in sequence. The decision rule: compose when the pipeline is fixed and reused. Stream when it varies. Same math, different economics.

## Recap: the whole lesson on one screen

1. **The task.** Transform whole datasets at once: rotate, scale, project, score.
2. **First attempt.** Write each transformation as code. It works once.
3. **Where it breaks.** Chaining by hand breeds index bugs. GPUs need rectangles, not loops.
4. **The key question.** One multiplication rule for rotation, scaling, projection, and network layers?
5. **The new idea.** Rows dot the vector. Equivalently, the answer is a mix of the columns weighted by the input. Columns are where the axes land.
6. **The four matrices.** Identity, scale, rotation, projection: memorize by sight.
7. **Composition.** (BA)v = B(Av). Order matters. AB != BA. Batching is free: MX transforms all columns at once.
8. **Projection by hand.** [3, 4] onto the x-axis: ((a.b)/(a.a)) * a = [3, 0]. Leftover [0, 4] is perpendicular. The complement holds the error.
9. **The transpose.** (AB)^T = B^T A^T, verified on numbers. M^T M is symmetric with nonnegative eigenvalues.
10. **The price and the bridge.** Matrices only do linear maps. Curves need nonlinearities between layers. L03 asks what happens when a matrix acts on its own favorite vectors: eigenvalues.

## Go deeper

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/kYB8IZa5AuE" title="3Blue1Brown: Linear transformations and matrices (Essence of linear algebra, chapter 3)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

- 3Blue1Brown, "Linear transformations and matrices" (Essence of linear algebra, ch. 3; the embed above): https://www.youtube.com/watch?v=kYB8IZa5AuE
- The NPTEL lecture for this lesson (frontmatter video): https://www.youtube.com/watch?v=VviJTNbznjE
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 2.3-2.4 (free): https://mml-book.github.io: matrix multiplication, transpose, linear maps.
- Strang, "Introduction to Linear Algebra", ch. 2 (elimination), ch. 4 (orthogonality, projections).

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lecture 02 (this lesson's video): [paper](https://www.youtube.com/watch?v=VviJTNbznjE)
- NPTEL course page (111107137): https://nptel.ac.in/courses/111107137

**Further reading:**
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 2.3-2.4 (free):
  - [matrix multiplication, transpose, linear maps.](https://mml-book.github.io)
- Strang, "Introduction to Linear Algebra", ch. 2 (elimination), ch. 4 (orthogonality, projections).

**Caveats.** Projection formula and transpose identities are standard. The toy numbers are the lesson's own. The lecture's exact worked examples are [uncertain] (transcripts for L02/L06/L08 were not recovered).

## Connections to the other courses

- **CS229 L02/L03:** least squares is projection. The normal equation is the projection formula in matrix form. L09 works it.
- **CS229S L02:** the attention score matrix QK^T is a matrix product. Softmax rows are linear combinations of value rows.
- **CS336:** every linear layer is a matrix multiply. The embedding table is a matrix whose rows are token vectors.
