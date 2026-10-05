---
page_id: math-ml-l09
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 9
nav: "L09 · Least Squares"
title: "Lecture 9: Least Squares and Linear Regression"
summary: "The oldest learning algorithm: fit a line by projection. The normal equation solved by hand on three points, and why more features than examples breaks it."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [least-squares, normal-equation, linear-regression, overdetermined, projection]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 21 (Least Square Approximation), 22-23 (Linear and Multiple Regression)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Strang, Introduction to Linear Algebra, ch. 4.3-4.4"
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 9"
    url: https://mml-book.github.io
---

## The task: draw the best line through noisy points

Three measurements of a spring: force x (newtons), stretch y (cm).

```ascii
(1, 1.1), (2, 1.9), (3, 3.2)
```

Hooke's law says stretch is proportional to force: y = w x. Find
the w that best explains the data. "Best" needs a definition
first: minimize the sum of squared errors.

```ascii
loss(w) = (w*1 - 1.1)^2 + (w*2 - 1.9)^2 + (w*3 - 3.2)^2
```

Why squares? Three reasons: they punish large errors more, they
are smooth (differentiable, unlike absolute value), and under
Gaussian noise they are the maximum likelihood estimator (L06).
Least squares is the oldest learning algorithm in statistics, and
it is still the first model every ML course fits (CS229 L02).

## First attempt: guess and check

Try w = 1.0: errors -0.1, -0.1, -0.2; squared sum = 0.01 + 0.01 +
0.04 = 0.06. Try w = 1.05: predictions 1.05, 2.1, 3.15; errors
-0.05, -0.2, -0.05; squared sum = 0.0025 + 0.04 + 0.0025 = 0.045.
Better. Try w = 1.1: 0.0025 + 0.09 + 0.0025... wait, predictions
1.1, 2.2, 3.3; errors 0, -0.3, -0.1; squared sum = 0 + 0.09 + 0.01
= 0.10. Worse. The best w sits near 1.05. Guessing works on one
parameter. With 50 features it is hopeless.

## The key question

Is there a formula for the best w, no guessing, no iteration?

## The new idea: differentiate and set to zero

The loss is convex in w (a sum of squares, L08's bowl). So the
minimum is where the derivative is zero. For the general problem,
write it in matrix form. X is the n x d feature matrix, y the n
targets:

```ascii
loss(w) = ||Xw - y||^2

gradient:  2 X^T (Xw - y) = 0
normal equation:  X^T X w = X^T y
```

One linear system solves all of regression. If X^T X is
invertible: w = (X^T X)^-1 X^T y. No iteration, no learning rate,
exact answer.

Solve the spring toy. X = [1, 2, 3]^T, y = [1.1, 1.9, 3.2]^T:

```ascii
X^T X = 1 + 4 + 9 = 14
X^T y = 1*1.1 + 2*1.9 + 3*3.2 = 1.1 + 3.8 + 9.6 = 14.5
w = 14.5 / 14 = 1.0357
```

Best slope: 1.036 cm per newton. Squared error at this w:
predictions 1.036, 2.071, 3.107; errors 0.064, -0.171, -0.093;
sum of squares = 0.0041 + 0.0294 + 0.0086 = 0.0421. Lower than
every guess. The formula beat the guessing.

## The geometry: it was projection all along

L02 promised this. The vector y = [1.1, 1.9, 3.2] lives in R^3.
The model's predictions Xw live in the column space of X: here,
the line spanned by [1, 2, 3]. y is not on that line (the points
are noisy). Least squares projects y onto the line. The projection
is the prediction; the leftover is perpendicular to the line.

Verify perpendicularity. Prediction p = 1.0357 * [1, 2, 3] =
[1.036, 2.071, 3.107]. Leftover r = y - p = [0.064, -0.171,
0.093]. Dot with the line direction [1, 2, 3]:

```ascii
r . [1,2,3] = 0.064*1 + (-0.171)*2 + 0.093*3
            = 0.064 - 0.342 + 0.279 = 0.001 ~ 0
```

Zero (up to rounding). The error is perpendicular to everything
the model can express. That is the normal equation's geometric
meaning: X^T (Xw - y) = 0 says the residual is orthogonal to every
column of X.

## Where it breaks: more features than sense

Two failure modes, both with numbers.

**Dependent features.** Add a second feature that is twice the
first: X = [[1, 2], [2, 4], [3, 6]]. Then X^T X = [[14, 28], [28,
56]]: row 2 is twice row 1. Determinant zero. Not invertible. The
normal equation has infinitely many solutions: w = [1.036, 0] and
w = [0, 0.518] fit identically. L01's linear dependence returns as
a singular matrix. Fix: drop the duplicate, or regularize (CS229
L03).

**More features than examples.** 100 features, 10 examples. X^T X
is 100x100 but built from 10 outer products: rank at most 10 (L04).
Ninety directions are unconstrained; infinite solutions fit the
training data perfectly and generalize terribly. This is
overfitting in its purest form. The playlist's "minimum normed
solution" (Lecture 21) picks the shortest w among the infinite
candidates [uncertain: exact lecture treatment unknown].

| Idea | Formula | Meaning |
|---|---|---|
| Least squares | min \|\|Xw - y\|\|^2 | best linear fit under squared error |
| Normal equation | X^T X w = X^T y | one linear system; exact solution |
| Geometry | project y onto column space of X | prediction is the shadow; error perpendicular |
| Dependent features | det(X^T X) = 0 | infinite solutions; drop or regularize |
| Min-norm solution | shortest w among fits | the tie-breaker when data underdetermines w |

> [!QA]
> Q: Derive the normal equation.
> A: Start from loss(w) = ||Xw - y||^2. Differentiate with respect to w: the gradient is 2 X^T (Xw - y). The loss is convex, so the minimum is where the gradient is zero: X^T X w = X^T y. In the spring toy, X^T X = 14 and X^T y = 14.5, giving w = 1.0357 directly. No iteration needed.
> Follow-up: When does the normal equation have no unique solution?
> A: When X^T X is singular: dependent columns (a duplicate feature makes the determinant zero, as in the [[14,28],[28,56]] toy) or more features than examples (rank at most n < d). Then infinite weight vectors fit equally well. The standard fixes are dropping redundant features, regularization, or the minimum-norm solution.

> [!QA]
> Q: What does least squares have to do with projection?
> A: Everything: least squares IS projection. The target y rarely lies in the feature matrix's column space, so you project it there; the projection is the prediction. In the toy, the residual [0.064, -0.171, 0.093] dots with the column [1, 2, 3] to give ~0: the error is perpendicular to the model's reach. The normal equation X^T(Xw - y) = 0 is exactly this orthogonality statement.
> Follow-up: Why minimize squared error rather than absolute error?
> A: Three reasons: squares punish large errors more, squares are smooth (absolute value has a kink at zero, breaking gradient methods), and under Gaussian noise least squares is the MLE (L06). Absolute error is more robust to outliers but harder to optimize: the classic accuracy-robustness trade.

> [!QA]
> Q: 100 features, 10 examples. What goes wrong?
> A: X^T X is 100x100 with rank at most 10: 90 directions unconstrained. Infinite weight vectors fit the 10 training points perfectly, most of them wild. Training error zero, test error catastrophic: overfitting in pure form. The minimum-norm solution picks the shortest such w, and regularization (CS229 L03) penalizes large weights to choose among the candidates sensibly.
> Follow-up: How do you detect this before training?
> A: Compare n and d. If d > n, expect singularity. Numerically, check the smallest eigenvalue of X^T X (L03): near zero means trouble. In practice, regularize by default when features outnumber examples.

## Recap: the whole lesson on one screen

1. **The task.** Fit y = w x to three spring measurements. Loss: sum of squared errors.
2. **First attempt.** Guess w: 1.0 gives 0.06, 1.05 gives 0.045, 1.1 gives 0.10. Works for one parameter.
3. **The key question.** A formula, no guessing?
4. **The new idea.** Differentiate, set to zero: X^T X w = X^T y. Toy: w = 14.5/14 = 1.0357, error 0.0421, beating every guess.
5. **The geometry.** Projection: residual dots the column space to ~0. Prediction is the shadow; error is perpendicular.
6. **Where it breaks.** Dependent features: det = 0, infinite solutions. More features than examples: rank <= n < d, overfitting in pure form.
7. **The fixes.** Drop duplicates, regularize, or take the minimum-norm solution.
8. **The price and the bridge.** Closed form costs O(d^3) via matrix inversion; with millions of features, iterate (L08) instead. L10 names the loss logistic regression minimizes when squares are the wrong noise model.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 21-23:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Strang, "Introduction to Linear Algebra", ch. 4.3-4.4 — least squares as projection.
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 9 (free):
  https://mml-book.github.io — linear regression, MLE view.

**Caveats.** The spring toy and all numbers are the lesson's own. The lecture's exact examples (L21-L23) are [uncertain] (transcripts not recovered).

## Connections to the other courses

- **CS229 L02/L03:** linear regression, the normal equation, regularization, and the probabilistic (Gaussian MLE) interpretation.
- **CS229 L10:** PCA uses the same projection machinery on the covariance matrix.
- **CS336:** linear layers solve least-squares-like problems at every step; the projection view explains residual streams.
