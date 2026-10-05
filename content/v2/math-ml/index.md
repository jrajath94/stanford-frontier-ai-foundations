---
page_id: math-ml-index
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 0
nav: "MATH-ML · Overview"
title: "Mathematical Foundations of Machine Learning"
summary: "The math that makes the ML courses work: linear algebra, probability, calculus, optimization, and information theory, each tied to a concrete ML use with hand-worked numbers."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [linear-algebra, probability, calculus, optimization, information-theory]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — full playlist (60 lectures)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "NPTEL course page (111107137)"
    url: https://archive.nptel.ac.in/courses/111/107/111107137/
---

## The math that makes the ML courses work

Every machine learning course in this system rests on five
pillars: linear algebra, probability, calculus, optimization, and
information theory. This course builds each pillar from zero, with
hand-worked numbers at every step, and ties each one to the exact
ML use that needs it.

The backbone is the NPTEL course "Essential Mathematics for
Machine Learning" (Prof. Sanjeev Kumar, lectures 1-20: linear
algebra; Prof. S. K. Gupta, lectures 21-60: calculus,
optimization, probability), 60 lectures on
[YouTube](https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE).
Each lesson below maps to its lectures and to the ML course that
consumes the math.

| Lesson | Topic | Playlist lectures | ML tie-in |
|---|---|---|---|
| [L01 · Vectors as Data](l01-vectors-as-data.md) | dot product, norm, cosine similarity | 01, 02, 07 | embeddings, attention scores (CS229S L02) |
| [L02 · Matrices as Machines](l02-matrices-as-machines.md) | matrix multiplication, projection | 02, 06, 08 | linear layers, least squares geometry |
| [L03 · Eigen-decomposition and PCA](l03-eigendecomposition-pca.md) | Av = lambda v, spectral theorem | 09-11, 16-17 | PCA (CS229 L10) |
| [L04 · SVD and Low Rank](l04-svd-low-rank.md) | A = U Sigma V^T, Eckart-Young | 12-14 | low-rank compression (CS229S L06) |
| [L05 · Probability and Bayes](l05-probability-bayes.md) | Bayes' theorem, expectation, variance | 47-50 | naive Bayes (CS229 L05) |
| [L06 · Distributions and Covariance](l06-distributions-covariance.md) | Bernoulli, Binomial, Gaussian, MLE | 51-53 | Gaussian MLE, counting as estimation |
| [L07 · Calculus for ML](l07-calculus-gradients-chainrule.md) | gradient, Jacobian, chain rule, backprop | 31-36 | backpropagation (CS229 L08) |
| [L08 · Convexity and Gradient Descent](l08-convexity-gradient-descent.md) | convex functions, GD traces, logistic loss | 24-26, 37-40, 44 | logistic regression training |
| [L09 · Least Squares](l09-least-squares-regression.md) | normal equation, projection | 21-23 | linear regression (CS229 L02) |
| [L10 · Information Theory](l10-information-theory.md) | entropy, KL, cross-entropy | supplement [uncertain] | softmax loss (CS229 L08, CS336) |

Also in this course: the [cheatsheet](cheatsheet.md) (every
formula on one page) and the [crash course](crash-course.md)
(the whole arc in thirty minutes).

**Reading order.** L01 -> L02 -> L03 -> L04 is the linear algebra
spine. L05 -> L06 is the probability spine. L07 -> L08 is the
calculus-to-optimization spine. L09 uses L02's projection;
L10 uses L05's probability. The spines are independent: start
with whichever your next ML course needs.
