---
page_id: math-ml-crash-course
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 12
nav: "MATH-ML · Crash Course"
title: "MATH-ML Crash Course"
summary: "The whole ten-lesson arc in thirty minutes: one worked number per idea, from vectors to cross-entropy."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [crash-course]
sources:
  - tag: synthesis
    label: "Synthesized from MATH-ML L01-L10"
---

## The arc in ten numbers

Machine learning is five kinds of arithmetic wearing one
trenchcoat. Each lesson below is one idea plus the one number
that proves it. Read the number, trust the idea, follow the link
for the full story.

**1. Vectors hold data; the dot product compares it.** Two
reviews score agreement 3 by dot product. But a tripled review
scores 9 with no new content: length fooled the meter. Divide
both lengths out and both score 0.77: cosine similarity compares
direction alone. ([L01](l01-vectors-as-data.md))

**2. Matrices transform data; projection is their kindest act.**
[3, 4] projected onto the x-axis is [3, 0]; the leftover [0, 4]
is perpendicular. Every prediction-error split in ML is this
picture in high dimensions. ([L02](l02-matrices-as-machines.md))

**3. Eigenvectors are the directions a matrix only stretches.**
For [2 1; 0 3]: [1, 1] triples, [1, 0] doubles. PCA
eigen-decomposes the covariance matrix and keeps the top
directions: four points with covariance [2 0; 0 0.5] compress
2-D to 1-D keeping 80% of the variance.
([L03](l03-eigendecomposition-pca.md))

**4. The SVD works on every matrix: rotation, stretch,
rotation.** A = [4 0; 3 0] splits into stretches 5 and 0: rank 1,
one dead direction. Drop a small singular value and the error
equals exactly what you dropped: the cheapest optimal
compression in ML. ([L04](l04-svd-low-rank.md))

**5. Bayes flips what tests tell you into what you want to know.**
A 95%-accurate test on a 1% disease gives an 8.8% posterior, not
95%: 990 false alarms drown 95 true hits. Posterior = likelihood
x prior, renormalized. ([L05](l05-probability-bayes.md))

**6. Three distributions run ML: Bernoulli, Binomial, Gaussian.**
Exactly 3 spam in 10 emails at p = 0.2: 0.201. Fitting parameters
by maximum likelihood is counting: 2 clicks in 5 impressions ->
p = 0.4. Covariance 1.33 on three collinear points; correlated
features inflate variance. ([L06](l06-distributions-covariance.md))

**7. The chain rule trains networks.** Two weights, x = 2:
dy/dw2 = 6, dy/dw1 = 8, each verified by nudging. Backprop is the
chain rule run backward with reuse: one sweep gradients every
weight. ([L07](l07-calculus-gradients-chainrule.md))

**8. Convexity guarantees the valley; the step size decides if
you reach it.** On (x-3)^2 with eta = 0.1: 0 -> 0.6 -> 1.08, loss
x0.64 per step. With eta = 1.1: divergence, loss 9.0 -> 26.86.
Logistic regression's loss is convex: one bowl, no traps.
([L08](l08-convexity-gradient-descent.md))

**9. Least squares is projection with a formula.** Normal
equation X^T X w = X^T y: spring data gives w = 1.0357, residual
perpendicular to the features. Duplicate a feature and the
matrix goes singular: infinite solutions.
([L09](l09-least-squares-regression.md))

**10. Cross-entropy prices predictions in bits.** Truth [1,0],
prediction [0.7, 0.3]: 0.515 bits. Confident and wrong
[0.01, 0.99]: 6.64 bits. This loss trains logistic regression,
softmax classifiers, and every language model.
([L10](l10-information-theory.md)) [uncertain: supplement beyond
the playlist]

## What to read next

- Training a classifier tomorrow? L05 -> L08 -> L10.
- Debugging embeddings or attention? L01 -> L02 -> L07.
- Compressing a model? L03 -> L04.
- The interview version? Every lesson ends with Q&A blocks
  written as an interviewer would ask them.
