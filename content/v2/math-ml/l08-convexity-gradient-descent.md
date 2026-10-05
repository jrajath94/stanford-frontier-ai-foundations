---
page_id: math-ml-l08
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 8
nav: "L08 · Convexity and Gradient Descent"
title: "Lecture 8: Convexity and Gradient Descent"
summary: "Why some losses have exactly one valley and gradient descent cannot miss it. A full GD trace on f(x) = (x-3)^2 with real step-by-step numbers, then the divergent step size that blows up."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [convex-function, convex-set, gradient-descent, learning-rate, logistic-regression, steepest-descent]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 24-26 (Logistic Regression, Classification Metrics), 37-40 (Convex Sets and Functions), 44 (Steepest Descent)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Boyd and Vandenberghe, Convex Optimization, ch. 2-3"
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 7"
    url: https://mml-book.github.io
---

## The task: walk downhill without getting trapped

L07 built the gradient: a compass pointing uphill. Training a
model means walking the other way, downhill, until the loss
bottoms out. But downhill walks can end in a shallow dip while a
deeper valley sits nearby. How do you know the valley you found is
the best one?

**Convexity** answers. A **convex function** is shaped like a
bowl: the line segment between any two points on its graph sits
above the graph. One valley, no traps. For convex functions, every
local minimum is the global minimum, and gradient descent with a
sane step size finds it. Logistic regression's loss is convex.
That is why it trains reliably.

A **convex set** is the domain version: a set containing the whole
segment between any two of its points. A solid disk is convex; a
crescent moon is not. Constraints in optimization are usually
convex sets (Lecture 37) [uncertain: exact lecture examples
unknown].

## First attempt: follow the slope with fixed steps

The algorithm is three lines. Start somewhere. Repeat: step
opposite the gradient. The step length is the **learning rate**,
eta.

```ascii
x_new = x_old - eta * f'(x_old)
```

Watch it on the simplest bowl: f(x) = (x - 3)^2, minimum at x =
3. f'(x) = 2(x - 3). Start at x0 = 0, eta = 0.1:

```ascii
x0 = 0.0     f = 9.0
x1 = 0 - 0.1 * 2(0-3)   = 0.6     f = 5.76
x2 = 0.6 - 0.1 * 2(0.6-3) = 1.08  f = 3.69
x3 = 1.08 - 0.1 * 2(1.08-3) = 1.464   f = 2.36
x4 = 1.464 - 0.1 * 2(1.464-3) = 1.771 f = 1.51
```

Each step covers 20% of the remaining distance: the gap shrinks by
(1 - 2*eta) = 0.8 per step. After 20 steps the gap is 0.8^20 =
0.0115 of the original 3: x is within 0.035 of the minimum.
Geometric convergence, visible in the numbers: 9.0, 5.76, 3.69,
2.36, 1.51. Every step multiplies the loss by 0.64.

## Where it breaks: the step size

Now eta = 1.1, slightly too large. The curvature here is f'' = 2,
and stability needs eta < 1 (half the inverse curvature, in
general: eta < 2 / max curvature):

```ascii
x0 = 0.0
x1 = 0 - 1.1 * 2(0-3)    = 6.6     f = 12.96
x2 = 6.6 - 1.1 * 2(6.6-3) = -1.32  f = 18.66
x3 = -1.32 - 1.1 * 2(-1.32-3) = 8.18  f = 26.86
```

Each step overshoots the valley and lands further out. The loss
grows: 9.0, 12.96, 18.66, 26.86. Divergence, from a step size 11x
instead of 10x too small a change to eyeball. The learning rate is
the most consequential hyperparameter in ML because the boundary
between convergence and explosion is this sharp. (Newton's method,
Lecture 45, adapts the step using curvature; steepest descent,
Lecture 44, is this fixed-step walk.)

## The key question

Which losses guarantee that downhill walking ends at the global
best, and how do you recognize them?

## The new idea: convex means one valley

Test for convex functions of one variable: f''(x) >= 0
everywhere. f(x) = (x-3)^2 has f'' = 2 > 0: convex. f(x) = x^3 has
f'' = 6x, negative for x < 0: not convex.

In higher dimensions the test uses the Hessian, the matrix of
second derivatives: convex means the Hessian never has a negative
eigenvalue (L03's eigenvalues return as curvature detectors).
Gradient descent on a convex function with eta < 2/L, where L is
the maximum curvature, converges to the global minimum. No traps,
one theorem, and the step-size rule that explains the 1.1
disaster: L = 2, so eta must stay below 1.

## The payoff: logistic regression is convex

**Logistic regression** classifies with probabilities. For label y
in {0, 1} and score z = w . x, it predicts P(y=1) = 1 / (1 +
e^-z), the sigmoid. The loss for one example:

```ascii
loss = -[ y log(p) + (1-y) log(1-p) ]    (cross-entropy, L10)
```

This loss is convex in w. Proof sketch via the Hessian test: the
second derivative matrix equals X^T D X with D diagonal and
positive, so its eigenvalues are nonnegative (L02's transpose
fact: M^T M never has negative eigenvalues). One bowl. Gradient
descent from any start, with a sane eta, reaches the single global
minimum. That reliability is why logistic regression is the first
classifier every ML course teaches (CS229 L05) and why the
playlist pairs it with convexity (L24 before L37 in numbering, but
the ideas interlock).

Classification **metrics** (Lecture 26) judge the result: accuracy
(fraction correct), precision (of predicted positives, how many
were truly positive), recall (of true positives, how many were
found). A rare-disease classifier that always says "healthy" has
99% accuracy and 0% recall: L05's base-rate lesson wearing a new
hat.

| Idea | Test | Consequence |
|---|---|---|
| Convex function | f'' >= 0; Hessian eigenvalues >= 0 | one valley; local min = global min |
| Gradient descent | x -= eta * grad | converges if eta < 2/L |
| Too-large eta | toy: 1.1 vs stability limit 1 | overshoot, divergence: 9.0 -> 26.86 |
| Logistic loss | Hessian = X^T D X, D > 0 | convex; GD cannot get trapped |
| Precision / recall | predicted-true vs true-found | accuracy lies on rare classes |

> [!QA]
> Q: Why does convexity matter for training?
> A: It guarantees the valley you walk into is the only valley. For a convex loss, every local minimum is the global minimum, and gradient descent with eta below 2/L converges to it. Logistic regression's loss is convex (its Hessian X^T D X has nonnegative eigenvalues), so it trains reliably from any start. Non-convex losses like deep networks offer no such promise: hence restarts, schedules, and momentum.
> Follow-up: Is the step-size bound eta < 2/L usable in practice?
> A: Rarely directly: L, the maximum curvature, is unknown and changes during training. Practitioners tune eta by search or use adaptive methods (Adam scales each coordinate by its history). But the bound explains every divergence you will ever see: the toy blew up at eta = 1.1 against a limit of 1.0. When loss explodes, the step was too big. Always.

> [!QA]
> Q: Work one gradient descent trace by hand.
> A: f(x) = (x-3)^2, x0 = 0, eta = 0.1. Gradient 2(x-3). Steps: 0 -> 0.6 -> 1.08 -> 1.464 -> 1.771, losses 9.0 -> 5.76 -> 3.69 -> 2.36 -> 1.51. Each step closes 20% of the gap; the loss multiplies by 0.64 per step. After 20 steps x is within 0.035 of the minimum 3. Geometric convergence you can watch.
> Follow-up: Why did eta = 1.1 diverge?
> A: Stability needs eta < 2/L with L = max curvature = f'' = 2, so eta < 1. At 1.1 each step overshoots: 0 -> 6.6 -> -1.32 -> 8.18, losses 9.0 -> 12.96 -> 18.66 -> 26.86. The overshoot grows because the gradient at the landing point is larger than at takeoff. Eleven percent over the limit is enough.

> [!QA]
> Q: Why is logistic regression the canonical first classifier?
> A: Because its loss is convex: one bowl, one minimum, gradient descent cannot get trapped. The model outputs calibrated probabilities via the sigmoid, and the cross-entropy loss penalizes confident wrong answers heavily. It is the simplest model where probability (L05), calculus (L07), and optimization (this lesson) meet.
> Follow-up: Accuracy 99% but the model is useless. How?
> A: Rare classes. A disease classifier that always predicts "healthy" scores 99% accuracy on 1% prevalence with 0% recall: it finds no patients. Report precision and recall alongside accuracy, always. This is L05's base-rate fallacy as a metric.

## Recap: the whole lesson on one screen

1. **The task.** Walk downhill to the loss minimum without getting trapped in a dip.
2. **Convexity.** Bowl-shaped: segment between any two graph points sits above the graph. One valley.
3. **Gradient descent.** x -= eta * grad. Toy trace: 0 -> 0.6 -> 1.08 -> 1.464 -> 1.771; loss 9.0 -> 1.51 in four steps.
4. **Where it breaks.** eta = 1.1 against stability limit 1.0: overshoot grows, loss 9.0 -> 26.86. The learning rate boundary is sharp.
5. **The key question.** Which losses guarantee the global best?
6. **The test.** f'' >= 0; Hessian eigenvalues >= 0. Converges for eta < 2/L.
7. **The payoff.** Logistic regression's loss is convex (Hessian X^T D X, D positive). Reliable training from any start. Metrics: precision and recall, because accuracy lies on rare classes.
8. **The price and the bridge.** Deep networks are not convex: no guarantees, hence the optimizer zoo. L09 applies least squares' closed form where no walking is needed; L10 explains the cross-entropy loss logistic regression minimizes.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 24-26, 37-40, 44:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Boyd and Vandenberghe, "Convex Optimization", ch. 2-3 (free):
  https://web.stanford.edu/~boyd/cvxbook/ — convex sets, functions, optimality.
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 7 (free):
  https://mml-book.github.io — continuous optimization, gradient descent.

**Caveats.** The GD traces, the stability numbers, and the metric examples are the lesson's own. The lecture's exact demos (L44 steepest descent, L24-26 logistic regression) are [uncertain] (transcripts not recovered).

## Connections to the other courses

- **CS229 L05:** logistic regression and Newton's method; this lesson is why its training converges.
- **CS229 L08:** deep networks abandon convexity; backprop (L07) plus the optimizer zoo takes over.
- **CS229S:** learning-rate tuning at scale; the eta boundary governs thousand-GPU training runs.
