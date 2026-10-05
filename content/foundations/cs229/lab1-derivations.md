---
title: "Lab 1: Derivations by Hand"
course: cs229
type: lab
---

Three derivations, pen and paper, 45 minutes. These are the three most-asked ML derivations in frontier-lab interviews.

## Problem 1: Normal equations (10 min)

Given \(X \in \mathbb{R}^{m \times n}\) and \(y \in \mathbb{R}^m\), derive the closed-form minimizer of

\[ J(\theta) = \frac{1}{2} \|X\theta - y\|^2. \]

Show each step: expand the norm, take the gradient with respect to \(\theta\), set to zero, solve.

**Check:** your answer should be \(\theta = (X^T X)^{-1} X^T y\). State the condition for the inverse to exist.

**Interview trap:** the interviewer will ask what happens when \(X^T X\) is singular. Answer: use the pseudoinverse, or add ridge regularization \(\lambda I\), which also connects to the Bayesian view in L06.

## Problem 2: Logistic regression gradient (15 min)

For binary labels \(y^{(i)} \in \{0, 1\}\) and hypothesis \(h_\theta(x) = 1/(1 + e^{-\theta^T x})\), derive the gradient of the log-likelihood

\[ \ell(\theta) = \sum_{i=1}^{m} y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)})). \]

**Hint:** first show \(\frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z))\). Then apply the chain rule. The final form should look familiar.

**Check:** \(\nabla_\theta \ell(\theta) = \sum_i (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}\). Same error-times-input form as LMS. If your derivation does not collapse to this, find the algebra mistake.

**Interview trap:** "Why not use squared error for logistic regression?" Answer: the resulting objective is non-convex. The log-likelihood (cross-entropy) is convex in \(\theta\).

## Problem 3: Backprop for one dense layer (20 min)

One training example. Forward pass:

\[ z = Wx + b, \quad a = \sigma(z), \quad L = \frac{1}{2}\|a - y\|^2. \]

Derive \(\frac{\partial L}{\partial W}\), \(\frac{\partial L}{\partial b}\), and \(\frac{\partial L}{\partial x}\) using the backward-function method from L08: compute the upstream gradient, multiply by the local Jacobian.

**Check:** \(\frac{\partial L}{\partial W} = \delta x^T\) where \(\delta = (a - y) \odot \sigma'(z)\). This is the outer-product / Hebbian form.

**Interview trap:** "What is the rank of the weight gradient for a single example?" Answer: rank 1. The batch gradient is a sum of rank-1 terms. Interviewers use this to test whether you understand the structure or just memorized the formula.

## Scoring

- 3/3 with clean algebra: strong hire signal on ML fundamentals.
- 2/3: review the one you missed in the corresponding lesson, then redo.
- Fewer: do not move to Lab 2. Redo this lab tomorrow.
