---
course_slug: cs229
course_name: Machine Learning (Stanford CS229)
course_order: 2
instructor: Tengyu Ma and Andrew Ng
date: '2026-10-06'
title: 'Cheatsheet: formulas, shapes, traps, decision rules'
nav: CS229 · Cheatsheet
page_id: cs229-cheatsheet
order: 181
summary: Every key formula, shape, trap, and one-line decision rule for the course.
---
# Cheatsheet: formulas, shapes, traps, decision rules

One dense page for the whole course. Formulas use the notation fixed in
[notation and shapes](notation_and_shapes.html). Section pointers name
the lesson that proves each line.

## U01, learning problems and risk

| Formula / shape | Trap | Decision rule |
|---|---|---|
| X: m x n, m examples, n features | Transposed X flips examples and features | Say "m by n" out loud before you code |
| Empirical risk: (1/m) sum loss | Optimizing it is not the goal | Watch the gap, not the train loss |
| Population risk: E[loss] | Unobservable directly | Estimate with a clean test set |

## U02, linear regression

| Formula / shape | Trap | Decision rule |
|---|---|---|
| theta := theta - alpha X^T(X theta - y)/m | alpha too large diverges | Sweep alpha on a log grid |
| theta = (X^T X)^{-1} X^T y | X^T X singular or ill-conditioned | Check cond before you invert |
| Gaussian noise => least squares is MLE | Real noise is not Gaussian | Treat it as a modeling choice |
| Local weighting: fit per query | O(n) memory and O(n) per query | Use for small n only |

## U03, classification and GLMs

| Formula / shape | Trap | Decision rule |
|---|---|---|
| h = 1/(1 + e^{-theta^T x}) | Overflow at extreme logits | Use the log-sum-exp stable form |
| grad = (y - h) x | Same shape as LMS, different h | Check the link before the gradient |
| Softmax: e^{z_i} / sum e^{z_j} | Forgetting the normalization | Rows must sum to 1 |
| Separable data => ||theta|| -> inf | Unregularized training never converges | Add l2 or stop early |

## U04, generative classifiers

| Formula / shape | Trap | Decision rule |
|---|---|---|
| p(y|x) = p(x|y)p(y)/p(x) | Wrong prior ruins the posterior | Estimate phi from data, do not guess |
| GDA shared Sigma => linear boundary | Separate covariances => quadratic | Test which fits with CV |
| Naive Bayes: features indep | Correlated features double-count | Check correlation first |
| Laplace smoothing | Zero counts kill products | Smooth before you multiply |

## U05, kernels

| Formula / shape | Trap | Decision rule |
|---|---|---|
| K(x,z) = <phi(x),phi(z)> | Kernel must be PSD | Check eigenvalues on a subset |
| Gram matrix: n x n | O(n^2) memory | 1M points = 8 TB: do not build it |
| Representer: w = sum alpha_i phi(x_i) | Solution needs all n alphas | Sparsity comes from the loss, not free |

## U06, SVMs

| Formula / shape | Trap | Decision rule |
|---|---|---|
| Geometric margin = y(w^T x+b)/||w|| | Functional margin scales freely | Constrain ||w||, then maximize |
| Dual in alpha_i, 0 <= alpha_i <= C | All alpha at C means C too small | Raise C or accept noise |
| SMO: 2 variables at a time | Kernel cache thrash | Keep the Gram rows you reuse |
| Support vectors define the boundary | Outliers become support vectors | Inspect them, do not worship them |

## U07, neural nets

| Formula / shape | Trap | Decision rule |
|---|---|---|
| dL/dW from dL/dY via chain rule | Shape mismatch in batched code | Write shapes on every line |
| ReLU: max(0, z) | Dead units give zero gradient | Monitor fraction of active units |
| Init: Var-preserving scale | Wrong scale => vanish/explode | Match init to the activation |
| Finite differences check | Too large epsilon lies | Use 1e-5 relative error below 1e-5 |

## U08, generalization

| Formula / shape | Trap | Decision rule |
|---|---|---|
| Err = bias^2 + var + noise | One number hides the split | Plot learning curves |
| Union bound: log|H|/m | Vacuous for huge H | Use it for intuition, not guarantees |
| Double descent peak at d = n | Tuned-away peak looks like no peak | Report the lambda you used |

## U09, regularization

| Formula / shape | Trap | Decision rule |
|---|---|---|
| Ridge: (X^T X + lambda I)^{-1} X^T y | lambda by habit | Choose by CV |
| LASSO zeroes coordinates | Exact zeros need coordinate descent | Do not expect zeros from SGD |
| CV error estimates test error | Tuning on test leaks | Lock the test set until the end |
| Early stopping | Stopping rule is a hyperparameter | Fix it before you compare |

## U10, clustering and EM

| Formula / shape | Trap | Decision rule |
|---|---|---|
| k-means: assign, recompute | Local minima | Restart many times |
| EM: E responsibilities, M maximize | Variance collapse => inf likelihood | Floor the variances |
| Likelihood never decreases | "Never decreases" is not "converges to MLE" | It finds a local max |

## U11, PCA, ICA, VI

| Formula / shape | Trap | Decision rule |
|---|---|---|
| PCA: eig of centered covariance | Forgetting to center | Center first, always |
| Keep k with 95%+ variance | Variance kept is not task signal | Validate downstream |
| ICA needs non-Gaussian sources | Gaussian mixtures cannot separate | Test non-Gaussianity first |
| ELBO <= log p(x) | Loose bound, good samples | Report the bound gap when you can |

## U12, diffusion

| Formula / shape | Trap | Decision rule |
|---|---|---|
| x_t = sqrt(alpha-bar_t) x_0 + sqrt(1-alpha-bar_t) eps | Schedule bug kills the signal | Assert alpha-bar_T = 0 |
| Denoising loss: ||eps - eps_theta||^2 | Predicting x0 vs eps changes optimization | Pick eps-pred by default |
| Reverse: p(x_{t-1}|x_t) Gaussian | Wrong variance schedule | Use the fixed posterior variance |

## U13, foundation models

| Formula / shape | Trap | Decision rule |
|---|---|---|
| LoRA: W + B A, r << d | r too small starves adaptation | Sweep r in {4, 8, 16} |
| InfoNCE: -log e^{s+}/sum e^{s} | Easy negatives teach nothing | Mine hard negatives |
| Linear probe accuracy | Probes the features, not the task | Compare with full fine-tune |
| RAG: retrieve then generate | Retrieval errors become answer errors | Evaluate retrieval separately |

## U14, LLMs

| Formula / shape | Trap | Decision rule |
|---|---|---|
| Attention: softmax(QK^T/sqrt(d_h) + M)V | Mask after softmax leaks | Add the mask before softmax |
| KV cache: 2 T d bytes per layer | Provisioned for max T always | Account per actual sequence length |
| SFT loss mask on answers only | Prompt tokens in the loss | Mask them out |
| Temperature/top-p at inference | Tuning them like training knobs | Fix before you evaluate |

## U15, reasoning and RL

| Formula / shape | Trap | Decision rule |
|---|---|---|
| V^pi(s) = R + gamma E[V^pi(s')] | gamma >= 1 diverges | Keep gamma in [0,1) |
| Value iteration: max Bellman backup | Async order stalls | Check the residual |
| RLVR reward from a verifier | Verifier games the metric | Audit the verifier independently |
| GRPO advantage: (R - mean)/std | Zero group variance => no signal | Monitor group diversity |

## U16, control and policy optimization

| Formula / shape | Trap | Decision rule |
|---|---|---|
| LQR gain: -(B^T Phi B - W)^{-1} B^T Phi A | The notes print it unsigned | Check Phi negative semidefinite |
| Riccati backward from Phi_T = -U_T | W not positive definite | Check eigenvalues first |
| Kalman gain K = P C^T(CPC^T + R)^{-1} | Wrong noise scale | Calibrate from data |
| PPO clip: min(r A, clip(r) A) | Many epochs on stale data | Watch KL and clipped fraction |
| Baseline keeps PG unbiased | Action-dependent baseline biases | Baseline on state only |

## U17, identities and craft

| Formula / shape | Trap | Decision rule |
|---|---|---|
| KL same-cov Gaussian: (1/2) d_Maha^2 | Different covariances | Use the full formula |
| KL chain rule: split joint KL | Uninformative factorization | Pick the factorization that localizes |
| se = s/sqrt(n) | n = 3 treated as Gaussian | Use the t-multiplier |
| One-factor ablation | Five changes at once | Change one thing per run |
| Acceptance gate: baseline, canary, rollback, owner | Shipping the training script | No gate, no deploy |
