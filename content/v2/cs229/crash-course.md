---
course_slug: cs229
course_name: Machine Learning (Stanford CS229)
course_order: 2
instructor: Tengyu Ma and Andrew Ng
date: '2026-10-06'
title: 'Crash course: the whole course in thirty minutes'
nav: CS229 · Crash course
page_id: cs229-crash
order: 180
summary: 'Seventeen units in about thirty minutes: one worked number, exam tables, memory aids, and self-tests per unit.'
---
# Crash course: the whole course in thirty minutes

Seventeen units, under two minutes each. Every worked number below
comes from the named lesson section. The crash course points, the
lessons prove.

## U01, Learning problems, risk, setup (lessons/u01)

A learning problem is data, a hypothesis family, a loss, and a
risk. Empirical risk is the average loss on the training set.
Population risk is the average over the true distribution. You
optimize the first and care about the second. The gap between them
is generalization.

Worked number (U01): m = 2 patients, n = 3 features. X has shape
(2, 3): 2 examples, 3 dimensions. theta = [1, 1] gives h(x) = 1 +
x. On x = 1, h = 2, y = 2, error 0.

> [!MEMORY] Risk is the goal. The training set is the evidence. Never confuse them.

| Object | Notation | Meaning |
|---|---|---|
| Features | x in R^n | The inputs |
| Target | y | The label |
| Hypothesis | h_theta | The predictor |
| Empirical risk | (1/m) sum loss | Training average |
| Population risk | E[loss] | True average |

Self-test:

> [!QA]
> Q: Why is low training error not enough?
> A: Training error is empirical risk. Deployment pays population risk. The gap is generalization, and U08 measures it.

## U02, Linear regression (lessons/u02)

Fit h(x) = theta^T x by minimizing squared error. Three ways:
stochastic gradient (one example at a time), batch gradient (all
examples), or the normal equations (one linear solve). Gaussian
noise makes least squares the MLE. Locally weighted regression
fits a new linear model at each query point: nonparametric, no
fixed theta.

Worked number (U02): theta = [0, 0], alpha = 0.1, first stochastic
step on (x = 1, y = 2): h = 0, error 2, theta := [0.2, 0.2].
Normal equations on X = [[1,1],[1,2],[1,3]], y = [2,3,4]:
X^T X = [[3,6],[6,14]], X^T y = [9,20], solve gives theta = [1,1].

> [!MEMORY] Gradient steps are cheap and many. The normal equations are one shot and cubic.

Self-test:

> [!QA]
> Q: When do you pick the normal equations over gradient descent?
> A: Small d, well-conditioned X^T X, and you want the exact answer. For n = 10^7 and d = 100, gradient descent wins on memory.

## U03, Classification and GLMs (lessons/u03)

Logistic regression models p(y = 1 | x) = sigmoid(theta^T x).
The gradient is (y - h) x: the same error-times-feature shape as
LMS. GLMs generalize this: pick an exponential-family noise
model, its canonical link gives the predictor. Softmax is the
multiclass link.

Worked number (U03): theta = [-4, 1], x = [1, 3], y = 1.
h = 0.2689, error 0.7311, alpha = 0.5: theta := [-3.6344,
2.0967]. New h = 0.9344. One step moves the probability from
0.27 to 0.93.

> [!MEMORY] Sigmoid in, cross-entropy out. The gradient is always error times feature.

Self-test:

> [!QA]
> Q: The data is linearly separable and the weights blow up. What happened?
> A: The MLE is at infinity on separable data. Regularize or stop early.

## U04, Generative classifiers (lessons/u04)

Model p(x | y) and p(y), then use Bayes rule. GDA fits a
Gaussian per class with shared covariance: the boundary is
linear. Naive Bayes assumes features independent given the
class. Discriminative models (logistic) usually need less data
when the Gaussian assumption is wrong, generative models win
when it is right.

Worked number (U04): Sigma = [[1, 0.8],[0.8, 1]], determinant
0.36. At x = [1,1], the Mahalanobis distance uses Sigma^{-1} =
[[2.778, -2.222],[-2.222, 2.778]].

> [!MEMORY] Generative models the inputs. Discriminative models the boundary.

Self-test:

> [!QA]
> Q: GDA and logistic regression both give linear boundaries. When does GDA win?
> A: When the Gaussian assumption holds and data is scarce: GDA uses the input distribution, logistic does not.

## U05, Kernels (lessons/u05)

A kernel K(x, z) = <phi(x), phi(z)> computes an inner product in
a rich feature space without building phi. The representer
theorem: the optimal solution lives in the span of the training
points. Cost: the n x n Gram matrix.

Worked number (U05): x = [1,2], z = [3,1], <x,z> = 5. Degree-2
polynomial kernel: (1 + 5)^2 = 36. Same as the explicit inner
product in the degree-2 feature space.

> [!MEMORY] The kernel is a shortcut. The Gram matrix is the bill.

Self-test:

> [!QA]
> Q: Why can a kernel beat explicit features?
> A: Infinite-dimensional phi (like RBF) has no explicit finite form. The kernel computes the inner product anyway.

## U06, SVMs (lessons/u06)

Maximize the geometric margin: the distance from the boundary to
the nearest point. The dual depends only on inner products, so
kernels plug in. Only support vectors matter. C trades margin
width against violations. SMO solves the dual two variables at a
time.

Worked number (U06): w = [3,4], b = -25, x = (4,3), y = 1.
Score -1, functional margin -1, ||w|| = 5, geometric margin
-0.2: the point is 0.2 units on the wrong side. Doubling
(w, b) doubles the functional margin but the geometric margin
stays -0.2.

> [!MEMORY] Functional margin scales. Geometric margin does not. Optimize the geometric one.

Self-test:

> [!QA]
> Q: All alphas hit the C bound. What does that tell you?
> A: C is too small or the data is very noisy: every point wants to be a margin violator.

## U07, Neural nets and backprop (lessons/u07)

A network is a composition of modules. Each module's backward
contract: given dL/dy, return dL/dx and dL/dW. The chain rule
does the rest. Activations must be nonlinear or the depth
collapses to one linear map. Check every implementation with
finite differences.

Worked number (U07): z = [1,2,3], LayerNorm gives [-1.2247, 0,
1.2247]. Scale by 5 first: identical output. The normalization
is scale-invariant.

> [!MEMORY] Forward is composition. Backward is the chain rule. Test with finite differences.

Self-test:

> [!QA]
> Q: Gradients are exactly zero below layer 3. Name two causes.
> A: Dead ReLUs everywhere below, or a detached graph (no_grad or wrong wiring).

## U08, Generalization (lessons/u08)

Test error = bias^2 + variance + noise. Simple models: high
bias, low variance. Complex models: low bias, high variance.
Finite hypothesis classes: union bound gives sample complexity
log|H|. Double descent: past the interpolation threshold, test
error falls again.

Worked number (U08): truth x^2 on [0,1], noise 0.3, n = 15.
Linear: bias^2 0.05, variance 0.002. Degree-5: bias^2 0,
variance 0.08. The quadratic hits bias^2 0, variance 0.008:
the sweet spot.

> [!MEMORY] Bias is what the family cannot fit. Variance is what the data cannot pin down.

Self-test:

> [!QA]
> Q: Test error rises while train error falls. Two causes, two fixes?
> A: Overfitting: regularize or stop early. Distribution shift: fix the data pipeline, not the training loop.

## U09, Regularization and model selection (lessons/u09)

Ridge (l2) shrinks weights, LASSO (l1) zeroes some. Early
stopping is implicit regularization. Bayesian view: MAP with a
prior. Choose hyperparameters by cross-validation, never on the
test set. Leakage is the silent killer: any test information in
training invalidates the estimate.

Worked number (U09): degree-8 polynomial, n = 12. Test MSE:
unregularized 2.41, ridge lambda = 1 gives 0.31, lambda = 100
gives 0.62. The sweet spot is between.

> [!MEMORY] Regularization is a prior. Cross-validation is the judge. Leakage is the crime.

Self-test:

> [!QA]
> Q: You tuned on the test set and report 0.95 accuracy. What is wrong?
> A: The test set is now a validation set. The reported number is optimistic. Get a fresh test set.

## U10, Clustering and EM (lessons/u10)

k-means alternates: assign points to the nearest center,
recompute centers. It converges to a local minimum, not the
global one. EM generalizes this to mixtures: the E step
computes soft responsibilities, the M step maximizes the
expected complete likelihood. Jensen's inequality builds the
lower bound. Likelihood never decreases.

Worked number (U10): two Gaussians, n = 200, k = 2. k-means
objective falls 7.96 to 0.98 in 4 iterations, monotone. With
k = 3 the objective is lower but the third cluster splits a
real one: lower objective is not truer clusters.

> [!MEMORY] k-means is hard EM with spherical Gaussians. EM never decreases the likelihood.

Self-test:

> [!QA]
> Q: EM converged but the clusters are meaningless. What do you check?
> A: Initialization (multiple restarts), k (model selection), and whether the likelihood gain came from a variance collapsing to zero.

## U11, VI, PCA, ICA (lessons/u11)

PCA keeps the directions of maximum variance: eigendecomposition
of the covariance, or SVD of the centered data. ICA separates
independent non-Gaussian sources: independence is stronger than
decorrelation. Variational inference maximizes the ELBO when the
posterior is intractable. The VAE is VI with neural nets.

Worked number (U11): 1000 points near a plane in 3d.
Eigenvalues 2.26, 1.62, 0.04. Top-2 PCA keeps 98.9% of the
variance.

> [!MEMORY] PCA decorrelates. ICA independizes. VI bounds what it cannot compute.

Self-test:

> [!QA]
> Q: PCA gives uncorrelated components. Why is that not enough for source separation?
> A: Uncorrelated is second-order. Independence needs higher orders. Gaussian sources cannot be separated at all.

## U12, Diffusion (lessons/u12)

Corrupt data with a fixed Gaussian schedule, then learn to
reverse it. The ELBO becomes a denoising objective: predict the
noise (or the clean x0, or the score: three parameterizations,
one model). Sampling runs the reverse chain. The schedule is
load-bearing: alpha-bar must reach 0 exactly at T.

Worked number (U12): T = 100, beta linear 1e-4 to 0.02, x_0 =
3.0. alpha-bar_50 = 0.7772, so x_50 has mean 2.6447 and
variance 0.2228.

> [!MEMORY] Forward is fixed corruption. Reverse is learned denoising. The schedule is the contract.

Self-test:

> [!QA]
> Q: Samples are blurry gray blobs. Two possible causes?
> A: The reverse net predicts the mean (underfit), or the schedule destroys signal too early.

## U13, Foundation models (lessons/u13)

Pretrain once on broad data, adapt cheaply: linear probing
(frozen backbone), full fine-tuning, or LoRA (low-rank
adapters). Contrastive learning pulls positives together and
pushes negatives apart: the InfoNCE loss. RAG grounds
generation in retrieved documents.

Worked number (U13): d = 4096, r = 8. Dense update: 16.8M
parameters. LoRA: 65,536 parameters, a 256x reduction.

> [!MEMORY] Pretraining buys the representation. Adaptation rents a small corner of it.

Self-test:

> [!QA]
> Q: Linear probing fails but full fine-tuning works. What does that tell you?
> A: The pretrained features do not linearly separate your task. The representation needs reshaping, not just a new head.

## U14, LLMs (lessons/u14)

Tokens in, next-token loss, transformer blocks. The causal mask
enforces autoregression: position t sees only the past. KV
cache makes generation linear per step instead of quadratic.
GQA and MQA shrink the cache. SFT trains on answer tokens only
(the loss mask). Temperature and top-p are inference choices,
not training choices.

Worked number (U14): d = 4096, 32 heads, fp16, T = 8192. KV
cache per layer: MHA 128 MiB, GQA (8 groups) 32 MiB, MQA 4
MiB. The cache scales with the number of KV heads.

> [!MEMORY] The mask is the autoregression. The cache is the memory bill. The loss mask is the SFT detail.

Self-test:

> [!QA]
> Q: Training loss is near zero after one epoch but generation is incoherent. Diagnose.
> A: The causal mask has a gap or sits in the wrong place: the model reads the future during training.

## U15, Reasoning and RL foundations (lessons/u15)

Chain of thought turns one hard prediction into many small
ones. RLVR trains traces with verifier rewards: the reward is
terminal, the KL penalty guards drift, GRPO replaces the
critic with group-relative advantages. Underneath: MDPs,
Bellman equations, value and policy iteration, model learning
from counts, and continuous-state methods (discretization vs
value approximation).

Worked number (U15): two-state toy, gamma = 0.9. V(B) = 10,
V(A) = 9. Value iteration error ratio exactly 0.9 per sweep:
the Bellman operator is a gamma-contraction. GRPO on rewards
[1,0,0,1]: advantages [1,-1,-1,1].

> [!MEMORY] The verifier is the objective. The Bellman equation is the recursion. The error shrinks by gamma.

Self-test:

> [!QA]
> Q: RLVR reward rises but task accuracy is flat. Diagnose.
> A: The verifier is invalid: it rewards a proxy (format, length), not the task. Audit the verifier, not the model.

## U16, Control and policy optimization (lessons/u16)

Finite-horizon MDPs solve backward with non-stationary
policies. LQR: linear dynamics plus quadratic costs give a
closed-form Riccati solution with a linear policy that
ignores the noise. DDP iterates linearizations along a
trajectory. LQG adds the Kalman filter for partial
observation. REINFORCE: the log-derivative trick, baselines
cut variance, PPO clips the likelihood ratio to stay near the
old policy.

Worked number (U16): 1d LQR, A = B = U = W = 1, T = 2.
Phi = [-1, -1.5, -1.6], gains L = [-0.5, -0.6]. From s_0 =
5: 5 -> 2 -> 1. Kalman 1d step: K = 2/3, estimate 1.0,
variance 2/3. PPO clip, eps = 0.2: (1, 1.3) -> 1.2 capped,
(-1, 0.5) -> -0.8 capped.

> [!MEMORY] Riccati goes backward. The Kalman gain weights the innovation. The clip bounds the step.

Self-test:

> [!QA]
> Q: The LQR closed loop diverges although the recursion ran. Two causes?
> A: The gain sign is flipped (the notes print it unsigned), or W is not positive definite.

## U17, Appendices and synthesis (lessons/u17)

The four identities: Gaussian sums stay Gaussian, Gaussian
conditioning, same-covariance KL, the KL chain rule. The
historical supplements (factor analysis, online learning,
HMMs, GPs) live on inside EM, Kalman, and kernels. Research
craft: baselines with tests, one-factor ablations with
matched budgets, honest error bars, and the deployment gap
map.

Worked number (U17): ten seeds, returns around 8.1. Mean
8.10, sample std 0.18, standard error 0.058. Bootstrap 95%
interval [7.99, 8.21]. KL(N([1,0],I) || N([0,0],I)) = 0.5,
Monte Carlo 0.4998.

> [!MEMORY] Report all seeds. One factor per ablation. A difference inside the error bars is not a win.

Self-test:

> [!QA]
> Q: Your method beats the baseline 0.83 to 0.79. Is it better?
> A: Not established: no spread, no sample size, no budget statement. Overlapping error bars mean noise.

## The whole course in one table

| Unit | Core mechanism | Key number |
|---|---|---|
| U01 | Empirical vs population risk | X shape (2,3) |
| U02 | Least squares, 3 solvers | theta [0.2,0.2] after 1 SGD step |
| U03 | Sigmoid, GLMs | h 0.27 -> 0.93 in 1 step |
| U04 | GDA, naive Bayes | det Sigma 0.36 |
| U05 | Kernel trick | (1+5)^2 = 36 |
| U06 | Max margin, dual | geometric margin -0.2 |
| U07 | Backprop chain rule | LayerNorm scale-invariant |
| U08 | Bias/variance | quadratic wins: var 0.008 |
| U09 | Ridge, CV | MSE 2.41 -> 0.31 |
| U10 | k-means, EM | J 7.96 -> 0.98 |
| U11 | PCA/ICA/VI | 98.9% variance in top-2 |
| U12 | Denoising objective | alpha-bar_50 0.7772 |
| U13 | Pretrain + adapt | LoRA 256x fewer params |
| U14 | Causal attention, KV cache | MQA 4 MiB vs MHA 128 MiB |
| U15 | Bellman, RLVR | error ratio 0.9/sweep |
| U16 | Riccati, PPO | 5 -> 2 -> 1 trajectory |
| U17 | KL identities, craft | se 0.058 over 10 seeds |

## Final self-test

> [!QA]
> Q: Name the assumption each of these breaks: naive Bayes on correlated features, LQR on a swing-up, PPO with 50 epochs per batch.
> A: Conditional independence, linearity near one fixed point, the local surrogate (the state distribution shifted).
