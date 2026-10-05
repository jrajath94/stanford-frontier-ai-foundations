---
page_id: cs229-crash
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 901
nav: "CS229 · Crash course"
title: "CS229 Crash Course"
summary: "Interview-speed review of CS229: the full story in 30 minutes, with images and links into the deep lessons."
---

<span class="crash-timer">30 minutes · interview speed</span>

This page tells the whole story fast. Each section gives you the working
version: enough to answer interview questions with confidence. Links at
the end of each section take you into the full lesson when you want the
derivations, the figures, and the follow-ups.

<div class="crash-section" markdown="1">

### 1. Machine learning: T, E, P

A program learns from experience E on tasks T, measured by P, if P
improves with E. That is Mitchell's 1998 definition, and it is the
sharpest one sentence in the field. Samuel's 1959 version is shorter:
learning without being explicitly programmed.

Three paradigms. Supervised: the data has answers, (x, y) pairs, predict
y. Unsupervised: no answers, find structure. Reinforcement: act in
sequences, get rewards.

<figure class="crash-fig"><img src="assets/svg/l01-paradigms.svg" alt="Three learning paradigms"><figcaption>Supervised predicts from labels. Unsupervised finds structure. RL acts for reward.</figcaption></figure>

<ul class="crash-links">
<li><a href="l01-introduction.html">Lecture 1: definitions and the course map</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 2. Linear regression and the loss chip

Hypothesis: h(x) = theta_0 + theta_1 x. Two parameters, fixed no matter
how much data arrives. That is a parametric model. The loss chip scores
wrongness: J(theta) = (1/2) sum (h - y)^2. The 1/2 is convention. The
square comes from solvability, history, and Gaussian noise.

Gradient descent walks downhill: theta := theta - alpha * grad J. The
gradient's shape is error times feature, the atomic update of the whole
course. SGD uses noisy minibatch gradients: fast, unbiased on average,
shuffle every epoch. The normal equations solve it in one shot:
theta = (X^T X)^-1 X^T y, when invertible.

<figure class="crash-fig"><img src="assets/svg/l02-loss-chip.svg" alt="The loss chip"><figcaption>J(theta): prediction minus truth, squared, averaged. Training minimizes it.</figcaption></figure>

<ul class="crash-links">
<li><a href="l02-linear-regression.html">Lecture 2: loss, GD, SGD, normal equations</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 3. MLE is the bedrock, logistic regression classifies

Maximum likelihood: write p(data; theta), take the log, maximize. IID
means product; log means sum. Gaussian noise plus MLE gives least
squares: the lecture 2 loss, derived, not guessed.

Classification keeps theta^T x but squashes it through the sigmoid:
h = 1/(1+e^(-theta^T x)) = P(y=1|x). Fit by gradient ascent or Newton's
method. Newton needs no step size and converges quadratically, at cubic
cost per step. On logistic loss, Newton is iteratively reweighted least
squares.

<figure class="crash-fig"><img src="assets/svg/l03-sigmoid.svg" alt="The sigmoid"><figcaption>S-shaped, 0 to 1. Height is confidence. Predict 1 past zero.</figcaption></figure>

<ul class="crash-links">
<li><a href="l03-logistic-regression.html">Lecture 3: MLE, sigmoid, Newton</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 4. GLMs unify everything, softmax picks classes

The exponential family: p(y; eta) = b(y) exp(eta^T T(y) - a(eta)).
Gaussian, Bernoulli, multinomial all fit. The GLM recipe: pick a
distribution, set eta = theta^T x, fit by MLE. Least squares, logistic,
and softmax regression are all GLMs.

Softmax: e^z_k / sum e^z_j. Positive, sums to one, max-entropy honest,
numerically stable via max subtraction. Cross-entropy, -log p of the
true class, is the classification MLE loss. Label smoothing softens
targets to curb overconfidence.

<figure class="crash-fig"><img src="assets/svg/l04-softmax.svg" alt="Softmax"><figcaption>Exponentiate, normalize. Every class gets a valid probability.</figcaption></figure>

<ul class="crash-links">
<li><a href="l04-glms-softmax.html">Lecture 4: exponential family, GLMs, softmax</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 5. Generative models: GDA and Naive Bayes

Discriminative models learn p(y|x). Generative models learn p(x|y) and
flip with Bayes' rule. GDA models each class as a Gaussian with shared
covariance: fit by class averages, closed form, linear boundary.
Separate covariances give quadratic boundaries.

The punchline: logistic regression reaches the same sigmoid with weaker
assumptions, so it usually wins on accuracy. Naive Bayes multiplies
per-word likelihoods for spam: false independence, true usefulness.
Laplace smoothing adds one to kill zero counts.

<figure class="crash-fig"><img src="assets/svg/l05-gda.svg" alt="GDA"><figcaption>Each class a Gaussian. Shared covariance: linear boundary.</figcaption></figure>

<ul class="crash-links">
<li><a href="l05-gda-naive-bayes.html">Lecture 5: GDA, Naive Bayes, spam</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 6. Bias, variance, and picking models

Error is bias (wrong assumptions) plus variance (sample sensitivity).
Regularization trades them; ridge adds lambda||theta||^2. Double
descent broke the U curve: past the interpolation peak, test error
falls again as the optimizer seeks minimum-norm fits.

Train fits, dev selects, test reports once. K-fold rotates validation
for small data. ImageNet-V2 showed shared test sets rot through
adaptive overfitting. Hyperband tunes hyperparameters by successive
halving: many cheap configs, double the winners' budgets.

<figure class="crash-fig"><img src="assets/svg/l06-dd.svg" alt="Double descent"><figcaption>Test error falls, peaks at interpolation, falls again.</figcaption></figure>

<ul class="crash-links">
<li><a href="l06-bias-variance.html">Lecture 6: bias-variance, double descent, Hyperband</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 7. Neural networks: bend every layer

A neuron: a = sigma(w.x + b). Linear mix, then bend. Without the bend,
stacks stay linear. ReLU, max(0,z), is the default: cheap, derivative 1
when active, no vanishing. An MLP stacks fully connected layers.
Universal approximation promises capacity, not trainability. Residual
blocks learn x + F(x): the skip carries gradients past depth.

<figure class="crash-fig"><img src="assets/svg/l07-residual.svg" alt="Residual block"><figcaption>Learn the change, add the input back. Deep stacks train.</figcaption></figure>

<ul class="crash-links">
<li><a href="l07-neural-networks-1.html">Lecture 7: neurons, ReLU, MLPs, residuals</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 8. Backprop: one walk, all gradients

The theorem: forward pass O(N), full gradient O(N), N the parameter
count. Each module has a backward function: dJ/dx = J_M^T dJ/du. Seed
dJ/dJ = 1, walk back, compose. For one example and one weight matrix,
the gradient is rank 1: error outer input. Apply the construction twice
for Hessian-vector products and second-order methods.

<figure class="crash-fig"><img src="assets/svg/l08-backprop.svg" alt="Backprop modules"><figcaption>Forward computes. Backward pushes gradients through each module.</figcaption></figure>

<ul class="crash-links">
<li><a href="l08-backpropagation.html">Lecture 8: the O(N) theorem, modules, rank-1</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 9. K-means and the soft GMM

No labels. K-means: random centers, assign nearest, move to means,
repeat. Distortion never increases; it converges to a local minimum.
Labels are meaningless; only center positions matter. K-means++
seeds far apart with a provable approximation ratio. The elbow picks k.

GMMs soften everything: each cluster a Gaussian, each point partially
in every cluster via responsibilities. K-means is the hard limit of a
GMM.

<figure class="crash-fig"><img src="assets/svg/l09-gmm.svg" alt="Hard vs soft clustering"><figcaption>K-means: 100% one cluster. GMM: split responsibilities.</figcaption></figure>

<ul class="crash-links">
<li><a href="l09-kmeans-gmm.html">Lecture 9: k-means, k-means++, GMMs</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 10. EM and PCA

GMM likelihood has a log of a sum. Jensen's inequality builds the
ELBO, a tractable lower bound. E-step: set Q to the posterior, making
the bound tight. M-step: weighted MLE. Likelihood never decreases.
Diffusion models train on ELBOs of this shape.

PCA: center, rescale, take the top eigenvectors of the covariance.
Feet-versus-miles warns: rescale or units vote. Eigenvalue gaps measure
trust. Report variance kept.

<figure class="crash-fig"><img src="assets/svg/l10-elbo.svg" alt="EM and the ELBO"><figcaption>E tightens the bound. M climbs it. Likelihood rises every round.</figcaption></figure>

<ul class="crash-links">
<li><a href="l10-em-pca.html">Lecture 10: ELBO, EM steps, PCA</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 11. Diffusion: noise up, learn to denoise

The predominant image generator. Forward process, fixed: x_t =
(1-beta_t)x_{t-1} + sqrt(beta_t) eps, ~1000 steps to pure noise.
Reverse process, learned: a network denoises one step at a time.
Gradual beats one-shot because each step is an easy problem. Training
is the ELBO over the chain. Sampling walks backward from noise.

<figure class="crash-fig"><img src="assets/svg/l11-diffusion.svg" alt="Diffusion chain"><figcaption>Fixed noising forward. Learned denoising back.</figcaption></figure>

<ul class="crash-links">
<li><a href="l11-diffusion-models.html">Lecture 11: forward, reverse, ELBO training</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 12. Foundation models: pre-train, then adapt

Post-GPT-3 paradigm, named by Percy Liang's team. Pre-train one model
on massive messy unlabeled data. Adapt it to many tasks. Never train
per task from scratch. Representation learning produces embeddings:
vectors where distance means similarity. Linear probing freezes the
representation and fits w: prediction = w.phi(x), a lower bound on
what is represented. LoRA adapts as W_0 + AB, low rank, cheap.

<figure class="crash-fig"><img src="assets/svg/l12-fm.svg" alt="Foundation model paradigm"><figcaption>One expensive pre-training, unlimited cheap adaptations.</figcaption></figure>

<ul class="crash-links">
<li><a href="l12-foundation-models.html">Lecture 12: paradigm, probing, LoRA</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 13. Contrastive learning and RAG

No labels? Manufacture supervision. Two augmentations of one image are
a positive pair; everything else is negative. Pull positives, push
negatives: diagonal big, off-diagonal small. Hard negatives, the
confused pairs, carry the signal.

RAG answers with private data and no training: retrieve 5-10 documents,
add them to context, generate. Modular, permission-governed, trivially
forgettable, cheap. Fine-tuning is for behavior; RAG is for knowledge.

<figure class="crash-fig"><img src="assets/svg/l13-rag.svg" alt="RAG"><figcaption>Retrieve private docs, answer with context. Nothing is trained.</figcaption></figure>

<ul class="crash-links">
<li><a href="l13-contrastive-rag.html">Lecture 13: contrastive loss, RAG</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 14. Transformers

Autoregressive: p(x) = product of p(x_t | past). Generate one token at
a time; training is parallel via teacher forcing. Tokenize first, fix
the tokenizer forever. Attention: queries, keys, values; scores =
q.k, softmax, weighted values. Causal mask hides the future. Cost:
O(T^2 d), the bottleneck. Block: attention, RMSNorm, MLP, residuals.

<figure class="crash-fig"><img src="assets/svg/l14-mask.svg" alt="Causal mask"><figcaption>Position t sees only the past. No peeking.</figcaption></figure>

<ul class="crash-links">
<li><a href="l14-transformers.html">Lecture 14: attention, masking, cost</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 15. Efficiency, in-context learning, SFT

KV cache stores keys and values across decode steps: compute only the
new token's. MQA/GQA share KV heads to shrink the cache. MoE routes
each token to a few of many experts: sparse compute, dense capacity.

Zero-shot: task description only. Few-shot: examples in the prompt. No
parameter updates; the GPT-3 shock. SFT, instruction tuning: train on
(instruction, answer) pairs, loss on answer tokens only. It shapes
behavior, not knowledge.

<figure class="crash-fig"><img src="assets/svg/l15-sft.svg" alt="SFT"><figcaption>Instruction seen, answer predicted. Loss only on y.</figcaption></figure>

<ul class="crash-links">
<li><a href="l15-efficient-icl-sft.html">Lecture 15: KV cache, MoE, ICL, SFT</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 16. RL basics and policy gradient

Sequential decisions: actions change the future, greedy fails. MDP:
states, actions, transition dynamics, reward, discount. Policy
pi(a|s) is the behavior; maximize expected return. Values compress the
future: V^pi(s), V*, Bellman recursion.

REINFORCE: grad J = E[sum grad log pi(a_t|s_t) * R]. Sample
trajectories, weight by return, shift probability toward winners.
Stochastic policies only. On-policy: fresh samples every update.
Variance needs baselines.

<figure class="crash-fig"><img src="assets/svg/l16-pg.svg" alt="Policy gradient"><figcaption>Sample, score by return, reinforce the winners.</figcaption></figure>

<ul class="crash-links">
<li><a href="l16-reinforcement-learning.html">Lecture 16: MDP, values, REINFORCE</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 17. PPO and verifiable rewards

PPO reuses samples via the ratio r = pi_new/pi_old, clipped to
[1-eps, 1+eps]: proximal updates, bounded damage. Advantage = return
minus baseline: same expected gradient, less noise.

For LLMs: state is history, action is the next token, reward lands at
the end. Math answers are verifiable and binary: 1 if the parsed answer
matches, 0 otherwise. No human grader. Chain-of-thought reasoning
emerges from answer-level rewards. Stability remains the open problem.

<figure class="crash-fig"><img src="assets/svg/l17-verify.svg" alt="Verifiable reward"><figcaption>Parse the answer, compare to truth. Reward 1 or 0.</figcaption></figure>

<ul class="crash-links">
<li><a href="l17-rl-for-llms.html">Lecture 17: PPO, advantages, RLVR</a></li>
</ul>

</div>

<div class="crash-section" markdown="1">

### 18. Rapid-fire: say the answer before you read it

**What is ML?** Improve P on T with E. Samuel: learn without explicit programming.

**Why the 1/2 in least squares?** Convention. Cancels the derivative's 2.

**GD or Newton?** Newton: fast, cubic cost, no step size. GD: cheap steps. Millions of params: GD.

**What is MLE?** argmax log L. IID gives product, log gives sum. The bedrock.

**Why sigmoid?** Smooth squash to (0,1); height reads as P(y=1|x).

**Softmax in one line?** e^z_k / sum e^z_j; subtract max for stability.

**Cross-entropy?** -log p of the true class.

**GDA vs logistic?** Same sigmoid; logistic assumes less, usually wins.

**Laplace smoothing?** Add one to every count. Never multiply by zero.

**Bias vs variance?** Wrong assumptions vs sample sensitivity. Ridge trades them.

**Double descent?** Past interpolation, error falls again; optimizer seeks minimum norm.

**Why ReLU?** Derivative 1 when active. No vanishing.

**Residual?** x + F(x). Skip carries gradients.

**Backprop cost?** O(N) for loss and gradient, by shared backward walk.

**Rank-1?** One example, one matrix: gradient = error outer input.

**EM?** E tightens the ELBO, M climbs it. Likelihood never drops.

**PCA steps?** Center, rescale, top eigenvectors of covariance.

**Diffusion?** Fixed noising forward, learned denoising back, ELBO trains it.

**Foundation model?** Pre-train massive and unlabeled, adapt to tasks.

**Linear probing?** Freeze phi, fit w. Lower bound on represented knowledge.

**LoRA?** W_0 + AB, low rank, cheap adaptation.

**Contrastive?** Positives pulled, negatives pushed, no labels.

**RAG?** Retrieve docs into context. Knowledge without training.

**Attention cost?** O(T^2 d). The mask enforces causality.

**KV cache?** Store K,V across steps; compute only the new token's.

**Zero vs few shot?** Description only vs examples in prompt. No training either way.

**SFT loss?** On answer tokens only. Behavior, not knowledge.

**Policy gradient?** E[sum grad log pi * R]. Stochastic policies only.

**PPO?** Clip the importance ratio. Proximal, stable, sample-efficient.

**Verifiable reward?** Binary answer match, parsed from tags. No human.

</div>
