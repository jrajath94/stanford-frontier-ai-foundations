---
page_id: cs229-cheatsheet
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 900
nav: "CS229 · Cheatsheet"
title: "CS229 Cheatsheet"
summary: "Every key fact from CS229 on one dense page: definitions, formulas, numbers, decisions, mistakes, interview lines."
---

## The setup

- Supervised: learn from labeled pairs (x, y). Unsupervised: find
  structure with no labels. Reinforcement: act, get rewards.
- Mitchell: learn from E on T under P when P at T improves with E.
- Hypothesis h_theta(x): the guess function. Parameters theta: the
  knobs. Regression: predict a number. Classification: predict a
  category.

## The loss chip (owned by this course)

- J(theta) = 1/(2m) sum (h - y)^2. Smaller is better. The 1/2
  cancels the derivative's 2.
- MLE: pick knobs maximizing P(data | knobs). Gaussian noise gives
  least squares. Every loss is a noise assumption.
- Cross-entropy: -log(p_true class). Two classes: logistic loss.
- The gradient shape everywhere: error times feature, summed.

## Optimization

- Gradient descent: theta_j := theta_j - alpha * dJ/dtheta_j. All
  knobs simultaneously from old values.
- Alpha too small: crawls (1,000 steps for a 1-D bowl). Too large:
  diverges (4, -8, 16, -48 on theta^2). Bouncing loss means turn it
  down.
- SGD: one example per step. Noisy, m times cheaper, wins on huge
  data. Shuffle every epoch. Mini-batch (32-256) is the middle.
- Normal equations: theta = (X^T X)^-1 X^T y. Exact. O(n^3).
  Dead if singular (redundant features, n < d) or neural.
- Newton: theta := theta - J'/J''. No alpha. O(n d^2 + d^3) per
  step. Stats at d=20 loves it. Dead at d=1B. Each step on
  logistic regression is weighted least squares (IRLS).

## The model zoo

- Linear regression: h = theta^T x. Logistic: h = sigmoid(theta^T
  x), g(z) = 1/(1+e^-z): 0.12, 0.5, 0.88 at -2, 0, 2. Name lies:
  it classifies.
- GLM recipe: pick exponential-family distribution, eta =
  theta^T x, predict a'(eta). Gaussian -> least squares.
  Bernoulli -> logistic.
- Softmax: e^z_j / sum e^z_c. Four whys: GLM-dictated, smooth,
  maximum-entropy, numerically convenient. O(k) per prediction.
- GDA: each class a Gaussian, shared Sigma. MLE = class averages.
  Boundary linear: w = Sigma^-1(mu_1 - mu_0).
- Naive Bayes: words independent given class. Fit by counting.
  Laplace: (count+1)/(total+V). Unseen words cannot veto.
- Neuron: sigma(w^T x + b). ReLU = max(0,z). MLP: stacked bends.
  No bend: deep collapses to linear.
- Residual: out = x + F(x). Skip carries gradient at strength 1.
  2015, 100+ layers.
- Attention: softmax(QK^T/sqrt(d))V. Toy: [1,1,2] -> [0.21, 0.21,
  0.58] -> [0.79, 0.79]. Causal mask: no peeking. Cost: N^2
  (16.7M at N=4,096).

## Training dynamics

- Bias: wrong assumptions. Variance: noise sensitivity. Error =
  bias^2 + variance + noise.
- Double descent: past the interpolation peak, bigger generalizes
  again. The U is the left half.
- Train fits, dev compares, test reports once. k-fold CV when data
  is scarce. Never decide on test.
- Ridge: loss + rho||theta||^2. theta = (X^T X + rho I)^-1 X^T y.
  Always invertible. Pick rho on dev.
- Hyperband: successive halving. 81 configs start, best 3 finish.
- Backprop: forward O(N) implies gradient O(N). Modules ship
  forward + backward. dL/dW is rank 1 per example. H*v in O(N):
  the matrix never exists.
- Vanishing: 0.25^50 = 10^-30 through sigmoid stacks. Residuals
  and short paths fix it.

## Unsupervised

- K-means: assign, mean-update, repeat. Converges to a local
  minimum. Seed decides.
- K-means++: seed proportional to squared distance. O(log k)
  approximation. Sklearn default.
- Elbow: distortion 125.5, 4.0, 2.7, 1.5 -> k=2. Heuristic.
- GMM: responsibilities gamma_j(x). K-means is the hard limit.
- EM: E-step sets Q to posterior (bound tight), M-step maximizes
  (weighted MLE). Likelihood rises every round. Local maxima,
  slow crawl, degenerate collapse.
- PCA: top eigenvectors of covariance. Eigenvalues = variance
  per axis. Toy: (5,0) -> one axis keeps 100 percent. Linear
  only, scale-sensitive, O(d^3).

## Modern

- Diffusion: forward x_t = sqrt(1-beta_t)x_{t-1} + sqrt(beta_t)eps
  (fixed). Reverse: learn the denoiser. ELBO -> noise MSE. Sample:
  T reversals from pure noise. Price: T evals per image.
- Foundation: pre-train broad, adapt per task. Linear probe:
  freeze + linear head (88 vs 58 percent). LoRA: W + AB, rank r:
  d=1000, r=10 -> 20K vs 1M dof.
- Contrastive: pull augmented views together, push others apart.
  Hard negatives (cat-soccer photo vs FIFA text) keep it
  teaching. Tau sharpens.
- RAG: retrieve top-k chunks, paste into context, generate
  grounded. Knowledge in the store, not the weights.
- KV cache: per-token O(t^2) -> O(t). Memory linear in T. Fills
  GPU -> batch ~10 -> memory-bound. GQA: 4x smaller cache. MoE:
  8x params, ~1x compute, router balances.
- ICL: frozen weights, examples in prompt. Brittle, free.
  Emerged with scale.
- SFT: (instruction, response) pairs teach the job description.
  Pre-training teaches the world. Narrowing tax.
- MDP: S, A, P, R, gamma. V(s) = E[R + gamma V(s')]. Greedy
  misses delayed rewards.
- REINFORCE: grad E[R] = E[R grad log pi]. High variance,
  on-policy, sample-hungry.
- Advantage: reward-to-go minus baseline. +0.5/-0.5. Same
  expectation, less variance.
- PPO: r = pi_new/pi_old, objective min(r*A, clip(r, 0.8,
  1.2)*A). Clip binds in the pull direction: A>0 caps r at 1.2.
  A<0 floors r at 0.8. A<0, r=1.5: unclipped full correction.
  Proximal: never jump far. RLVR: binary verifiable rewards
  train thinking.

## Interview one-liners

- "Least squares is MLE under Gaussian noise."
- "Newton is O(n d^2 + d^3) per step: great at d=20, dead at
  d=1B. SGD is the workhorse."
- "Softmax has four whys: GLM form, smoothness, max-entropy,
  numerical convenience."
- "Backprop: forward O(N) implies gradient O(N). That is why
  billion-parameter models train."
- "Double descent: the U-curve is the left half. Past the peak,
  bigger generalizes."
- "LoRA: W+AB, rank r. 256x fewer dof at d=4096, r=8."
- "PPO: clip the importance ratio. Stay proximal."
- "RAG: knowledge in the store, not the weights."

## Never-confuse pairs

- **MLE vs MAP.** MLE: maximize P(data|theta). MAP: maximize
  P(theta|data) = P(data|theta)P(theta): MLE plus a prior. Ridge
  is MAP with a Gaussian prior.
- **V vs Q.** V(s): value of the state under the policy. Q(s,a):
  value of doing a now, then the policy. Q chooses (argmax), V
  evaluates. V(s) = E_a[Q(s,a)].
- **Bias (estimator) vs bias (model).** L06's bias-variance is
  about model error. L16's baseline is unbiased as an estimator:
  different "bias" word.
- **On-policy vs off-policy.** REINFORCE: fresh rollouts only
  (on-policy). PPO: reuses via importance weights (near-policy).
  Q-learning: fully off-policy (not in this course).
- **ICL vs SFT vs RLVR.** ICL: frozen weights, examples in the
  prompt. SFT: weights change on (instruction, response) pairs.
  RLVR: weights change on verifiable reward.
- **Ridge vs lasso.** Ridge (L2): shrinks, keeps all features,
  always invertible. Lasso (L1): zeroes features, selects.
- **GDA vs naive Bayes.** GDA: Gaussian per class, features
  correlated via Sigma. Naive Bayes: independence given class,
  counts words. Both generative.
- **Attention vs KV cache.** Attention: the O(N^2) computation.
  KV cache: the O(N) memory that avoids recompute. GQA shrinks
  the cache, not the math.
- **Temperature (softmax) vs temperature (diffusion).** L04's
  tau sharpens class probabilities. Diffusion has its own noise
  schedule: unrelated dial, same name.
- **ELBO (EM) vs ELBO (diffusion).** Same Jensen trick, different
  job: EM's ELBO bounds the marginal likelihood for missing
  labels. Diffusion's ELBO becomes noise-prediction MSE.

## If this, then that

- Loss bouncing or exploding: turn alpha down. Crawling: turn it
  up or scale features.
- d > 1M parameters: SGD/mini-batch, never Newton, never normal
  equations.
- Singular X^T X (redundant features, n < d): ridge, or drop
  features.
- Tuning rho on training error: stop. Rho falls to 0 on train.
  Tune on dev.
- Test accuracy far below dev: distribution shift or test
  leakage into train. Check the pipeline, not the model.
- K-means gives different answers per run: seed lottery. Use
  k-means++, multiple restarts, keep the best distortion.
- EM likelihood flat for rounds: slow crawl near a ridge. Check
  for degenerate collapse (variance -> 0).
- Attention weights all equal (~1/N): forgot sqrt(d), or tau
  too high. Sharp collapse to one token: tau too low.
- PPO ratio histogram spreads (many r near 0, few huge): too
  many epochs per batch. Refresh rollouts.
- RLVR reward climbs but traces look broken: reward hacking.
  Audit traces, strengthen the verifier.
- ICL accuracy swings with example order: brittleness, not a
  bug. SFT the task if it is fixed.
- GPU OOM in serving: KV cache. GQA, smaller batch, shorter
  context, or quantization: in that order.

## Mnemonics

- **"Error times feature"**: every supervised gradient is
  (prediction - truth) * input. If your gradient lacks this
  shape, recheck.
- **"Greedy is blind, values price"**: RL in six words.
- **"Subtract the luck"**: baselines and advantages.
- **"Clip the pull direction"**: PPO's four cases.
- **"Prototype in prompt, ship in weights"**: ICL vs SFT.
- **"Knowledge in the store, not the weights"**: RAG.
- **"The 1/2 cancels the 2"**: why J has 1/(2m).
- **"Train fits, dev compares, test reports once."**

## Classic mistakes

- Updating theta_j sequentially instead of simultaneously.
- Tuning rho on training error (falls monotonically to 0).
- Deciding on the test set (adaptive overfitting).
- Forgetting sqrt(d) in attention (collapse in large models).
- No Laplace smoothing (one unseen word zeroes everything).
- Confusing "can represent" (universal approximation) with "can
  learn".
- Trusting ICL prompts without testing wording variants.
