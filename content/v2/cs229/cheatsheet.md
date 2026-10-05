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

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### Definitions

**ML (Mitchell 1998):** program learns from experience E on tasks T,
measured by P, if P improves with E.

**Supervised:** (x, y) pairs; predict y. **Unsupervised:** x only; find
structure. **RL:** act, get reward, sequential.

**Hypothesis** h: X to Y. **Parameters** theta: learned. **Features** x:
given. **Loss** J(theta): scalar wrongness; training minimizes it.

**Generative:** models p(x|y). **Discriminative:** models p(y|x) directly.

</div>

<div class="cheat-block" markdown="1">

### Core formulas

**Least squares:** J(theta) = (1/2) sum (h - y)^2. The 1/2 is convention.

**Gradient descent:** theta := theta - alpha * grad J. Alpha is the step
size. Error-times-input is the atomic update.

**Normal equations:** theta = (X^T X)^-1 X^T y. Needs n > d. Singular
gives a null space: infinite thetas.

**MLE:** argmax log L(theta). IID gives product; log gives sum. Gaussian
noise + MLE = least squares.

**Sigmoid:** g(z) = 1/(1+e^-z). Derivative g(1-g), max 0.25.

**Softmax:** e^z_k / sum e^z_j. Subtract max logit for stability.

**Cross-entropy:** -log p of the true class.

**Neuron:** a = sigma(w.x + b). **MLP:** u^l = sigma(W^l u^(l-1)+b^l).
**Residual:** x + F(x).

**Backprop:** dJ/dx = J_M^T dJ/du per module. Forward O(N), gradient O(N).

**ELBO:** lower bound via Jensen; E-step tightens (Q = posterior),
M-step climbs. Likelihood never decreases.

**PCA:** eigenvectors of covariance, top k. Center and rescale first.

**Diffusion forward:** x_t = (1-beta_t)x_{t-1} + sqrt(beta_t) eps_t.
Reverse: learned denoiser. Train on ELBO.

**Attention:** scores = QK^T, softmax, times V. Cost O(T^2 d). Mask the
future.

**PPO:** min(rA, clip(r)A), r = pi_new/pi_old. Advantage A = R - b(s).

</div>

<div class="cheat-block" markdown="1">

### Numbers to memorize

- Sigmoid derivative peaks at 0.25: the vanishing seed
- ReLU derivative: 1 active, 0 dead
- Diffusion steps: T ~ 1000 in the original paper
- Backprop costs ~2x a forward pass; training step ~3x
- MoE example: 8 of 128 experts active per token
- RAG: 5-10 retrieved documents is typical
- Ridge: J + lambda||theta||^2; lambda is the dial
- K-means++: seed proportional to squared distance

</div>

<div class="cheat-block" markdown="1">

### Decisions

**Loss for regression?** Squared: solvable, Gaussian MLE.

**Loss for classification?** Cross-entropy: Bernoulli/multinomial MLE.
Squared sometimes works anyway; prefer the principled one.

**Newton or GD?** Newton: quadratic convergence, cubic cost per step,
no step size. GD/SGD: cheap steps, needs alpha. Small params: Newton.
Millions: GD.

**Generative or discriminative?** Discriminative usually wins on
accuracy with data. Generative wins on small data and can sample.

**GDA or logistic?** Same sigmoid, weaker assumptions for logistic.
GDA needs less data if Gaussian story is true.

**K-means or GMM?** K-means: fast, spherical, hard. GMM: shaped, soft,
likelihood. Overlap favors GMM.

**RAG or fine-tuning?** RAG for knowledge: editable, permissioned,
cheap. Fine-tuning for behavior. SFT shapes conduct, not facts.

**Few-shot or SFT?** Few-shot: instant, variable. SFT: baked in,
reliable. Prompt for flexibility, train for reliability.

**Bigger model?** Double descent says do not fear overparameterization,
but measure test error. Size and training time beat schedule tuning.

</div>

<div class="cheat-block" markdown="1">

### Common mistakes

- Thinking the 1/2 in J(theta) matters. Convention only.
- Calling logistic regression "regression." It classifies; the name is history.
- Confusing bias (lecture 6, wrong assumptions) with bias b (neuron, offset term).
- Touching the test set during tuning. It becomes a dev set.
- Forgetting to rescale before PCA or k-means. Units vote otherwise.
- Trusting the elbow. Heuristic, not theorem.
- Expecting SFT to teach facts. It teaches behavior.
- Thinking probes give upper bounds. Lower bounds only.
- Assuming bigger always wins. Measure; do not assume.
- Treating PPO theory as settled. The lecturer calls it mysterious.

</div>

<div class="cheat-block" markdown="1">

### Interview one-liners

- "MLE is the bedrock: every loss is a probabilistic story."
- "The 1/2 is convention; the square is Gaussian noise."
- "Logistic regression is the last layer of every model you use."
- "Newton is IRLS on logistic loss: weighted least squares per step."
- "Softmax: positive, max-entropy, stable via max subtraction."
- "GDA assumes; logistic regression just fits. Weaker wins with data."
- "Laplace smoothing: never multiply by zero."
- "Double descent: past interpolation, the optimizer seeks minimum norm."
- "ReLU's derivative is 1 when active; that is why deep nets train."
- "Backprop is O(N) because modules share one backward walk."
- "One example, one matrix: the gradient is rank 1."
- "EM: bound the hard thing with the ELBO, climb the bound."
- "Diffusion: a thousand easy denoising steps beat one impossible jump."
- "In-context learning was never trained for; it emerged."
- "SFT loss goes on the answer tokens only."
- "Policy gradient needs stochastic policies; determinism has no gradient."
- "Verifiable rewards: the task grades itself. Binary, parsed, no human."

</div>

</div>
