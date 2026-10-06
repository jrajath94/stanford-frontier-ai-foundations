---
page_id: cs229-index
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 0
nav: "CS229 · Overview"
title: "CS229: Machine Learning"
summary: "Mathematical foundations of machine learning, from linear models to deep learning and RL."
instructor: "Tengyu Ma, Chris Ré"
offering: "Spring 2026"
---

Machine learning rebuilt from zero: loss, probability, optimization,
neural nets, generative models, modern LLM machinery, and reinforcement
learning. Seventeen lectures, Spring 2026 (Ma and Ré). No background
assumed: every term is defined at first use.

**Start:** [Lecture 1](l01-introduction.html) → the three paradigms.
**Fast review:** [Cheatsheet](cheatsheet.html) · [Crash course](crash-course.html)

<div class="recap-grid">
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Paradigms">
<div class="rc-body">
<strong><a href="l01-introduction.html">L1. Introduction</a></strong>
<p>Mitchell's definition, supervised vs unsupervised vs RL, the course
arc. Machine learning in one sentence.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l02-loss-chip.svg" alt="Loss chip">
<div class="rc-body">
<strong><a href="l02-linear-regression.html">L2. Linear Regression</a></strong>
<p>Hypothesis, the loss chip, gradient descent, SGD, normal equations.
The cross-course loss symbol debuts.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-sigmoid.svg" alt="Sigmoid">
<div class="rc-body">
<strong><a href="l03-logistic-regression.html">L3. MLE, Logistic Regression</a></strong>
<p>Maximum likelihood as bedrock. Sigmoid, Newton, IRLS. Every loss is a
probabilistic story.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l04-softmax.svg" alt="Softmax">
<div class="rc-body">
<strong><a href="l04-glms-softmax.html">L4. GLMs, Softmax</a></strong>
<p>Exponential family unifies regression and classification. Softmax,
cross-entropy, label smoothing.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l05-gda.svg" alt="GDA">
<div class="rc-body">
<strong><a href="l05-gda-naive-bayes.html">L5. GDA, Naive Bayes</a></strong>
<p>Generative vs discriminative. Gaussians, quadratic boundaries,
Bayes' rule flips, Laplace smoothing.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l06-dd.svg" alt="Double descent">
<div class="rc-body">
<strong><a href="l06-bias-variance.html">L6. Bias, Variance, Selection</a></strong>
<p>Bias-variance, double descent, train/dev/test, Hyperband. Test sets
rot. Use them once.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l07-residual.svg" alt="Residual">
<div class="rc-body">
<strong><a href="l07-neural-networks-1.html">L7. Neural Networks</a></strong>
<p>Neurons, ReLU, MLPs, universal approximation, residuals. Bend every
layer or depth is pointless.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l08-backprop.svg" alt="Backprop">
<div class="rc-body">
<strong><a href="l08-backpropagation.html">L8. Backpropagation</a></strong>
<p>O(N) theorem, modules, rank-1 gradients, Hessian-vector products.
One walk, all gradients.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l09-gmm.svg" alt="GMM">
<div class="rc-body">
<strong><a href="l09-kmeans-gmm.html">L9. K-means, GMM</a></strong>
<p>Clustering: hard assignments, k-means++, soft responsibilities,
EM for mixture weights.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l10-elbo.svg" alt="ELBO">
<div class="rc-body">
<strong><a href="l10-em-pca.html">L10. EM, PCA</a></strong>
<p>ELBO and the E/M steps. PCA: center, rescale, eigenfaces. Variance
kept is the headline.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l11-diffusion.svg" alt="Diffusion">
<div class="rc-body">
<strong><a href="l11-diffusion-models.html">L11. Diffusion Models</a></strong>
<p>Fixed noising forward, learned denoising back, ELBO training. A
thousand easy steps beat one jump.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l12-fm.svg" alt="Foundation models">
<div class="rc-body">
<strong><a href="l12-foundation-models.html">L12. Foundation Models</a></strong>
<p>Pre-train once, adapt everywhere. Embeddings, linear probing, LoRA.
The paradigm shift, named here.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l13-rag.svg" alt="RAG">
<div class="rc-body">
<strong><a href="l13-contrastive-rag.html">L13. Contrastive, RAG</a></strong>
<p>Self-supervision without labels. RAG: private knowledge, no
training. Behavior vs knowledge, split.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l14-mask.svg" alt="Causal mask">
<div class="rc-body">
<strong><a href="l14-transformers.html">L14. Transformers</a></strong>
<p>Autoregressive modeling, attention, causal masks, O(T^2 d). The
engine under everything modern.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l15-sft.svg" alt="SFT">
<div class="rc-body">
<strong><a href="l15-efficient-icl-sft.html">L15. Efficiency, ICL, SFT</a></strong>
<p>KV cache, MoE, zero/few-shot, instruction tuning. Prompt for
flexibility, train for reliability.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l16-pg.svg" alt="Policy gradient">
<div class="rc-body">
<strong><a href="l16-reinforcement-learning.html">L16. RL Basics</a></strong>
<p>MDPs, values, Bellman, REINFORCE. Sequential decisions, policy
gradient, stochasticity is load-bearing.</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l17-verify.svg" alt="Verifiable reward">
<div class="rc-body">
<strong><a href="l17-rl-for-llms.html">L17. RL for LLMs</a></strong>
<p>PPO, advantages, verifiable rewards, chain-of-thought. The course
ends at the frontier.</p>
</div>
</div>
</div>
