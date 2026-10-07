# course_map.md, cs229

Date: 2026-10-06. 17 parent units, 204 leaf concepts. Build complete 2026-10-06: 204/204 rows TAUGHT+ASSESSED.

## Part 1: supervised learning foundations

- U01 Learning problems, risk, and mathematical setup (12 concepts).
  Setup language for everything: features, targets, hypothesis,
  empirical vs population risk, losses, train/test assumptions.
- U02 Linear regression and local weighting (12). LMS, normal
  equations, probabilistic reading, locally weighted regression.
- U03 Classification and generalized linear models (12). Logistic
  regression, softmax, exponential family, GLMs, link functions.
- U04 Generative classifiers and density assumptions (12). GDA,
  naive Bayes, Laplace smoothing, misspecification, calibration.
- U05 Feature maps and kernel methods (12). Kernel trick, Gram
  matrix, valid kernels, costs.
- U06 Margins, SVMs, and dual optimization (12). Functional and
  geometric margin, duality, KKT, support vectors, SMO, slack.

## Part 2: deep learning

- U07 Neural networks and backpropagation (12). Forward pass,
  reverse-mode autodiff, activations, init, normalization,
  gradient checks, optimization failures.

## Part 3: generalization and regularization

- U08 Generalization and sample complexity (12). Bias/variance,
  finite and infinite hypothesis classes, union bound, uniform
  convergence.
- U09 Regularization and model selection (12). Penalties, implicit
  regularization, early stopping, MAP, cross-validation, leakage.

## Part 4: unsupervised learning

- U10 Clustering and latent mixture models (12). k-means, EM,
  Jensen bound, mixture likelihoods, covariance collapse.
- U11 Variational inference, VAE, PCA, and ICA (12). ELBO, VAE,
  eigen/SVD PCA, ICA ambiguities, source separation.

## Part 5: generative models and foundation models

- U12 Diffusion models (12). Forward/reverse Gaussian chains, ELBO,
  denoising objective, score relation, sampling.
- U13 Foundation models and representations (12). Pretraining,
  probing, fine-tuning, LoRA, contrastive objectives, RAG.
- U14 LLM architecture and training (12). Tokenization, autoregression,
  transformer, attention variants, MoE, SFT, in-context learning.

## Part 6: reinforcement learning and control

- U15 Reasoning and reinforcement learning foundations (12). MDP,
  Bellman, value/policy iteration, exploration, reward validity.
- U16 Control and policy optimization (12). LQR, Riccati, DDP,
  LQG, REINFORCE, baselines, PPO.

## Appendices

- U17 Appendices, historical supplements, research synthesis (12).
  Gaussian/KL identities, HMM and GP supplements, chapter map,
  baselines, ablations, oral defense, deployment gap map.

## Dependency spine

U01 feeds every unit. U02 feeds U03, U05, U06. U05 feeds U06.
U07 feeds U12-U14. U08 feeds U09. U10 feeds U11. U11 feeds U12.
U15 feeds U16. U17 supports all.
