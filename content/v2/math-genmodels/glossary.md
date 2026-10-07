# Glossary: math-genmodels

Date: 2026-10-06. Baseline: October 6, 2026.

- Amortization: one shared network predicts per-point variational
  parameters instead of optimizing each point separately.
- Base density: the simple law (often Gaussian) a flow transforms.
- Change of variables: the density transform under an invertible map,
  with the Jacobian determinant as volume correction.
- Codebook: the finite set of vectors a VQ model quantizes latents to.
- Coupling layer: a flow layer that transforms half the coordinates
  with parameters from the other half, giving a triangular Jacobian.
- DDIM: Denoising Diffusion Implicit Models, a deterministic sampler
  for diffusion models with fewer steps.
- DDPM: Denoising Diffusion Probabilistic Models, the standard
  discrete-time diffusion training and sampling recipe.
- Causal mask: a lower-triangular mask that stops each position
  from reading future positions.
- Chain rule: p(x_1..x_L) as a product of conditionals, one per
  position.
- Classifier-free guidance: steering a diffusion sampler by
  extrapolating between conditional and unconditional scores.
- Density ratio: p(x)/q(x), the object a discriminator estimates
  without modeling either density directly.
- Discriminator optimum: D*(x) = p(x)/(p(x)+q(x)), the best
  real-versus-fake classifier for a fixed generator.
- Denoising: training on noisy samples to learn the score of the
  noisy law. Tweedie turns the score into a clean estimate.
- ELBO: evidence lower bound. A tractable lower bound on log
  likelihood, tight exactly when the approximate posterior matches the
  true posterior.
- Empirical distribution: the law that puts mass 1/n on each observed
  sample.
- Exposure bias: the train-sample mismatch of teacher forcing. The
  model never learns to recover from its own errors.
- Falsification: stating what would prove a claim wrong before
  running the experiment.
- FID: Frechet Inception Distance. Compares Gaussian fits to feature
  means and covariances. Blind past second moments.
- Flow matching: training a continuous-time velocity field to match a
  target probability path, a labeled extension in U07-C11.
- Forward KL: D_KL(p||q), mean-seeking, covers all of p. Reverse KL:
  D_KL(q||p), mode-seeking.
- GAN: Generative Adversarial Network. A generator and a
  discriminator trained against each other.
- Gradient penalty: a soft 1-Lipschitz enforcement that penalizes
  the critic's gradient norm away from 1 on interpolated points.
- Guidance: conditioning a diffusion sampler toward a class or prompt,
  e.g. classifier-free guidance.
- Held-out likelihood: the likelihood on data the model did not
  train on. The honest generalization score.
- Identifiability: whether distinct parameters give distinct observable
  laws. Non-identifiable models admit silent parameter swaps.
- Implicit distribution: a law defined only by a sampling procedure,
  with no tractable density formula.
- Jacobian: the matrix of partial derivatives of a vector map.
- Jensen inequality: E[phi(X)] >= phi(E[X]) for convex phi. The engine
  of the ELBO.
- Jensen-Shannon divergence: the symmetric divergence the GAN game
  minimizes at the discriminator optimum. Bounded by log 2.
- KV cache: stored keys and values that make autoregressive
  generation O(L^2) instead of O(L^3).
- Langevin dynamics: gradient ascent on log-density plus noise, which
  samples from the density in the limit.
- Likelihood: p(data | parameters), the parameters' score on fixed data.
- Lipschitz constraint: a bound on how fast a function can change,
  used in WGAN to enforce the Kantorovich dual.
- Marginalization: summing or integrating out variables to get a
  marginal law.
- Mode collapse: a generator that covers a subset of the data modes.
- Mode covering: a fit that spreads mass over all data modes, often
  at the cost of blur.
- NLL: negative log-likelihood. The training loss for exact-likelihood
  models.
- Optimal transport: the cheapest way to move mass from one law to
  another. The 1-Wasserstein distance is its cost.
- Perplexity: exp(NLL per token). The effective branching factor of a
  sequence model.
- Posterior: p(z|x), the law over latents given data.
- Posterior collapse: a VAE whose approximate posterior equals the
  prior on some dimensions, so those latents carry no information.
- Precision (generative): fraction of model samples on the data
  manifold. Recall: fraction of data modes the model covers.
- Probability flow ODE: the deterministic twin of the reverse SDE
  with identical marginals. DDIM discretizes it.
- Rate/distortion: the tradeoff between latent bit cost and
  reconstruction error.
- Reparameterization: writing a sample as a deterministic function of
  parameters plus fixed noise, so gradients flow through sampling.
- Score: nabla_x log p(x), the gradient of log-density.
- Score matching: fitting a model to the score instead of the density,
  which avoids the normalizing constant.
- Straight-through estimator: a biased gradient for discrete choices
  that copies the upstream gradient through the hard decision.
- Support: the set where a density is positive. Support mismatch
  breaks likelihood and divergences.
- Teacher forcing: training an autoregressive model on ground-truth
  prefixes instead of its own predictions.
- Tweedie: E[x_0 | x] = x + sigma^2 grad log p(x). The posterior mean
  from a noisy observation under Gaussian noise.
- VAE: Variational Autoencoder. An autoencoder trained with the ELBO.
- WGAN: Wasserstein GAN. Replaces the Jensen-Shannon objective with an
  optimal-transport distance.
