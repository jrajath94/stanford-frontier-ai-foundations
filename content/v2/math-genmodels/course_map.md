# Course map: math-genmodels

Date: 2026-10-06. Baseline: October 6, 2026.

HONESTY NOTE: unit order is the builder's provisional bridge order,
not asserted official syllabus. See README.md and source_gaps.md G1-G2.

## The spine

Generative modeling asks one question: how do we learn a probability
law from samples, and then draw new samples from what we learned? The
eight units answer it in order of mathematical dependency.

## Unit dependency graph

U01 Probability, measures, and density estimation (P01, P06, P07, P08)
  -> U02 Latent models and variational inference (P05, P08, P18)
  -> U03 Autoencoders and representation objectives (P04, P11, P18)
  -> U04 Change of variables and normalizing flows (P04, P05, P08)
  -> U05 Adversarial and ratio-based estimation (P08, P09, P11)
  -> U06 Scores, diffusion, and stochastic dynamics (P05, P06, P18)
  -> U07 Autoregressive models and modern bridges (P13, P14, P15)
  -> U08 Evaluation, inference, and experimental synthesis (P07, P10, P22)

Cross links: U01 feeds every unit. U02 feeds U03 (VAE), U05 (divergence
choice), and U06 (variational view of diffusion). U04 and U07 both give
exact likelihood and are compared in U07-C12 and U08. U05 and U06 meet
in score-based versus adversarial tradeoffs in U08. U08 closes the loop
back to U01: held-out likelihood is density estimation judged honestly.

## Unit summaries

- U01: the objects. Probability versus density, support, normalization,
  discrete and continuous variables, joint and marginal laws, sampling,
  likelihood, the empirical distribution, estimation, entropy,
  divergence, identifiability. File: lessons/u01.md.
- U02: the intractable integral. Latent marginalization, posterior,
  Bayes, Jensen, the ELBO identity, the variational gap, forward versus
  reverse KL, reparameterization, Monte Carlo variance, amortization,
  expressive families, degeneracy. File: lessons/u02.md.
- U03: codes that reconstruct and codes that generate. Deterministic
  reconstruction, generative versus representation goals, VAE, Gaussian
  KL in closed form, posterior collapse, beta weighting, discrete
  latents, codebooks, straight-through bias, likelihood choice,
  rate/distortion, evaluation. File: lessons/u03.md.
- U04: exact likelihood by construction. Invertibility, determinant as
  volume, Jacobian, the density transform, coupling and autoregressive
  flows, triangular Jacobians, base densities, exact likelihood,
  sampling direction, computational asymmetry, expressivity, numerical
  constraints. File: lessons/u04.md.
- U05: learning without a density formula. Implicit distributions,
  density ratios, the discriminator optimum, the GAN objective,
  divergence assumptions, optimal transport, the Lipschitz constraint,
  WGAN and gradient penalty, mode collapse, identifiability,
  stability, evaluation limits. File: lessons/u05.md.
- U06: gradients of log-density. Score, score matching, denoising,
  the noising chain, reverse kernels, the DDPM loss, DDIM, Langevin
  dynamics, SDE/ODE assumptions, conditioning and guidance, solvers,
  support and boundary behavior. File: lessons/u06.md.
- U07: the chain rule at scale. Chain rule factorization, conditional
  densities, discrete tokens, teacher forcing, causal networks, the
  transformer as density model, sampling, exact likelihood, context
  cost, masked objectives, flow matching as a labeled extension, model
  comparison. File: lessons/u07.md.
- U08: judging generators honestly. Held-out likelihood, bound versus
  exact scores, mode coverage, sample precision and recall, FID limits,
  memorization, compute-matched baselines, synthetic ground truth,
  uncertainty, falsification, the source-identity report, the role gap
  map. File: lessons/u08.md.

## Files per unit

Each unit ships: lessons/u0N.md, lessons/keys/keys-u0N.md,
labs/lab-u0N.md, labs/keys/keys-u0N.md, visuals/u0N/*.png with
visuals/render_u0N.py. Shared banks: interview/transfer-sets.md,
interview/keys-transfer.md, interview/oral-defenses.md,
interview/keys-oral.md. Capstones: capstones/capstone-research-gp-critic.md,
capstones/capstone-applied-fde.md.

## Diagnostics

prerequisites.md holds the entry diagnostic: 12 questions, one per
bridge area, with a pass bar and a remediation pointer per question.
Take it before U01. A fail on any bridge sends the learner to the
shared P-module and the unit-local remediation, not forward.
