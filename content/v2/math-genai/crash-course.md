---
page_id: math-genai-crash-course
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 99
nav: "Crash course"
title: "MATH-GENAI Crash Course"
summary: "The whole arc in fifteen minutes: from counting what exists to making what does not."
instructor: "Prof. Prathosh A P"
offering: "2025"
---

## The fifteen-minute arc

**The job (L01).** We have samples from an unknown distribution
P_X. We want new samples. The recipe: pick a parametric family
P_θ, choose a divergence D(P_X, P_θ), turn θ to shrink it.
Memorization fails: it assigns zero probability to everything
unseen. The push-forward trick (noise z through a network g_θ)
gives samples with no density formula.

**The ruler (L02).** KL divergence: Σ P_X log(P_X/P_θ), the
average log surprise ratio. Coin toy: 0.511 nats. Exact match
scores 0. Forward KL (weight by truth) is mode-covering.
reverse is mode-seeking. The identity: min KL = max
likelihood. Training is (1/n)Σ log P_θ(x_i).

**The family (L03).** KL is one f-divergence of many: JS
(0.102, symmetric, saturates), TV (0.4, simple, kinked).
The variational bound measures divergence from samples
alone: max over critics of E[T(x)] − E[f*(T(x̂))]. The GAN
objective falls out of one choice of f.

**The game (L04).** D maximizes, G minimizes
E[log D(x)] + E[log(1−D(G(z)))]. Hand trace works. Two
failures: saturation (at D = 0.001, doubling the score
moves the loss 0.001. The non-saturating variant moves it
0.693) and mode collapse (JS 0.216 with half the
distribution missing).

**The fixes (L05).** Wasserstein distance = earth-moving
cost: W = |θ|, gradient 1 everywhere, no saturation.
WGAN: raw critic scores with a 1-Lipschitz speed limit.
FID judges without likelihood: Fréchet distance between
Inception-feature Gaussians (toy: 4).

**The hidden causes (L06).** Data comes from hidden z:
p_θ(x) = ∫ p_θ(x,z)dz. The log of the integral is
intractable, so Jensen builds a floor: ELBO =
E_q[log p_θ(x,z)/q(z|x)]. Gap = KL(q‖true posterior),
exactly 0.063 in the hand toy. EM is the exact-posterior
special case.

**The autoencoder (L07).** q becomes an encoder network.
Sampling blocks backprop. The reparameterization trick
(z = μ + σε) moves randomness to ε. KL rent: 0.443 nats
in the toy. Posterior collapse (KL → 0, latents dead) is
the failure. β-VAE and VQ-VAE are the variants.

**The destruction (L08).** Fix the encoder to pure
noising: x_t = √α_t x_{t−1} + √(1−α_t)ε_t. Toy:
4.0 → 4.111 → 3.900. Jump anywhere: x_t = √ᾱ_t x_0 +
√(1−ᾱ_t)ε. End of chain: pure N(0,1). Learn only the
reverse: small denoising steps.

**The simplification (L09).** Condition on known x_0:
tractable posterior (mean 3.944 in the toy). Gaussian KL
= squared error. Reparameterize through noise: the whole
ELBO becomes L_simple = ‖ε − ε_θ‖² (0.0354 in the toy).
Sampling walks the chain backward (3.9 → 4.059).

**The views (L10).** Noise prediction is score
prediction (−1.578 in the toy): walk uphill on
probability. DDIM strides skip steps (x_100 = 1.5 →
x_50 = 2.844 in one jump). Guidance steers with a
classifier (−0.578). Latent diffusion shrinks 48×.
Autoregressive models factor with the chain rule
(p(a,b,a) = 0.036): exact likelihood, sequential
sampling.

## The one table

| Family | Objective | Price |
|---|---|---|
| GAN | Minimax game | Saturation, mode collapse |
| VAE | ELBO | Blurry samples |
| Diffusion | Noise MSE | Slow sampling |
| Autoregressive | Exact MLE | Sequential generation |

Four answers to "how do you teach a machine to create?",
each with its bill attached. For the full story with
every number worked by hand, read the lessons in order.
