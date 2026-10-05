---
page_id: math-genai-l07
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 7
nav: "L07 · VAE"
title: "Lecture 7: Variational Autoencoders"
summary: "The ELBO made trainable: an encoder network guesses the causes, a decoder renders from them, and the reparameterization trick lets gradients flow through sampling. KL computed by hand (0.443), posterior collapse shown as a number."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
video_id: blh_AnhwIpw
concepts: [vae, reparameterization-trick, encoder, decoder, posterior-collapse, beta-vae, vq-vae]
sources:
  - tag: video
    label: "W5L20: Variational Autoencoder, VAE (video RN3_gkjlYoA)"
    url: https://www.youtube.com/watch?v=RN3_gkjlYoA
  - tag: video
    label: "W6L21: Training VAE, reparameterization (video blh_AnhwIpw)"
    url: https://www.youtube.com/watch?v=blh_AnhwIpw
  - tag: paper
    label: "Kingma and Welling, Auto-Encoding Variational Bayes (2013)"
    url: https://arxiv.org/abs/1312.6114
---

## The task: learn the guesser

Lesson 6 derived the ELBO but left q(z|x), the guess at the
hidden causes, as an abstract distribution. The VAE makes it
concrete: q is a neural network, the **encoder**. It takes a
data point x and outputs a distribution over causes. A second
network, the **decoder**, takes a cause z and outputs a
distribution over data. The prior p(z) is fixed to the
standard Gaussian, N(0, 1): causes live in a round, centered
cloud.

```ascii
x (photo) --encoder-->  q(z|x) = N(mu(x), sigma(x))  --sample--> z
                                                              |
z --decoder--> p(x|z) (reconstructed photo)                   |
                                                              v
                                                     prior: z ~ N(0, 1)
```

The encoder compresses. The decoder renders. The latent space
is the compressed imagination from Lesson 6, now with
coordinates a network learns.

Concretely, the encoder outputs two vectors: μ(x), the most
likely causes, and σ(x), the uncertainty. So
q(z|x) = N(μ(x), diag(σ(x)²)): a Gaussian per data point.
The ELBO becomes a loss a computer can evaluate:

```ascii
loss = -E_q[ log p_theta(x|z) ]  +  KL( q(z|x) || N(0,1) )
        reconstruction error         keep causes near the prior
```

For a Gaussian decoder, the reconstruction term is the
squared pixel error between x and the decoder's output.
Familiar territory: an autoencoder. The KL term is the new
rent from Lesson 6.

## Where sampling breaks backprop

Training needs the gradient of the loss with respect to the
encoder's weights. The reconstruction term is an expectation
over z ~ q(z|x), estimated by sampling. Here is the wall:
sampling is not differentiable. The operation "draw z at
random" has no gradient. Backpropagation stops at the random
draw, and the encoder never learns.

The naive fix, the score-function estimator, differentiates
through the sampling probabilities instead. It works but its
variance is enormous: the gradient estimate jumps wildly
between samples, and training crawls. The lecture's answer is
cleaner.

## The key question

How do you draw a random sample whose gradient flows back
into the network that chose the distribution?

## The new idea: move the randomness out

The **reparameterization trick**: do not sample from
N(μ, σ²) directly. Sample ε from N(0, 1), a fixed
distribution with no parameters, and set:

```ascii
z = mu + sigma * epsilon,   epsilon ~ N(0, 1)
```

If ε ~ N(0,1), then μ + σε ~ N(μ, σ²). Same distribution,
different plumbing. The randomness now lives in ε, which
has no knobs. The knobs μ and σ sit in a deterministic
formula, and gradients flow through them:

```ascii
dz/dmu = 1,   dz/dsigma = epsilon
```

Work it with numbers. The encoder says μ = 0.5, σ = 0.5
for some x. Draw ε = 1.2 from the standard Gaussian.

```ascii
z = 0.5 + 0.5 * 1.2 = 1.1
```

The sample is 1.1. If the loss says "z should have been
smaller", the gradient flows: ∂z/∂μ = 1 (raise μ raises
z one-for-one), ∂z/∂σ = 1.2 (σ's effect scales with the
draw). The encoder learns from every sample. One line of
algebra removed the wall.

## The KL term, by hand

The regularization term also needs computing:
KL(q(z|x) || N(0,1)) for q = N(μ, σ²). For Gaussians this
has a closed form (per dimension):

```ascii
KL = -0.5 * ( 1 + log(sigma^2) - mu^2 - sigma^2 )
```

With μ = 0.5, σ² = 0.25:

```ascii
KL = -0.5 * ( 1 + log(0.25) - 0.25 - 0.25 )
   = -0.5 * ( 1 - 1.386 - 0.5 )
   = -0.5 * ( -0.886 )
   = 0.443
```

The encoder pays 0.443 nats of rent for placing its
causes at 0.5 ± 0.5 instead of at the prior's 0 ± 1.
Read the formula's incentives: log(σ²) punishes tiny
σ (overconfidence), −μ² punishes drift from zero,
−σ² punishes excess spread. The cheapest q is the
prior itself (μ = 0, σ = 1 gives KL = 0), but then the
causes carry no information about x and reconstruction
suffers. Training balances the two terms automatically.

Now the full training step is mechanical: encode x to
(μ, σ). Draw ε. Form z = μ + σε. Decode to x̂. Loss =
pixel error + 0.443-style KL. Backpropagate through
everything. Sampling, once the wall, is now one line.

## Where it breaks: posterior collapse

The balance can tip pathologically. Suppose the decoder
is powerful enough to model the data alone, ignoring z
entirely. Then the ELBO is maximized by setting q(z|x) =
p(z) for every x: KL = 0, reconstruction handled by the
decoder's own strength. The numbers look great (KL fell
from 0.443 to 0!) while the latent space died: z carries
zero information about x, and sampling z produces no
variety. This is **posterior collapse**. The model
technically optimized the bound and learned nothing.

Detect it by the KL term: healthy training keeps the KL
clearly above zero (the encoder is saying something).
A KL glued to zero with good reconstructions is the
corpse, not the success. Fixes include weakening the
decoder, annealing the KL weight from 0 upward during
training, or the β-VAE variant below.

Two descendants from the lecture, briefly. **β-VAE**
multiplies the KL term by β > 1. The extra rent pressure
forces each latent dimension to earn its keep, which
empirically disentangles causes (one dimension for pose,
one for lighting). The price is worse reconstruction:
higher β, blurrier images. **VQ-VAE** replaces the
continuous Gaussian with a discrete codebook: z snaps
to the nearest of K learned vectors. No KL-to-Gaussian
term, no collapse of the same kind, and the discrete
codes suit transformers downstream. The lecture covers
both as answers to the VAE's weaknesses.

## The honest price

The VAE's samples are blurry, and now you know exactly
why: three compounding causes. The forward-KL heritage
(Lesson 2) covers modes and generates in-between blends.
The Gaussian decoder assumes pixel noise, which smears
edges. And the diagonal-Gaussian q cannot express
multimodal posteriors, so the bound stays loose where
the truth is complex. The VAE won stability (plain
maximization, no adversary) and a usable latent space
(interpolate, edit, infer), and paid in sharpness.
Diffusion keeps the stability and reclaims the
sharpness. Lessons 8-10.

| VAE pain | Mechanism | Number |
|---|---|---|
| Cannot backprop through sampling | Reparameterization: z = μ + σε | ∂z/∂μ = 1, ∂z/∂σ = ε = 1.2 in the toy |
| KL term looks intractable | Closed form for Gaussians | 0.443 for μ = 0.5, σ² = 0.25 |
| Posterior collapse | Decoder ignores z. KL → 0 | KL 0 with good reconstructions = dead latents |
| Blurry samples | Forward KL + Gaussian decoder + diagonal q | Price of stability |

> [!QA]
> Q: What is the reparameterization trick?
> A: Rewrite sampling from N(μ, σ²) as z = μ + σ·ε with ε ~ N(0,1) fixed. The distribution is identical, but the randomness moved to the parameter-free ε, so gradients flow: ∂z/∂μ = 1, ∂z/∂σ = ε. In the toy, μ = 0.5, σ = 0.5, ε = 1.2 gave z = 1.1, and the encoder learns from that sample.
> Follow-up: Why not just use the score-function estimator?
> A: It differentiates through sampling probabilities and is unbiased, but its variance is so large that training needs far more samples to make progress. Reparameterization gives a low-variance gradient from a single sample. It only works for continuous variables you can reparameterize. Discrete latents need other tricks (VQ-VAE sidesteps this).

> [!QA]
> Q: What does the VAE loss actually compute?
> A: Reconstruction error plus KL(q(z|x) || N(0,1)). The encoder outputs μ(x), σ(x). You sample z = μ + σε, decode, and pay pixel error plus the KL rent. For μ = 0.5, σ² = 0.25 the rent is 0.443 nats. The loss balances fitting the data against keeping causes near the prior.
> Follow-up: What is posterior collapse?
> A: The decoder learns to ignore z, so the ELBO is maximized by q(z|x) = prior: KL = 0, reconstructions fine, latents dead. A KL glued to zero alongside good reconstructions is the diagnostic. Fixes: weaker decoder, KL annealing, or β-VAE/VQ-VAE variants.

> [!QA]
> Q: Why are VAE samples blurry?
> A: Three compounding reasons. The forward-KL objective is mode-covering: it spreads mass over all modes including the valleys between them. The Gaussian decoder models pixels as independent noise, smearing edges. The diagonal-Gaussian q cannot capture complex posteriors, leaving the bound loose. GANs look sharper because their JS-like objective tolerates dropping modes instead of blending them.
> Follow-up: What is β-VAE buying with its β?
> A: Disentanglement at the cost of fidelity. β > 1 raises the KL rent, forcing each latent dimension to justify itself, which empirically separates causes like pose and lighting into different dimensions. Reconstruction gets worse as β rises. It is a dial between interpretability and sharpness.

## Recap: the whole lesson on one screen

1. **The task.** Make q(z|x) a neural network (encoder) and learn it with the decoder.
2. **Where it breaks.** Sampling z is not differentiable. Backprop stops at the random draw.
3. **The key question.** How do you differentiate through a sample?
4. **The fix.** z = μ + σ·ε, ε ~ N(0,1). Randomness moves to ε. Gradients flow: ∂z/∂μ = 1, ∂z/∂σ = ε.
5. **Worked.** μ = 0.5, σ = 0.5, ε = 1.2 → z = 1.1. KL rent = 0.443 nats by the closed form.
6. **Training.** Encode, sample, decode, pay pixel error + KL, backpropagate through everything.
7. **Posterior collapse.** KL → 0 with good reconstructions means the latents died. The decoder works alone.
8. **The price.** Blurry samples (forward KL + Gaussian decoder + diagonal q) in exchange for stability and a usable latent space.

## Official sources and further reading

**Official:**
- W5L20: Variational Autoencoder (VAE):
  https://www.youtube.com/watch?v=RN3_gkjlYoA
- W6L21: Training VAE, reparameterization methods:
  https://www.youtube.com/watch?v=blh_AnhwIpw
- W6L24/L25: Beta-VAE, VQ-VAE.

**Further reading:**
- Kingma and Welling, "Auto-Encoding Variational Bayes" (2013):
  https://arxiv.org/abs/1312.6114: reparameterization and the VAE loss.
- Higgins et al., "β-VAE" (2017):
  https://openreview.net/forum?id=Sy2fzU9gl: the disentanglement dial.
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017):
  https://arxiv.org/abs/1711.00937: discrete latents.

**Caveats.** The W5L20 transcript was bot-blocked, so this lesson follows the standard Kingma-Welling presentation with the ELBO framing confirmed in the W5L18 transcript. [uncertain] The lecture's exact examples, its reparameterization variants, and its β-VAE/VQ-VAE emphasis are unknown.

## Connections to the other courses

- **CS229 L10 (EM/PCA):** that course shows PCA is the optimal linear compressor. A linear VAE recovers the same answer. Worked tie: data (2,1), (1,2), (−2,−1), (−1,−2) has covariance [[2.5, 2],[2, 2.5]], whose top eigenvector is (1,1)/√2 with eigenvalue 4.5. The optimal 1-D codes are the projections: ±3/√2 ≈ ±2.12. A linear VAE's ELBO is maximized by exactly this direction: PCA is the VAE with linear encoder, linear decoder, and no KL.
- **CS229 L11 (diffusion models):** that course's models can be read as VAEs with a very deep hierarchy of latents and a fixed encoder. Lessons 8-9 make that reading exact.
