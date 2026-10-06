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
video_title: "W6L21: Training VAE: Reparameterization methods"
video_caption: "The lecture video for this lesson: training the VAE, reparameterization on the board. Timestamps in the text link to the exact moment."
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

### The loss a computer can evaluate

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

### The wall

Training needs the gradient of the loss with respect to the
encoder's weights. The reconstruction term is an expectation
over z ~ q(z|x), estimated by sampling. Here is the wall:
sampling is not differentiable. The operation "draw z at
random" has no gradient. Backpropagation stops at the random
draw, and the encoder never learns.

### The naive fix and its variance

The naive fix, the score-function estimator, differentiates
through the sampling probabilities instead. It works but its
variance is enormous: the gradient estimate jumps wildly
between samples, and training crawls. The variance scales
with the dimension of z and the sharpness of the loss
landscape. In practice you need many samples per step to
make progress, which is expensive. The lecture's answer is
cleaner.

## The key question

How do you draw a random sample whose gradient flows back
into the network that chose the distribution?

## The new idea: move the randomness out

### The trick

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

### Worked with numbers

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

### Why the distribution is unchanged

The trick is not an approximation. A linear transform of a
Gaussian is exactly Gaussian: ε ~ N(0,1) implies
μ + σε ~ N(μ, σ²) with no error. The samples come from
the identical distribution the encoder specified. Only the
plumbing changed: randomness moved from the parameterized
node to a fixed node upstream. That is why the ELBO stays
exact and only the gradient estimator changes.

![Move the randomness out: z = mu + sigma * eps](assets/l07-reparam.webp "Same distribution N(mu, sigma^2). Gradients flow through mu and sigma. Shell 3. Source: original toy. Project: Stanford Frontier AI.")

## The KL term, by hand

### The closed form

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

### Reading the incentives

Read the formula's incentives: log(σ²) punishes tiny
σ (overconfidence), −μ² punishes drift from zero,
−σ² punishes excess spread. The cheapest q is the
prior itself (μ = 0, σ = 1 gives KL = 0), but then the
causes carry no information about x and reconstruction
suffers. Training balances the two terms automatically.

![The KL rent: 0.443 nats, and what each term punishes](assets/l07-kl-term.webp "Cheapest q is the prior itself. Shell 2. Source: original toy. Project: Stanford Frontier AI.")

### The full training step, mechanical

Now the full training step is mechanical: encode x to
(μ, σ). Draw ε. Form z = μ + σε. Decode to x̂. Loss =
pixel error + 0.443-style KL. Backpropagate through
everything. Sampling, once the wall, is now one line.

## Where it breaks: posterior collapse

### The pathological balance

The balance can tip pathologically. Suppose the decoder
is powerful enough to model the data alone, ignoring z
entirely. Then the ELBO is maximized by setting q(z|x) =
p(z) for every x: KL = 0, reconstruction handled by the
decoder's own strength. The numbers look great (KL fell
from 0.443 to 0!) while the latent space died: z carries
zero information about x, and sampling z produces no
variety. This is **posterior collapse**. The model
technically optimized the bound and learned nothing.

### The diagnostic

Detect it by the KL term: healthy training keeps the KL
clearly above zero (the encoder is saying something).
A KL glued to zero with good reconstructions is the
corpse, not the success. The decision rule: plot the KL
per dimension over training. If all dimensions sit at
~0 while the reconstruction loss falls, the latents are
dead. Do not celebrate the low total loss. It is the
bound being gamed, exactly the failure Lesson 6 warned
about.

![Posterior collapse: KL = 0 with good reconstructions is the corpse](assets/l07-collapse.webp "Watch the KL term, not the total loss. Shell 3. Source: original. Project: Stanford Frontier AI.")

### The fixes: three answers

Fixes include weakening the decoder (a less powerful
decoder must lean on z), annealing the KL weight from 0
upward during training (let reconstruction win early,
then charge rent), or the β-VAE variant below.

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

### Where VAEs run in real systems

The VAE's second life is compression, not generation.
Latent diffusion models compress images with a VAE
encoder into a small latent grid, run diffusion there,
and decode back: the VAE is the codec, diffusion is the
generator. The reparameterization trick itself outlived
the VAE: any model that samples a continuous latent
during training uses it. β-VAE's disentanglement dial
survives in representation-learning research. VQ-VAE's
discrete codebook became the standard way to tokenize
images and audio for transformer models. [uncertain]
Which current production systems use each variant is not
public.

## Videos for this lesson

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/RN3_gkjlYoA" title="W5L20: Variational Autoencoder (VAE)" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Lecture video: the VAE, encoder to decoder, the ELBO made concrete. If the embed is blocked: <a href="https://www.youtube.com/watch?v=RN3_gkjlYoA" target="_blank" rel="noopener">watch on YouTube</a>.</p></div>

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/hMsQLwxYHhQ" title="5 Types of Autoencoders Explained Visually" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">External explainer: vanilla autoencoders versus VAEs, and where VQ-VAE and masked variants fit. If the embed is blocked: <a href="https://www.youtube.com/watch?v=hMsQLwxYHhQ" target="_blank" rel="noopener">watch on YouTube</a>.</p></div>

> [!QA]
> Q: What is the reparameterization trick?
> A: Rewrite sampling from N(μ, σ²) as z = μ + σ·ε with ε ~ N(0,1) fixed. The distribution is identical, but the randomness moved to the parameter-free ε, so gradients flow: ∂z/∂μ = 1, ∂z/∂σ = ε. In the toy, μ = 0.5, σ = 0.5, ε = 1.2 gave z = 1.1, and the encoder learns from that sample.
> Follow-up: Why not just use the score-function estimator?
> A: It differentiates through sampling probabilities and is unbiased, but its variance is so large that training needs far more samples to make progress. Reparameterization gives a low-variance gradient from a single sample. It only works for continuous variables you can reparameterize. Discrete latents need other tricks (VQ-VAE sidesteps this).

> [!QA]
> Q: Walk me through a reparameterization on fresh numbers.
> A: Encoder outputs μ = −1.0, σ = 2.0. Draw ε = 0.5. Then z = −1.0 + 2.0·0.5 = 0.0. Gradients: ∂z/∂μ = 1, so raising μ by 0.1 raises z by 0.1. ∂z/∂σ = 0.5, so raising σ by 0.1 raises z by 0.05. The KL rent for this q: −0.5·(1 + log 4 − 1 − 4) = −0.5·(1 + 1.386 − 5) = 1.807 nats. Expensive: the encoder placed its mass far from the prior and wide.
> Follow-up: The KL is 1.807. Is that good or bad?
> A: Neither by itself. It is the rent for an informative q. If reconstruction is excellent, the rent is justified. If the KL were 0.001 with the same reconstruction, the encoder would be saying nothing and the latents would be dead. Judge the KL against what the latents buy.

> [!QA]
> Q: What does the VAE loss actually compute?
> A: Reconstruction error plus KL(q(z|x) || N(0,1)). The encoder outputs μ(x), σ(x). You sample z = μ + σε, decode, and pay pixel error plus the KL rent. For μ = 0.5, σ² = 0.25 the rent is 0.443 nats. The loss balances fitting the data against keeping causes near the prior.
> Follow-up: What is posterior collapse?
> A: The decoder learns to ignore z, so the ELBO is maximized by q(z|x) = prior: KL = 0, reconstructions fine, latents dead. A KL glued to zero alongside good reconstructions is the diagnostic. Fixes: weaker decoder, KL annealing, or β-VAE/VQ-VAE variants.

> [!QA]
> Q: Derive the KL closed form check: q = prior. What do you get?
> A: μ = 0, σ² = 1. Plug in: −0.5·(1 + log 1 − 0 − 1) = −0.5·(0) = 0. Correct: identical distributions have zero KL. Now q = N(0, 0.01): −0.5·(1 + log 0.01 − 0 − 0.01) = −0.5·(1 − 4.605 − 0.01) = 1.807. A confident spike far from the prior's spread pays 1.8 nats. The log term is what punishes overconfidence.
> Follow-up: Why does the formula punish small σ so hard?
> A: Because log(σ²) → −∞ as σ → 0. A spike claims near-certainty about the cause, and the prior (spread 1) disagrees violently. The rent prices the disagreement. This is the mechanism that keeps the latent space smooth and sampleable.

> [!QA]
> Q: Why are VAE samples blurry?
> A: Three compounding reasons. The forward-KL objective is mode-covering: it spreads mass over all modes including the valleys between them. The Gaussian decoder models pixels as independent noise, smearing edges. The diagonal-Gaussian q cannot capture complex posteriors, leaving the bound loose. GANs look sharper because their JS-like objective tolerates dropping modes instead of blending them.
> Follow-up: What is β-VAE buying with its β?
> A: Disentanglement at the cost of fidelity. β > 1 raises the KL rent, forcing each latent dimension to justify itself, which empirically separates causes like pose and lighting into different dimensions. Reconstruction gets worse as β rises. It is a dial between interpretability and sharpness.

> [!QA]
> Q: Your VAE trains to low loss but interpolations between two faces jump instead of morphing. Diagnose it.
> A: The latent space is not smooth: nearby z values decode to unrelated images. Likely cause: the KL term is too weak relative to reconstruction (β effectively below 1, or annealing never ramped up), so the encoder placed data points in isolated islands with empty space between them. Sampling or interpolating through the gaps hits undecodable regions. Fix: raise the KL weight toward 1 and retrain, or check for partial posterior collapse on some dimensions.
> Follow-up: How do you verify smoothness directly?
> A: Linearly interpolate z between two encodings in 10 steps and decode each. Smooth morphing means a smooth space. Jumps mean islands. Also sample z ~ N(0,1) fresh: if the samples look nothing like reconstructions, the encoder's q has drifted from the prior and the prior is no longer a valid sampler.

> [!QA]
> Q: Design a VAE-based anomaly detector for factory sensor readings. How do the pieces map?
> A: Train a VAE on normal readings only. The encoder q(z|x) compresses each reading. The decoder reconstructs it. The anomaly score is the reconstruction error plus the KL. Normal data reconstructs well with low rent. Anomalous data either reconstructs badly (unseen pattern) or needs an exotic q (high KL). Threshold the sum on held-out normal data. The reparameterization trick is what makes the encoder trainable. Without it there is no gradient.
> Follow-up: Why not just use the reconstruction error alone?
> A: A powerful decoder can reconstruct anomalies too (it memorized the manifold broadly). The KL term catches the cases where the encoder had to contort q to explain the input. Both terms are the ELBO. Dropping one drops half the evidence signal.

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
- W5L20: Variational Autoencoder (VAE): [paper](https://www.youtube.com/watch?v=RN3_gkjlYoA)
- W6L21: Training VAE, reparameterization methods: [paper](https://www.youtube.com/watch?v=blh_AnhwIpw)

**Further reading:**
- Kingma and Welling, "Auto-Encoding Variational Bayes" (2013):
  - [reparameterization and the VAE loss.](https://arxiv.org/abs/1312.6114)
- Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017):
  - [the disentanglement dial.](https://arxiv.org/abs/1802.06875)
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017):
  - [discrete latents.](https://arxiv.org/abs/1711.00937)

**Caveats.** The W5L20 transcript was bot-blocked, so this lesson follows the standard Kingma-Welling presentation with the ELBO framing confirmed in the W5L18 transcript. [uncertain] The lecture's exact examples, its reparameterization variants, and its β-VAE/VQ-VAE emphasis are unknown.

## Connections to the other courses

- **CS229 L10 (EM/PCA):** that course shows PCA is the optimal linear compressor. A linear VAE recovers the same answer. Worked tie: data (2,1), (1,2), (−2,−1), (−1,−2) has covariance [[2.5, 2],[2, 2.5]], whose top eigenvector is (1,1)/√2 with eigenvalue 4.5. The optimal 1-D codes are the projections: ±3/√2 ≈ ±2.12. A linear VAE's ELBO is maximized by exactly this direction: PCA is the VAE with linear encoder, linear decoder, and no KL.
- **CS229 L11 (diffusion models):** that course's models can be read as VAEs with a very deep hierarchy of latents and a fixed encoder. Lessons 8-9 make that reading exact.
