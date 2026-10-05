---
page_id: math-genai-l10
course_slug: math-genai
course_name: "Mathematical Foundations of Generative AI"
course_order: 11
order: 10
nav: "L10 · Score, DDIM, AR"
title: "Lecture 10: Scores, Shortcuts, and the Autoregressive Family"
summary: "Three views that complete diffusion: the score (walk uphill on probability, worked: 3.821), DDIM strides that skip steps (100 → 50 in one jump: 2.844), guided and latent diffusion: then the course's fourth family, autoregressive models, worked on a three-token toy."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
video_id: 2Sp0BqAWWXY
concepts: [score-matching, langevin, ddim, classifier-guidance, latent-diffusion, autoregressive-models]
sources:
  - tag: video
    label: "W9L35: DDPMs as score-predictors (video 2Sp0BqAWWXY)"
    url: https://www.youtube.com/watch?v=2Sp0BqAWWXY
  - tag: video
    label: "W9L38/L39: DDIMs and inference (video qiMJBB8chzI)"
    url: https://www.youtube.com/watch?v=qiMJBB8chzI
  - tag: video
    label: "W10L40: Auto-regressive models (video PtDFqdTbQUY)"
    url: https://www.youtube.com/watch?v=PtDFqdTbQUY
  - tag: paper
    label: "Song, Meng, Ermon, Denoising Diffusion Implicit Models (2020)"
    url: https://arxiv.org/abs/2010.02502
---

## The task: what did the denoiser really learn?

Lesson 9 trained ε_θ to predict added noise. The W9
lectures step back and ask what that means. Recall the
equivalence: predicting ε is predicting the **score**,
the gradient of the log-density:

```ascii
score(x_t) = grad log p(x_t) = -epsilon / sqrt(1 - a_bar_t)
```

The score is an arrow at every point, pointing toward
higher probability. In the Lesson 9 toy:
−0.688/0.4359 = −1.578. At x_2 = 3.9, the arrow points
left with strength 1.578: probability increases toward
smaller x. The denoiser learned a map of arrows covering
the whole noisy space. That reframes everything:
generation is hill-climbing on probability.

## First attempt: follow the arrows naively

If the score points uphill, sample by walking uphill:
start anywhere, take small steps along the score, add a
little noise so you explore instead of collapsing to the
peak. This is **Langevin dynamics**:

```ascii
x <- x + (delta/2) * score(x) + sqrt(delta) * z,   z ~ N(0,1)
```

Try it on the toy. x = 3.9, score = −1.578, δ = 0.1,
draw z = 0.4:

```ascii
x <- 3.9 + 0.05 * (-1.578) + 0.316 * 0.4
   = 3.9 - 0.079 + 0.126 = 3.947
```

The point moved from 3.9 to 3.947: uphill toward the
clean 4.0, with exploration noise. Repeat thousands of
times and the walk's positions follow p(x). This is the
older **score matching** literature's sampler, and the
lecture's point is that DDPM training already learns
exactly the object it needs. Two communities, one
arrow field.

Where naive Langevin breaks: it needs tiny steps and
thousands of them, and the step size δ is fiddly. Too
large: the walk diverges. Too small: it never arrives.
DDPM's ancestral sampler (Lesson 9) is the stabilized,
scheduled version of this walk.

## The key question

The sampler still costs T steps. Can we take bigger
strides without retraining?

## The new idea: DDIM strides

**DDIM** (Denoising Diffusion Implicit Models) observes
that training only ever used the marginals q(x_t|x_0):
the closed-form jump from Lesson 8. The Markov
step-by-step chain was one choice consistent with those
marginals. A non-Markovian process with the same
marginals trains the identical ε_θ. And with the
non-Markovian form, the reverse can skip steps: jump
from t = 100 to t = 50 in one stride, deterministically.

The stride formula: predict the clean x_0 from x_t via
ε_θ, then re-noise it to the target step s:

```ascii
x_hat_0 = ( x_t - sqrt(1 - a_bar_t) * e_theta ) / sqrt(a_bar_t)
x_s     = sqrt(a_bar_s) * x_hat_0 + sqrt(1 - a_bar_s) * e_theta
```

Work it. t = 100, s = 50, ᾱ_100 = 0.05, ᾱ_50 = 0.5,
x_100 = 1.5, ε_θ = 0.8:

```ascii
x_hat_0 = (1.5 - sqrt(0.95)*0.8) / sqrt(0.05)
        = (1.5 - 0.7798) / 0.2236 = 3.221
x_50 = sqrt(0.5)*3.221 + sqrt(0.5)*0.8
     = 2.278 + 0.566 = 2.844
```

One stride: 1.5 → 2.844, landing where 50 DDPM steps
would have gone. Fifty strides replace a thousand
steps. The price: the deterministic stride (no z
wobble) explores less, so diversity drops slightly.
re-adding controlled noise per stride recovers most of
it. Same network, faster sampler, no retraining.

## Steering and shrinking: guided and latent diffusion

Two more W9 ideas, each one mechanism.

**Guided diffusion** steers generation toward a target,
like "a cat". Train a classifier p(y|x_t) on noisy
images. Then sample with the conditional score:

```ascii
score_guided = score(x_t) + s * grad log p(y | x_t)
```

The extra term is an arrow pointing toward "more
cat-like". Toy numbers: score −1.578, classifier
gradient +0.5, guidance scale s = 2: guided score =
−1.578 + 1.0 = −0.578. The walk still climbs the data
probability but bends toward the requested class.
Larger s: stronger steering, weirder images. The
scale is the dial.

**Latent diffusion** shrinks the problem. Lesson 8's
price was full-size latents at every step. Instead,
compress the image with a VAE encoder (Lesson 7's
machinery), diffuse in the small latent space, decode
once at the end. A 512×512×3 image (786,432 numbers)
becomes a 64×64×4 latent (16,384 numbers): 48 times
smaller, so every diffusion step costs 48 times less.
This is the architecture behind Stable Diffusion.
The price is the VAE's bottleneck: compression
artifacts the diffusion cannot fix.

## The fourth family: autoregressive models

The course closes (W10) with the family that needs no
latent variables and no adversary: factor the joint
distribution with the chain rule and predict one piece
at a time.

```ascii
p(x_1, x_2, x_3) = p(x_1) * p(x_2 | x_1) * p(x_3 | x_1, x_2)
```

Each factor is a small classification problem: given
the past, what comes next? Train by maximum likelihood
(Lesson 2's identity): maximize the log-probability of
the true next piece. Generate left to right: sample
x_1, feed it in, sample x_2, and so on.

Work it on three tokens from {a, b}. Model says:
p(a) = 0.6. p(a|a) = 0.7, p(b|a) = 0.3.
p(a|a,b) = 0.2, p(b|a,b) = 0.8.

```ascii
p(a, b, a) = p(a) * p(b|a) * p(a|a,b)
           = 0.6 * 0.3 * 0.2 = 0.036
```

The sequence "aba" gets probability 0.036 under the
model. Training on real text pushes these factors
toward the true continuations. Generation samples
each factor in turn. This is next-token prediction,
the engine of every large language model. The
transformer architecture that makes it scale is
CS336/CS229S territory. This course contributes the
probabilistic frame: AR models are the chain-rule
family, trained by plain MLE, with exact likelihoods
and slow sequential sampling.

## The honest price, per shortcut

Langevin: principled but fiddly. Step size can
diverge the walk. DDPM's scheduled sampler is the
practical form. DDIM: 20 times fewer steps, slightly
less diversity, and the strides assume the marginals
match (they do, by construction). Guidance: needs a
classifier on noisy inputs and a hand-tuned scale.
too much guidance degrades quality. Latent
diffusion: 48 times cheaper per step, bottlenecked by
the autoencoder's fidelity. Autoregressive: exact
likelihood and simple training, but generation is
strictly sequential: token 1000 waits for 999 before
it. Every family pays somewhere.

## The course in one table

Four families, one recipe (family, divergence,
optimization), four prices:

| Family | Latent? | Objective | Generator gives | Price |
|---|---|---|---|---|
| GAN (L3-5) | None (noise only) | Minimax game on a variational bound | Samples, no density | Saddle point: saturation, mode collapse |
| VAE (L6-7) | Learned code z | ELBO: reconstruction − KL | Samples + density bound + editable code | Blurry: mode-covering + Gaussian decoder |
| Diffusion (L8-10) | Fixed noise chain | ELBO → noise MSE | Samples via T denoising steps | Slow sampling. Fixed schedule |
| Autoregressive (L10) | None (chain rule) | Exact MLE per piece | Samples + exact likelihood | Sequential generation |

The arc of the course: from counting what exists
(density estimation) to making what does not
(generation). GANs learn the boundary between real
and fake. VAEs learn a compressed imagination.
Diffusion learns to clean noise, one small step at a
time. Autoregressive models learn what comes next.
Four answers to "how do you teach a machine to
create?", each with its bill attached.

> [!QA]
> Q: What is the score, and why does it matter?
> A: The score is ∇log p(x): an arrow at each point toward higher probability. DDPM's noise prediction is a scaled score: −ε/√(1−ᾱ_t) = −1.578 in the toy. It matters because it unifies two literatures: DDPM training learns the object score matching always wanted, so score-based samplers like Langevin dynamics apply directly.
> Follow-up: What does one Langevin step look like?
> A: x ← x + (δ/2)·score + √δ·z. In the toy: 3.9 + 0.05·(−1.578) + 0.316·0.4 = 3.947. Uphill toward the clean 4.0 plus exploration noise. DDPM's sampler is this walk with a stabilizing schedule.

> [!QA]
> Q: How does DDIM sample faster without retraining?
> A: Training only used the marginals q(x_t|x_0), which a non-Markovian process shares. DDIM exploits that: predict x̂_0 from x_t, then jump straight to step s. In the toy, one stride took x_100 = 1.5 to x_50 = 2.844 via x̂_0 = 3.221. Fifty strides replace a thousand steps. The cost is slightly reduced diversity.
> Follow-up: Why does the deterministic stride lose diversity?
> A: DDPM's sampler adds fresh noise z at every step, exploring around the mean path. DDIM's basic stride has no z term: same x_T always gives the same x_0. Re-adding partial noise per stride recovers most of the variety.

> [!QA]
> Q: How does guided diffusion steer generation?
> A: Add a classifier's gradient to the score: score + s·∇log p(y|x_t). The extra arrow points toward the requested class y. In the toy, −1.578 + 2·0.5 = −0.578: still climbing data probability, bent toward the class. The scale s trades steering strength against image quality.
> Follow-up: What is latent diffusion buying?
> A: Speed. Diffuse in a VAE's compressed latent space instead of pixels: 786,432 numbers become 16,384 (48× smaller), so each step costs 48× less. The autoencoder's compression artifacts are the price.

> [!QA]
> Q: What is an autoregressive model?
> A: A model that factors the joint distribution with the chain rule and predicts one piece at a time: p(x_1,x_2,x_3) = p(x_1)·p(x_2|x_1)·p(x_3|x_1,x_2). In the toy, p(a,b,a) = 0.6·0.3·0.2 = 0.036. Train each factor by MLE, generate left to right. Exact likelihoods, simple training, strictly sequential sampling.
> Follow-up: Why does the course end here?
> A: Because it completes the four-family map: adversarial (GAN), learned-latent (VAE), fixed-latent (diffusion), no-latent (autoregressive). The transformer architecture that scales AR models belongs to CS336/CS229S. This course's contribution is the probabilistic frame all four share.

## Recap: the whole lesson on one screen

1. **The reframe.** ε_θ learns the score ∇log p(x_t): an arrow field pointing uphill on probability (−1.578 in the toy).
2. **Langevin.** Walk the arrows with noise: 3.9 → 3.947 in one step. DDPM's sampler is the scheduled version.
3. **The key question.** Can we stride instead of stepping, without retraining?
4. **DDIM.** Same marginals, non-Markovian reverse: x_100 = 1.5 → x_50 = 2.844 in one stride via x̂_0 = 3.221.
5. **Guidance.** score + s·∇log p(y|x): steer with a classifier's arrows (−0.578 in the toy).
6. **Latent.** Diffuse in VAE space: 48× cheaper per step, bottlenecked by the autoencoder.
7. **Autoregressive.** Chain rule, one piece at a time: p(a,b,a) = 0.036. Exact likelihood, sequential sampling.
8. **The course.** Four families, one recipe, four prices. Density estimation → generation.

## Official sources and further reading

**Official:**
- W9L35: DDPMs as score-predictors:
  https://www.youtube.com/watch?v=2Sp0BqAWWXY
- W9L36/L37: Guided diffusion, latent diffusion models.
- W9L38/L39: DDIMs and inference:
  https://www.youtube.com/watch?v=qiMJBB8chzI
- W10L40: Auto-regressive models:
  https://www.youtube.com/watch?v=PtDFqdTbQUY

**Further reading:**
- Song and Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (2019):
  https://arxiv.org/abs/1907.05600: the score view.
- Song, Meng, Ermon, "DDIM" (2020):
  https://arxiv.org/abs/2010.02502: the strides.
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2021):
  https://arxiv.org/abs/2112.10752: latent diffusion (Stable Diffusion).

**Caveats.** The W9L35 and W9L38 transcripts were not recovered (IDs from the playlist). This lesson follows the standard Song/Ermon and Rombach presentations, consistent with the lecture titles. [uncertain] The lecture's exact treatment and examples are unknown.

## Connections to the other courses

- **CS229 L11 (diffusion models):** that course's sampler is this lesson's DDIM with the same ε_θ. A worked tie: 50 DDIM strides on a 1000-step schedule evaluate the network 50 times instead of 1000, a 20× speedup, while the training loss never changes. Same weights, different walk.
- **CS336:** the autoregressive family is that course's home ground. The chain-rule factorization here is its next-token objective: p(token_t | tokens_<t). This lesson's 0.036 toy is the smallest possible language model.
