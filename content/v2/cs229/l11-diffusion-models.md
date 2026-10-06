---
page_id: cs229-l11
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 11
nav: "L11 · Diffusion Models"
title: "Lecture 11: Diffusion Models"
summary: "The predominant image generator: a fixed noising process, a learned denoiser, ELBO training, and sampling from pure noise."
date: "2026-05-11"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "1:18:01"
video_id: dqUMCzWjZSI
video_title: "Lecture 11: Diffusion Models"
video_caption: "Original lecture. Tengyu Ma builds diffusion models: the fixed forward process, the learned reverse denoiser, and ELBO training."
concepts: [diffusion-model, generative-model, forward-process, reverse-process, denoising, ELBO, VLA, GAN, VAE]
sources:
  - tag: video
    label: "Lecture 11 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=dqUMCzWjZSI
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: draw a cat that does not exist

Type "a cat astronaut on the moon" and get a photograph of one. No
such photo exists. The machine must invent pixels that look real.
This is **generative modeling** of images: learn p(x), the
distribution of natural images, then sample new x from it. Lectures
5 and 10 modeled distributions with Gaussians and mixtures. Images
have millions of pixels. Bell curves cannot touch them.

## First attempt: predict the pixels directly

The naive idea: train a network to output an image in one shot from
a random seed. This is roughly the GAN story the lecture contrasts
against. It works, but training is a fight: the generator and the
discriminator chase each other, modes collapse (the model draws the
same cat forever), and there is no clean likelihood to optimize.
The lecture's verdict by placement: the field moved to diffusion,
which trains on a stable, well-understood objective.

## The key question

What if generation were destruction in reverse? Take a real image
and destroy it gradually with noise until nothing remains. That
destruction is easy and needs no learning. Then learn to reverse
each tiny destruction step. To generate, start from pure noise and
run the reversals. The hard job (invent an image) becomes many easy
jobs (remove a little noise).

## The forward process: fixed destruction

Start with a clean image x_0. Each step shrinks it slightly and adds
a little Gaussian noise:

```ascii
x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon,  epsilon ~ N(0, I)
```

**Beta_t** is the noise schedule: small numbers like 10^-4 growing
to 10^-2. Each step keeps most of the image (sqrt(1-beta) is near 1)
and adds a whisper of noise. After one step the image is slightly
blurry. After T steps, with T in the hundreds or thousands, x_T is
pure static: the original is gone.

The crucial property: the forward process needs no learning. Given
x_0, you can sample x_t at any t directly in closed form (Gaussians
compose). So training data for the reverser is free: take real
images, noise them to any level, and you have (noisy, clean) pairs
at every noise level.

Toy with numbers. One pixel, x_0 = 0.8, beta_1 = 0.01, sampled
epsilon_1 = 0.5. x_1 = sqrt(0.99)*0.8 + sqrt(0.01)*0.5 = 0.796 +
0.05 = 0.846. Barely changed. After 100 such steps the pixel has
wandered far. After 1,000 it is standard Gaussian noise,
independent of the 0.8 it started from.

![Diffusion forward process](assets/svg/l11-diffusion.svg "The diffusion forward process. A clean image is destroyed step by step into pure noise. No learning: the destruction is fixed. Source: original plate for Stanford Frontier AI.")

## The reverse process: the learned denoiser

The **reverse process** learns p(x_{t-1} | x_t): given the noisy
image, predict one step cleaner. A neural network (usually a U-Net)
takes (x_t, t) and predicts the noise that was added, or
equivalently the cleaner x_{t-1}. Each single step is easy: the
noise added per step is tiny, so the reversal is a small correction.

Training is the ELBO from lecture 10, grown up. The latent
variables are the whole chain x_1 ... x_T. The ELBO decomposes into
a sum over steps, and each step's term is essentially: how well did
the network predict the noise at level t? In practice it simplifies
to a beautifully plain objective: mean squared error between the
true noise epsilon and the network's prediction. The lecture's
bottom line: diffusion trains by denoising score matching, which
walks and talks like a pile of regression problems, one per noise
level. Stable, no adversary, no mode collapse games.

## Sampling: noise to image

To generate: sample x_T from pure Gaussian noise. Run the learned
reverser T times: x_T -> x_{T-1} -> ... -> x_0. Each step removes a
little noise, guided by the network. Out comes an image. Condition
on text ("a cat astronaut") by feeding the text embedding into the
denoiser at every step, steering each correction toward the prompt.

## Why T is large

Why not destroy in 10 big steps instead of 1,000 small ones? Two
reasons. First, each reversal must be easy to learn. A tiny noise
step has a near-Gaussian reversal the network can fit. A giant leap
has a complex multimodal reversal it cannot. Second, the ELBO is
tight only when each step's reversal is close to the true posterior.
Small steps keep the approximation honest. The price is sampling
speed: generating one image needs T network evaluations. T = 1,000
means 1,000 forward passes per image. The field's whole
distillation and few-step sampler industry exists to pay this price
down.

## The honest price

Diffusion buys stable training and stunning samples, and pays in
sampling cost: hundreds to thousands of network evaluations per
image versus one for a GAN. It pays in likelihood too: the ELBO is
a bound, not the exact likelihood, so comparing models by ELBO is
comparing bounds. And it pays in data: the denoiser must see every
noise level of every kind of image, which takes enormous datasets
and compute. The lecture contrasts with GANs (fast sampling,
unstable training) and VAEs (clean latent story, blurrier samples):
diffusion won image generation by being the most trainable, not the
most elegant.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| Forward process | Learning to invent pixels directly is unstable | Fixed destruction: x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) eps; free (noisy, clean) pairs at every level |
| Reverse denoiser | One-shot generation is too hard a job | T easy jobs: predict one step's noise; pixel toy: 0.8 -> 0.846 in one step |
| ELBO training | GAN training fights; no clean objective | ELBO over the chain simplifies to noise-prediction MSE per level; stable regression |
| Large T | Big jumps have unlearnable reversals | Hundreds-thousands of tiny steps keep each reversal near-Gaussian and the bound tight |
| Text conditioning | Unconditional samples ignore the prompt | Feed text embedding into the denoiser every step; each correction steers toward the prompt |

> [!QA]
> Q: What is the forward process in a diffusion model?
> A: Fixed, learning-free destruction. Start from a clean image x_0 and iterate x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) epsilon with small betas (10^-4 to 10^-2). Each step barely changes the image. After T = hundreds to thousands of steps, x_T is pure Gaussian noise. Because Gaussians compose, you can sample x_t at any t directly from x_0 in closed form, so training pairs at every noise level are free.
> Follow-up: Why is the forward process fixed rather than learned?
> A: Because destruction needs no intelligence: adding noise is trivial. Fixing it makes the training data for the reverser free and the ELBO tractable. All learning concentrates in the reverse process, which is where the intelligence belongs. A learned forward process would add parameters with no benefit.

> [!QA]
> Q: What does the diffusion network actually learn to predict?
> A: The noise. Given a noisy image x_t and the timestep t, the network predicts the Gaussian noise epsilon that was added, or equivalently the slightly cleaner x_{t-1}. The training objective (from the ELBO) simplifies to mean squared error between true and predicted noise, averaged over timesteps and images. Each noise level is one regression problem. The network shares weights across all levels with t as input.
> Follow-up: Why predict the noise instead of the clean image directly?
> A: They are equivalent by algebra (x_0 can be recovered from x_t minus the noise), but noise prediction is better conditioned: the target epsilon is always standard Gaussian, same scale at every timestep, while x_0's scale varies. Stable targets train stably. The lecture presents noise prediction as the practical parameterization.

> [!QA]
> Q: Why does sampling need so many steps, and how is that being fixed?
> A: Each learned reversal only undoes a tiny noise step accurately. Big jumps have complex reversals the network cannot fit, and the ELBO bound loosens. So generation runs the network T times (hundreds to thousands) per image. The field attacks this with distillation (train a student to do in 4 steps what the teacher does in 1,000), better samplers (DDIM, DPM-Solver take larger principled steps), and noise schedule design. The price is fundamental to the method. The discounts are engineering.
> Follow-up: Compare diffusion with GANs and VAEs.
> A: GANs: one forward pass to sample (fast), but adversarial training is unstable and modes collapse. VAEs: clean probabilistic latent story, but samples are blurrier and likelihoods are bounds too. Diffusion: slowest to sample, but the most stable to train (plain regression objectives) and the best sample quality. The field chose trainability.

## Recap: the whole lesson on one screen

1. **The job.** Draw a cat astronaut that never existed. Learn
   p(x) for images, sample new x.
2. **First attempt.** One-shot generation (GAN-style): unstable
   training, mode collapse, no clean objective.
3. **The key question.** What if generation were destruction in
   reverse?
4. **Forward.** x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) eps.
   Fixed, free training pairs. Pixel toy: 0.8 -> 0.846.
5. **Reverse.** Network predicts each step's noise. ELBO becomes
   noise-prediction MSE per level. Stable.
6. **Sampling.** Pure noise x_T, run T reversals, steer each with
   the text prompt.
7. **Large T.** Tiny steps keep reversals learnable and the bound
   tight. Price: T network evals per image.
8. **The honest price.** Slow sampling, bound not likelihood, huge
   data. Won by trainability, not elegance.

## Official sources and further reading

**Official:**
- Lecture 11 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=dqUMCzWjZSI — Tengyu Ma derives
  the forward process, the ELBO training objective, and sampling,
  contrasting with GANs and VAEs.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the full
  ELBO derivation for diffusion.

**Caveats from these sources.** The lecture's beta schedule values
(10^-4 to 10^-2) and T in the hundreds-to-thousands are the
standard DDPM regime the lecture presents. The pixel toy is an
original miniature of the lecture's per-step arithmetic. Distillation
and few-step samplers are the field's ongoing answer to the
sampling price, surveyed but not derived in the lecture.

## Connections to the other courses

- **CS229 L10:** the ELBO, grown from one latent variable to a
  chain of T.
- **CS229 L05:** generative modeling's classical roots: from
  class-conditional Gaussians to learned denoisers.
- **CS229 L13:** text conditioning: the prompt embeddings that
  steer each denoising step.
- **CS229 L14:** the transformer backbones inside modern
  denoisers (DiT).
- **CS336:** training diffusion models at scale: the compute
  behind the samples.
