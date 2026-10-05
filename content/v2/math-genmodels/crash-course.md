---
page_id: math-genmodels-crash
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 901
nav: "Math · Generative Models · Crash course"
title: "Math-GenModels Crash Course"
summary: "Interview-speed review of Mathematical Foundations of Generative Models: one question, five machines, the judges — in 30 minutes, with links into the deep lessons."
---

<span class="crash-timer">30 minutes · interview speed</span>

This page tells the whole model-families story fast. Each section
gives you the working version: enough to answer interview
questions with confidence. Links at the end of each section take
you into the full lesson for the derivations and follow-ups.

<div class="crash-section" markdown="1">

### 1. The one question

Learn the rule behind samples, then draw fresh samples from it.
Counting needs one counter per outcome: 2^1024 for a thumbnail,
~10^473000 for a photo. Every family replaces the table with a
smooth function. Five machines: the storyteller (steps), the
sculptor (hidden causes), the warper (invertible bends), the
restorer (denoising), the critic (energy valleys).
[Lecture 1](l01-the-one-question.html)

</div>

<div class="crash-section" markdown="1">

### 2. The storyteller: chain rule and teacher forcing

Factor the sequence: 0.80 x 1.00 x 0.75 = 0.60, an identity, not
an approximation. Train on true histories (teacher forcing):
loss 1.155 nats, perplexity 1.47 on the toy. The causal mask
makes attention autoregressive: "helped" becomes [0.50, 0.50].
Two prices: exposure bias (only 0.9^20 = 12% of 20-token samples
stay fully on track) and serial sampling (1,000 tokens at 20 ms
= 20 seconds).
[Lecture 2](l02-autoregressive-models.html)

</div>

<div class="crash-section" markdown="1">

### 3. The sculptor: VAEs and the reparameterization trick

Encoder outputs a distribution (mu, sigma), not a point. ELBO =
reconstruction minus KL rent: 2.318 nats at mu = 2.0, sigma =
0.5. The trick: z = mu + sigma x epsilon moves the dice roll out
of the parameter path (z = 2.3, dz/dmu = 1). Sample z ~ N(0,1)
and decode. Price: blur, because the KL leash crowds codes and
the decoder averages. The encoder is an amortized E-step.
[Lecture 3](l03-variational-autoencoders.html)

</div>

<div class="crash-section" markdown="1">

### 4. The warper: exact densities via change of variables

p_x = p_z / |det J|. Stretch by 2, density halves (0.5 on
[1,3]). Folds need branch sums, so invertibility is mandatory.
General determinants cost O(d^3). Coupling layers make the
Jacobian triangular so the determinant is a diagonal product
(log det = 0.693 on the toy). Prices: rigid architectures, no
compression, no residual blocks.
[Lecture 4](l04-normalizing-flows.html)

</div>

<div class="crash-section" markdown="1">

### 5. The restorer: DDPM forward and reverse

Fixed destruction: 0.8 -> 0.846 -> 0.80177, signal gone after
1,000 steps (0.99^1000 = 0.000043). Closed-form jumps give free
training pairs at every noise level. The network predicts the
added noise: loss (0.5 - 0.42)^2 = 0.0064, plain regression, no
adversary. Sampling runs 1,000 reverse steps from pure noise.
Price: 50 seconds per image at 50 ms per step.
[Lecture 5](l05-diffusion-ddpm.html)

</div>

<div class="crash-section" markdown="1">

### 6. The slope reader: score matching and shortcuts

Score s(x) = grad log p(x) = -x for N(0,1). Learn it by
denoising: target (x - x-tilde)/sigma^2 = -1.2 on the toy, no
density estimated. Sample with Langevin: 2.3 -> 2.059. DDPM's
noise predictor IS a score predictor: s = -0.42/0.1 = -4.2.
DDIM reuses the denoiser for 50 jumps instead of 1,000 steps.
Guidance: epsilon_u + w(epsilon_c - epsilon_u) = 0.96 at w = 3.
Price: no likelihoods. Step sizes must stay small.
[Lecture 6](l06-score-matching-ddim.html)

</div>

<div class="crash-section" markdown="1">

### 7. The critic: energies without normalization

p(x) = exp(-E(x))/Z. Toy Z = 1.8711. Z over images is
intractable (~10^290 years), so train by contrast: lower data
energy, raise fantasy energy. CD-1: one MCMC step from data,
theta [0,0,0,0] -> [0,-0.1,0,0.1]. Prices: biased gradients, no
normalized probabilities, chains stuck in single valleys.
Superpower: energies compose by addition.
[Lecture 7](l07-energy-based-models.html)

</div>

<div class="crash-section" markdown="1">

### 8. The judges

Likelihood: exact for storyteller and warper. Blurry beats Sharp
by infinity on coverage. FID: Frechet distance of Inception
feature Gaussians. 0.5 for a mean shift, 1.5 for doubled spread.
Blind to memorization (scores 0) and to shape (spike mixture
scores 0.005). Inception Score: a 1,000-photo memory machine
scores 1000. Real answer: a portfolio, with precision/recall
(0.8/0.5 on the toy, 1.0/0.1 under mode collapse) for diagnosis.
[Lecture 8](l08-judging-a-creator.html)

</div>
