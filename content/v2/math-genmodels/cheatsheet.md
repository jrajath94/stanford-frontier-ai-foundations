---
page_id: math-genmodels-cheatsheet
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 900
nav: "Math · Generative Models · Cheatsheet"
title: "Math-GenModels Cheatsheet"
summary: "Every key fact from Mathematical Foundations of Generative Models on one dense page: the five machines, their formulas, their worked numbers, their prices, and the judges."
---

The whole course as short stories. Each block tells one idea the
way the lesson tells it: the problem, the number, the fix. Follow
the links for the full derivations.

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### The one question

Learn the rule behind samples, then draw fresh samples from it.
The naive counting machine needs one counter per outcome: 2^1024
for a 32x32 binary image, ~10^473000 for a color photo. Every
family replaces the table with a smooth function so similar
outcomes share learning. [Lecture 1](l01-the-one-question.html)

</div>

<div class="cheat-block" markdown="1">

### The storyteller (autoregressive)

P(x1..xL) = product of P(xt | x<t). Toy: 0.80 x 1.00 x 0.75 =
0.60, exact. Teacher forcing feeds true histories. Toy loss 1.155
nats, perplexity 1.47. Causal mask on the attention toy: "helped"
becomes [0.50, 0.50], "frame" stays [0.79, 0.79]. Prices:
exposure bias (0.9^20 = 0.12 clean samples) and serial sampling
(1,000 tokens x 20 ms = 20 s). [Lecture 2](l02-autoregressive-models.html)

</div>

<div class="cheat-block" markdown="1">

### The sculptor (VAE)

Encoder outputs mu, sigma. ELBO = E[log p(x|z)] -
KL(q(z|x)||p(z)). Toy KL at mu = 2.0, sigma = 0.5: 2.318 nats of
rent. Reparameterization: z = mu + sigma x epsilon. With epsilon
= 0.6, z = 2.3, dz/dmu = 1, dz/dsigma = 0.6. Sample: z ~ N(0,1),
decode. Price: blur. Codes shrink, z = 0 decodes to the average.
The encoder is an amortized E-step. [Lecture 3](l03-variational-autoencoders.html)

</div>

<div class="cheat-block" markdown="1">

### The warper (flows)

p_x(x) = p_z(z) / |det J|. Toy 1-D: x = 2z+1 gives p_x = 0.5 on
[1,3]. Folds break it: x = z^2 needs a branch sum. General
determinants cost O(d^3): 10^9 ops at d = 1,000. Coupling layer:
x_b = z_b x s(z_a) + t(z_a), triangular Jacobian, det = product
of diagonal. Toy: [0.5,1.0] -> [0.5,3.0], log det = 0.693.
Price: rigidity. No compression. Cannot use residual blocks.
[Lecture 4](l04-normalizing-flows.html)

</div>

<div class="cheat-block" markdown="1">

### The restorer (DDPM)

Forward: x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) epsilon,
fixed. Toy: 0.8 -> 0.846 -> 0.80177. Signal 0.99^1000 =
0.000043. Closed form: x_t = sqrt(alpha-bar_t) x_0 +
sqrt(1-alpha-bar_t) epsilon. Reverse: predict the noise. Toy loss
(0.5-0.42)^2 = 0.0064. Sample: 1,000 reverse steps from N(0,1).
Price: 1,000 x 50 ms = 50 s per image. [Lecture 5](l05-diffusion-ddpm.html)

</div>

<div class="cheat-block" markdown="1">

### The slope reader (score)

s(x) = grad log p(x). For N(0,1), s(x) = -x. Density-first
breaks: a 30% wobble turns true score -3 into -1.793.
Denoising target: (x - x-tilde)/sigma^2. Toy: x-tilde = 2.3,
target -1.2. Langevin: 2.3 -> 2.059 at delta = 0.1. Delta = 1.0
overshoots to 1.15. DDPM noise IS score: s = -0.42/0.1 = -4.2.
DDIM: 1,000 -> 50 steps. Guidance: 0.42 + 3 x 0.18 = 0.96.
Price: no likelihoods. [Lecture 6](l06-score-matching-ddim.html)

</div>

<div class="cheat-block" markdown="1">

### The critic (EBM)

p(x) = exp(-E(x))/Z. Toy: E = [2,1,1,0], Z = 1.8711, p =
[0.0723, 0.1966, 0.1966, 0.5345]. Z over images: ~10^290 years.
Gradient = data term minus model term. CD-1: theta [0,0,0,0] ->
[0,-0.1,0,0.1], mass fantasy -> data (0.2256 vs 0.2756). Score =
-grad E. Prices: biased gradients, no normalized probabilities,
chains stuck in one valley. [Lecture 7](l07-energy-based-models.html)

</div>

<div class="cheat-block" markdown="1">

### The judges

Likelihood: log 0.6 = -0.511 beats log 0.4 = -0.916. Blind spot:
Blurry (-0.693) beats Sharp (-infinity). FID: ||mu_r -
mu_g||^2 + Tr(...). Toy 0.5 for mean shift, 1.5 for doubled
spread. Blind spots: memorization scores 0, spike mixture scores
0.005. IS: 1,000-photo memory machine scores 1000. Portfolio:
precision/recall. Toy 0.8/0.5, mode collapse 1.0/0.1.
[Lecture 8](l08-judging-a-creator.html)

</div>

<div class="cheat-block" markdown="1">

### Interview lines

"Teacher forcing trains on true histories, so sampling meets
guesses it never saw. 0.9^20 = 12% stay clean." "The
reparameterization trick moves randomness to epsilon so dz/dmu =
1." "Flows are exact because the Jacobian determinant is the
volume exchange rate. Triangular makes it O(d)." "DDPM predicts
noise, which is the score divided by sqrt(1-alpha-bar_t)."
"CD trains energies without Z by contrasting data against
one-step fantasies." "FID 0.005 on a spike mixture: same
mean/variance, totally different shape."

</div>

</div>
