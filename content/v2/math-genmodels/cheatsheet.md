---
page_id: math-genmodels-cheatsheet
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 900
nav: "Math · Generative Models · Cheatsheet"
title: "Math-GenModels Cheatsheet"
summary: "Every key fact from Mathematical Foundations of Generative Models on one dense page: the five machines, their formulas, their worked numbers, their prices, the judges, memory aids, and decision rules."
---

The whole course as short stories. Each block tells one idea the
way the lesson tells it: the problem, the number, the fix. Follow
the links for the full derivations. The memory aids at the end
are the exam kit: mnemonics, never-confuse pairs, and
if-this-then-that rules.

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### The one question

Learn the rule behind samples, then draw fresh samples from it.
The naive counting machine needs one counter per outcome: 2^1024
for a 32x32 binary image, ~10^473000 for a color photo. Every
family replaces the table with a smooth function so similar
outcomes share learning. Counting is maximum likelihood: the
counts are the rule that makes the data most probable.
[L01](l01-the-one-question.html)

</div>

<div class="cheat-block" markdown="1">

### The storyteller (autoregressive)

P(x1..xL) = product of P(xt | x<t). Toy: 0.80 x 1.00 x 0.75 =
0.60, exact. Teacher forcing feeds true histories: toy loss 1.155
nats, perplexity 1.47 = exp(1.155/3). Causal mask on the
attention toy: "helped" becomes [0.50, 0.50], "frame" stays [0.79,
0.79]. Decoding: T = 0.5 sharpens "cat" to 0.864, T = 2.0
flattens to 0.502. Prices: exposure bias (0.9^20 = 0.12 clean
samples) and serial sampling (1,000 tokens x 20 ms = 20 s). KV
cache cuts per-step cost, not the serial wait.
[L02](l02-autoregressive-models.html)

</div>

<div class="cheat-block" markdown="1">

### The sculptor (VAE)

Encoder outputs mu, sigma. ELBO = E[log p(x|z)] -
KL(q(z|x)||p(z)). Toy KL at mu = 2.0, sigma = 0.5: 2.318 nats of
rent. Reparameterization: z = mu + sigma x epsilon. With epsilon
= 0.6, z = 2.3, dz/dmu = 1, dz/dsigma = 0.6. Sample: z ~ N(0,1),
decode. Price: blur. Codes shrink, z = 0 decodes to the average.
Posterior collapse: KL = 0, decoder ignores z. beta-VAE: beta <
1 sharpens, holes return. VQ-VAE: discrete codebook, 8,192 codes
x 13 bits, the storyteller's tokenizer. The encoder is an
amortized E-step. [L03](l03-variational-autoencoders.html)

</div>

<div class="cheat-block" markdown="1">

### The warper (flows)

p_x(x) = p_z(z) / |det J|. Toy 1-D: x = 2z+1 gives p_x = 0.5 on
[1,3]. Folds break it: x = z^2 needs a branch sum. General
determinants cost O(d^3): 10^9 ops at d = 1,000. Coupling layer:
x_b = z_b x s(z_a) + t(z_a), triangular Jacobian, det = product
of diagonal. Toy: [0.5,1.0] -> [0.5,3.0], log det = 0.693.
MAF: fast density, serial sampling. IAF: the mirror. Dequantize:
pixel 123 -> 123.4. Price: rigidity. No compression. Cannot use
residual blocks. [L04](l04-normalizing-flows.html)

</div>

<div class="cheat-block" markdown="1">

### The restorer (DDPM)

Forward: x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) epsilon,
fixed. Toy: 0.8 -> 0.846 -> 0.80177. Signal 0.99^1000 =
0.000043. Schedule: beta 1e-4 -> 2e-2. Alpha-bar 0.90 at t =
100, 0.08 at t = 500, 4e-5 at t = 1000. Closed form: x_t =
sqrt(alpha-bar_t) x_0 + sqrt(1-alpha-bar_t) epsilon. Reverse:
predict the noise. Toy loss (0.5-0.42)^2 = 0.0064. Targets:
epsilon (stable), x_0, v. Latent diffusion: 512x512x3 ->
64x64x4, 48x cheaper. Sample: 1,000 reverse steps from N(0,1).
Price: 1,000 x 50 ms = 50 s per image.
[L05](l05-diffusion-ddpm.html)

</div>

<div class="cheat-block" markdown="1">

### The slope reader (score)

s(x) = grad log p(x). For N(0,1), s(x) = -x. Density-first
breaks: a 30% wobble turns true score -3 into -1.793.
Denoising target: (x - x-tilde)/sigma^2. Toy: x-tilde = 2.3,
target -1.2. Langevin: 2.3 -> 2.059 at delta = 0.1. Delta = 1.0
overshoots to 1.15. DDPM noise IS score: s = -0.42/0.1 = -4.2.
SDE trio: forward SDE, reverse SDE, probability-flow ODE, one
score. DDIM: 1,000 -> 50 jumps, eta = 0 deterministic.
Guidance: 0.42 + 3 x 0.18 = 0.96, w ~ 7 in production. Flow
matching: x_t = (1-t) x_0 + t eps, velocity targets, SD3.
Price: no likelihoods. [L06](l06-score-matching-ddim.html)

</div>

<div class="cheat-block" markdown="1">

### The critic (EBM)

p(x) = exp(-E(x))/Z. Toy: E = [2,1,1,0], Z = 1.8711, p =
[0.0723, 0.1966, 0.1966, 0.5345]. Z over images: ~10^290 years.
Gradient = data term minus model term. CD-1: theta [0,0,0,0] ->
[0,-0.1,0,0.1], mass fantasy -> data (0.2256 vs 0.2756).
Persistent CD: replay buffer, chains survive across batches.
Score = -grad E. Score matching trains E without Z
(differentiation kills constants). Superpower: energies add.
E_face + E_smile = [2,4] -> p = [0.881, 0.119]. Prices: biased
gradients, no normalized probabilities, chains stuck in one
valley. [L07](l07-energy-based-models.html)

</div>

<div class="cheat-block" markdown="1">

### The judges

Likelihood: log 0.6 = -0.511 beats log 0.4 = -0.916. Blind spot:
Blurry (-0.693) beats Sharp (-infinity). Bits per dimension is
the fair scale. FID: ||mu_r - mu_g||^2 + Tr(...). Toy 0.5 for
mean shift, 1.5 for doubled spread. Blind spots: memorization
scores 0, spike mixture scores 0.005. KID: unbiased, O(n^2).
IS: 1,000-photo memory machine scores 1000. Catch the parrot:
nearest-neighbor distances ~0, train/test FID split. Portfolio:
precision/recall. Toy 0.8/0.5, mode collapse 1.0/0.1. Density
and coverage refine against gaming. [L08](l08-judging-a-creator.html)

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
mean/variance, totally different shape." "Energies add: the
critic composes, nobody else does."

</div>

</div>

## Memory aids

### Mnemonics

- **The five machines:** "Stories Sculpt Warps, Restorers Criticize." Storyteller, sculptor, warper, restorer, critic. In course order.
- **The five prices:** "Serial, Blurry, Rigid, Slow, Blind." Storyteller samples serially. Sculptor blurs. Warper is rigid. Restorer is slow. Critic is blind to probabilities.
- **Exact-density families:** "Stories Warp exactly." Only the storyteller and the warper give exact likelihoods. The sculptor and restorer give bounds. The critic gives energies.
- **DDIM vs DDPM:** "DDIM Jumps, DDPM Walks." Same network. Jumps skip, walks step.
- **Guidance:** "Push past the conditional." eps-tilde = eps_u + w(eps_c - eps_u). w pushes past.

### Never-confuse pairs

| Pair | Difference |
|---|---|
| Chain rule vs teacher forcing | The chain rule is an identity (exact factorization). Teacher forcing is a training procedure (feed true pasts). |
| ELBO vs likelihood | The ELBO lower-bounds the likelihood. The gap is the encoder's error. Maximizing the bound is not maximizing the truth. |
| Score vs energy | The score is -grad E: the slope. The energy is the surface. Score models keep the slope. EBMs keep the surface. |
| DDPM vs DDIM | DDPM: 1,000 stochastic steps, the training-time process. DDIM: few deterministic jumps, same trained network. |
| FID vs KID | FID: biased estimator, O(n), the leaderboard standard. KID: unbiased MMD, O(n^2), for small samples. |
| Precision vs recall | Precision: are my samples good (near real)? Recall: did I cover the real modes? Collapse = precision 1.0, recall 0.1. |
| MAF vs IAF | MAF: fast density, serial sampling. IAF: fast sampling, serial density. Coupling: both fast, weaker layers. |
| KL rent vs reconstruction | Rent pulls codes to the prior (blur, holes filled). Reconstruction pulls codes apart (sharp, holes open). Every VAE lives in this tension. |
| Temperature vs top-p | Temperature reshapes the distribution (mood). Top-p cuts the tail (sanity guardrail). |
| Forward SDE vs reverse SDE vs ODE | Forward: fixed destruction. Reverse SDE: stochastic restoration, needs the score. ODE: deterministic restoration, same score. |

### If-this-then-that rules

- If the data has a natural order (text, audio) -> reach for the storyteller.
- If you need exact p(x) per point (anomaly detection) -> reach for the warper (MAF: density is its fast direction).
- If you need the best samples and can afford slow sampling -> reach for the restorer (latent diffusion + DDIM + guidance).
- If models must compose without retraining -> reach for the critic (energies add).
- If sampling is too slow -> DDIM jumps first (no retraining), then distillation.
- If the prompt is ignored -> raise guidance w before touching the sampler.
- If the latent does nothing (collapse) -> check KL per axis. Anneal KL or weaken the decoder.
- If FID improves but users complain -> check memorization (nearest-neighbor), then precision vs recall.
- If comparing likelihoods -> use bits per dimension, same dataset, same dequantization, and say bound vs exact.
- If the schedule misbehaves at the ends -> switch epsilon-prediction to v-prediction.

### The one-glance table

| Family | Learns | Samples | Density | Price | Production example |
|---|---|---|---|---|---|
| Storyteller | Conditionals P(xt\|x<t) | Token by token | Exact | Serial sampling | GPT-4, Claude, WaveNet |
| Sculptor | Encoder + decoder, ELBO | z ~ N(0,1), decode | Bound | Blur | SD's VAE, VITS |
| Warper | Invertible warp | z ~ N(0,I), warp | Exact | Rigidity | WaveGlow |
| Restorer | Denoiser, MSE on noise | 1,000 reverse steps | Bound | Slow sampling | Stable Diffusion, Sora |
| Critic | Energy via CD | Langevin downhill | None (unnormalized) | Bias, no Z, stuck chains | Research only |
| GAN (6th) | Generator vs discriminator | One forward pass | None | Unstable training | StyleGAN, Real-ESRGAN |
