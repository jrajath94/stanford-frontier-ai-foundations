---
page_id: math-genmodels-l06
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 6
nav: "L06 · Score Matching & Fast Diffusion"
title: "Lecture 6: Score Matching and Diffusion II: Reading the Slope"
summary: "A new lens on the restorer: learn the slope of the log-density (the score) by denoising, then walk uphill with Langevin dynamics. DDPM's noise predictor is a score predictor. DDIM shortcuts the 1,000 steps. Guidance steers them."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
video_id: 2Sp0BqAWWXY
video_title: "W9L35: DDPMs as score-predictors (IIT Madras)"
video_caption: "The lecture this chapter follows. The punchline: DDPM predicts scores."
concepts: [score-matching, score-function, denoising-score-matching, langevin-dynamics, ddim, classifier-free-guidance, sde, flow-matching]
sources:
  - tag: video
    label: "W9L35: DDPMs as score-predictors (video 2Sp0BqAWWXY)"
    url: https://www.youtube.com/watch?v=2Sp0BqAWWXY
  - tag: video
    label: "W9L38: DDIMs (video qiMJBB8chzI)"
    url: https://www.youtube.com/watch?v=qiMJBB8chzI
  - tag: paper
    label: "Song & Ermon, Generative Modeling by Estimating Gradients of the Data Distribution (2019)"
    url: https://arxiv.org/abs/1907.05600
---

## The question, through a new lens

Restate the one question: learn the rule behind samples, draw
fresh samples from it. L05's restorer learns to remove noise step
by step. This lesson re-reads the same machine through a new
lens: at every noise level, the denoiser points uphill on the
probability surface. Learn the slope, walk uphill, and you
sample. The slope has a name: the **score**, the gradient of the
log-density, s(x) = grad log p(x). It points toward higher
probability, with length equal to the steepness.

Work it on the bell curve p(x) = N(0,1). log p(x) = -x^2/2 minus
a constant. Differentiate: s(x) = -x. At x = 2.3, the score is
-2.3: pointing left, back toward the mean, with strength 2.3. At
x = 0 the score is 0: the peak, no slope. The score is a vector
field that always says "higher probability is that way."

![The score always points uphill](assets/plate-l06-score-field.webp "s(x) = -x for N(0,1). At x = 2.3 the slope is -2.3, back toward the mean. Shell 2. Source: original toy. Project: Stanford Frontier AI.")

## First attempt: differentiate a learned density

The naive route: learn the density p(x) from samples, then
differentiate it to get the score. Watch it fail. True
distribution N(0,1), true score s(x) = -x. Suppose the learned
density picks up a small spurious wobble from finite samples:

```ascii
p-hat(x) ~ exp(-x^2/2) x (1 + 0.3 sin(4x))
score-hat(x) = -x + 1.2 cos(4x) / (1 + 0.3 sin(4x))
```

At x = 3, the true score is -3. The wobble term is 1.2 x
cos(12) / (1 + 0.3 x sin(12)) = 1.2 x 0.8439 / (1 - 0.161) =
1.207. The estimated score is -3 + 1.207 = -1.793, off by 40%.
A 30% wiggle in the density became a 40% error in the score.
Differentiation amplifies every bump the density estimate got
wrong.

The failure concentrates where it hurts most. Far from the data
(x = 3 has few samples), the density estimate is worst, so the
score estimate is worst exactly where a sampler needs guidance
to find its way back. Learning the density first is the wrong
order: the score is what sampling needs, and it deserves its own
objective.

## The key question

What if we learn the score directly, without ever learning the
density, by asking the machine to denoise?

## The new idea: denoising score matching

Corrupt a clean sample: x-tilde = x + sigma x epsilon. Train a
network s_theta(x-tilde) to predict (x - x-tilde)/sigma^2, the
direction from the corrupted point back to the clean one, scaled.
That target is minus the noise over the noise variance: -epsilon
/ sigma.

Work it. Clean x = 2.0, sigma = 0.5, drawn epsilon = 0.6:

```ascii
x-tilde = 2.0 + 0.5 x 0.6 = 2.3
target  = (2.0 - 2.3) / 0.25 = -1.2
```

One training target: -1.2. Averaged over many corruptions of many
clean samples, this target equals the score of the
noise-perturbed distribution, grad log q_sigma(x-tilde). As sigma
shrinks, q_sigma approaches the true p, and the learned score
approaches the true score. No density is ever estimated. The
wobble problem vanishes because there is no density to wobble.

Noise corruption also fixes the empty-region problem. The
corrupted samples blanket the space around the data, so the
network trains on points far from any clean sample and learns
sensible return directions everywhere. The score field is
populated by construction.

## Sampling: walk uphill with noise

Given the score, sample with **Langevin dynamics**: follow the
score uphill, plus a dash of noise to explore:

```ascii
x <- x + (delta/2) s_theta(x) + sqrt(delta) z,   z ~ N(0,1)
```

Work two steps at x = 2.3 with learned score -2.3 and delta =
0.1. Draw z = -0.4:

```ascii
x <- 2.3 + 0.05 x (-2.3) + 0.316 x (-0.4)
   = 2.3 - 0.115 - 0.126 = 2.059
```

The score term pulls toward the mean. The noise term jitters.
Repeated, the chain wanders but lingers where the score is
small, which is where probability is high. Run it at decreasing
noise levels (annealed Langevin): coarse scores first, fine
scores later. That annealing schedule is exactly DDPM's reverse
chain from L05, wearing different notation.

### Subchapter: the forward SDE: destruction in continuous time

Discrete steps become a continuous story. The **forward SDE**
(stochastic differential equation) destroys data in continuous
time: dx = f(x,t) dt + g(t) dw. The drift f(x,t) steers the
destruction. The noise g(t) dw shakes it. L05's 1,000 discrete
steps are one discretization of this equation.

### Subchapter: the reverse SDE: Anderson's theorem

**Anderson's theorem** (1982) says reversing time in a diffusion
gives another diffusion. The reverse drift is the forward drift
minus the noise squared times the score, so the reverse run needs
exactly one learned object: the score s(x,t). This is the
**reverse SDE**: it runs backward in time, from noise to data,
driven by the learned score.

### Subchapter: the probability-flow ODE: drop the noise

Drop the randomness from the reverse SDE and the same score
drives a deterministic path from noise to data: the
**probability-flow ODE**. It keeps the same per-time marginals
as the SDE, so the score learned by denoising still applies, but
every run is reproducible: the same starting noise gives the same
sample.

![One score drives three machines](assets/plate-l06-sde.webp "Forward SDE destroys, reverse SDE restores, the probability-flow ODE walks straight. Shell 4. Source: original (Song et al., 2021). Project: Stanford Frontier AI.")

Three machines, one score. DDPM is the reverse SDE discretized
into 1,000 steps. DDIM (below) is the probability-flow ODE
discretized into jumps. The score view is what unifies them:
learn s(x,t) once, then choose your sampler. (Song et al., 2021,
arxiv 2011.13456, carries the full unification. Here the working
fact is the trio and the shared score.)

## The lecture's punchline: DDPM predicts scores

W9L35 reframes L05's denoiser: predicting the noise epsilon is
predicting the score. The relation is one division:

```ascii
s(x_t) = - epsilon-hat(x_t, t) / sqrt(1 - alpha-bar_t)
```

Work it on L05's toy. At t = 1, alpha-bar_1 = 0.99, so
sqrt(1 - 0.99) = 0.1. The toy's
predicted noise was epsilon-hat = 0.42:

```ascii
s(x_1) = -0.42 / 0.1 = -4.2
```

The denoiser that predicted noise 0.42 at x_1 = 0.846 was really
saying "the log-density slopes downward with steepness 4.2
here." DDPM training (MSE on noise) is denoising score matching
at every noise level simultaneously. Two lectures, one machine.

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/B4oHJpEJBAA" title="Diffusion Models From Scratch: Score-Based Generative Models Explained" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Explainer: Diffusion Models From Scratch, score-based generative models. Score, score matching, denoising score matching, sampling, and the link to DDPM, DDIM, and EDM (EDM: Elucidating the Design Space of diffusion models, Karras et al.). Watch after the DDPM-predicts-scores section.</p></div>

## Mapping back: what each property fixes

| Density-first failure | Score-matching answer | How |
|---|---|---|
| Differentiation amplifies density wobbles: -3 estimated as -1.793 | Learn the score directly | Denoising target. No density estimated, nothing to differentiate |
| Empty regions have the worst scores | Corrupt with noise during training | Corrupted samples blanket the space. Return directions learned everywhere |
| One noise level gives one blurred view | Many noise levels (annealed) | Coarse-to-fine schedule. Equals DDPM's reverse chain |

## The honest price, and two ways to pay less

**Price 1: no likelihoods.** The score discards the normalizing
constant: grad log p(x) = grad p(x)/p(x) never sees the
partition function. A score model samples beautifully and cannot
tell you p(x). For anomaly detection or compression, which need
densities, the warper (L04) keeps its exactness crown.

**Price 2: step-size fragility, demonstrated.** Langevin with too
large a delta overshoots. At x = 2.3, score -2.3, delta = 1.0,
z = 0:

```ascii
x <- 2.3 + 0.5 x (-2.3) + 0 = 1.15
```

One step jumped from 2.3 to 1.15, past the region the score was
estimated for. Repeated overshooting oscillates or diverges.
Delta = 0.1 crept safely to 2.059. The sampler needs small steps
and many of them: the serial bill again.

### Subchapter: DDIM derived: skip steps, keep the denoiser

L05's sampler needs all 1,000 steps because each reverse step
was derived for the full Markov chain. DDIM (Song, Meng and
Ermon, 2020) generalizes the forward process to a non-Markovian
one with the same per-step marginals, which frees the sampler
to jump. The update at each jump: use epsilon-hat to estimate
x_0, then re-noise to the target level:

```ascii
x_0-hat = (x_t - sqrt(1 - alpha-bar_t) eps-hat) / sqrt(alpha-bar_t)
x_s     = sqrt(alpha-bar_s) x_0-hat + sqrt(1 - alpha-bar_s) eps-hat
```

No new noise is added (eta = 0): the jumps are deterministic.
Fifty jumps (t = 1000 -> 980 -> ... or 1000 -> 800 -> 600)
replace a thousand steps: a 20x speedup from the same trained
network, with a small quality cost. The plate shows the shape:
ten dots become six jumps.

![DDIM: 50 jumps replace 1000 steps](assets/plate-l06-ddim.webp "Same trained denoiser. Jump t = 1000 -> 800 -> 600: estimate x_0, re-noise, repeat. Shell 3. Source: original toy (Song, Meng and Ermon, 2020). Project: Stanford Frontier AI.")

The decision rule: DDIM is the first tool whenever sampling is
too slow, because it needs no retraining. Eta = 0 gives
deterministic, reproducible samples (same x_T, same image).
Eta = 1 recovers DDPM-like stochasticity. Between them, eta
trades diversity for determinism.

### Subchapter: classifier guidance vs classifier-free guidance

To steer sampling toward a condition (a text prompt, a class),
add the condition's score to the mix. **Classifier guidance**
trains a separate classifier on noisy images and adds w times
grad log p(y|x_t) to the score. It works, but it needs a
noise-conditional classifier trained at every noise level: a whole
second model.

**Classifier-free guidance** (Ho and Salimans, 2022) skips the
classifier. Train the denoiser twice in one: conditional
epsilon_c and unconditional epsilon_u (trained by randomly
dropping the condition). Sample with the extrapolated noise:

```ascii
epsilon-tilde = epsilon_u + w (epsilon_c - epsilon_u)
```

![Guidance extrapolates toward the condition](assets/plate-l06-guidance.webp "eps_u = 0.42, eps_c = 0.60, w = 3: eps-tilde = 0.42 + 3 x 0.18 = 0.96. Shell 3. Source: original toy (Ho and Salimans, 2022). Project: Stanford Frontier AI.")

Work it: epsilon_u = 0.42, epsilon_c = 0.60, w = 3 gives
epsilon-tilde = 0.42 + 3 x 0.18 = 0.96. The guidance scale w
pushes the sample past the conditional prediction, toward
stronger prompt match. Higher w means stronger match but
shrinking diversity and, past a point, artifacts (oversaturated
colors, melted textures). The decision rule: w around 7 is the
production default for text-to-image. Raise it for prompt
fidelity, lower it for variety. Classifier-free won because it
needs no second model: the conditional model guides itself.

### Subchapter: flow matching: the straight-line cousin

Diffusion paths curve: the forward process adds noise gradually,
and the reverse path bends. **Flow matching** (Lipman et al.,
2022) draws straight lines instead: x_t = (1 - t) x_0 + t eps,
and trains the network to predict the **velocity** v = eps -
x_0, the straight-line direction. The loss is the same plain
regression. The target is simpler.

![Flow matching: straight lines from noise to data](assets/plate-l06-flow.webp "x_t = (1 - t) x_0 + t eps. The target is the velocity, not the noise. Shell 3. Source: original (Lipman et al., 2022). Project: Stanford Frontier AI.")

Straight paths need fewer steps to integrate: this is why
**Stable Diffusion 3** trains on rectified flow (Esser et al.,
2024). The probability-flow ODE from the subchapters above is the bridge: flow
matching learns its velocity field
directly, without simulating any SDE. The decision rule: when
few-step sampling matters most, straighten the path. Diffusion
curves, flow matching draws the chord.

## What is used where: scores and shortcuts in production

| Tool | Where it runs | Evidence |
|---|---|---|
| Classifier-free guidance | Stable Diffusion, DALL-E 2, Imagen: the prompt-following knob (w ~ 7) | Public: Ho and Salimans, 2022, arxiv 2207.12598, model cards |
| DDIM / DPM-Solver schedulers | HuggingFace diffusers: the default few-step samplers | Public: diffusers library, open source |
| Rectified flow | Stable Diffusion 3: straight-line training objective | Public: Esser et al., 2024, arxiv 2403.03206 |
| Flow matching | Meta's Movie Gen and recent video models [uncertain] | Research direction. Exact production use varies |
| EDM (Karras et al., 2022) | Elucidating the Design Space of diffusion models: the preconditioning recipe behind many tuned samplers | Public research: arxiv 2206.00364 |

The pattern: nobody samples DDPM's raw 1,000 steps in
production. The stack is latent diffusion (L05) plus a
few-step ODE sampler (DDIM/DPM-Solver) plus classifier-free
guidance. The score view is what made that stack composable.

> [!QA]
> Q: What is the score, in plain terms?
> A: The gradient of the log-density: s(x) = grad log p(x). It points toward higher probability with strength equal to the steepness. For N(0,1), s(x) = -x: at x = 2.3 the score is -2.3, pointing back toward the mean. A sampler that follows the score walks uphill on the probability surface.
> Follow-up: Why the log-density and not the density itself?
> A: The log turns products into sums (the chain rule of L02 becomes addition) and the normalizing constant vanishes under differentiation. The score sees the shape of p but not its scale, which is why score models sample well yet cannot report likelihoods.

> [!QA]
> Q: How does denoising teach the score without a density?
> A: Corrupt x to x-tilde = x + sigma x epsilon and train the network to output (x - x-tilde)/sigma^2. On the toy: x = 2.0, sigma = 0.5, epsilon = 0.6 gives x-tilde = 2.3 and target -1.2. Averaged over corruptions, this target equals the score of the noise-perturbed distribution, which approaches the true score as sigma shrinks. The network never estimates a density, so there is nothing to differentiate and no wobble to amplify.
> Follow-up: Why corrupt at many noise levels instead of one small sigma?
> A: One small sigma leaves distant regions empty: a corruption of width 0.01 never reaches x = 3 from data near 0, so the score there stays unlearned. Large sigmas blanket the space and teach coarse return directions. Small sigmas refine them. The multi-level schedule is the annealed Langevin chain, which is DDPM's reverse process.

> [!QA]
> Q: What is the exact relation between DDPM's noise prediction and the score?
> A: s(x_t) = -epsilon-hat(x_t, t) / sqrt(1 - alpha-bar_t). On the L05 toy at t = 1: epsilon-hat = 0.42, sqrt(1 - 0.99) = 0.1, so s = -4.2. DDPM's MSE-on-noise training is denoising score matching at every noise level at once. Same machine, two notations.
> Follow-up: If they are the same, why have both views?
> A: The DDPM view gives the ELBO, the sampling loop, and DDIM's shortcuts. The score view gives Langevin sampling, the SDE unification (Song et al. 2021), and guidance as score arithmetic. Each view gives different tools. The lecture's point is that the tools compose because the object is one.

> [!QA]
> Q: Walk me through one DDIM jump.
> A: Start at t = 1000 with x_1000 (pure noise). The network predicts eps-hat. Estimate the clean image: x_0-hat = (x_1000 - sqrt(1 - alpha-bar_1000) eps-hat) / sqrt(alpha-bar_1000). Pick the target s = 800. Re-noise the estimate to that level: x_800 = sqrt(alpha-bar_800) x_0-hat + sqrt(1 - alpha-bar_800) eps-hat. No fresh noise added (eta = 0). Repeat: 800 -> 600 -> ... -> 0. Fifty jumps, same network, 20x fewer calls than DDPM.
> Follow-up: Why is eta = 0 deterministic?
> A: Because every jump is a fixed function of x_t and the network's prediction: estimate x_0, re-noise with the predicted (not random) noise. Same starting x_T gives the same final image, exactly. Eta = 1 adds fresh Gaussian noise at each jump, recovering DDPM-like stochasticity and diversity.

> [!QA]
> Q: Classifier guidance vs classifier-free guidance: which and when?
> A: Classifier guidance adds w times a separately trained noisy classifier's gradient to the score. It works but needs a second model trained at every noise level. Classifier-free guidance trains one denoiser with random condition-dropout, giving conditional and unconditional predictions, then extrapolates: eps_u + w(eps_c - eps_u). On the toy: 0.42 + 3 x 0.18 = 0.96. Classifier-free won in production because there is no second model to train and maintain.
> Follow-up: What goes wrong at very high guidance scale?
> A: The extrapolation overshoots: colors oversaturate, textures melt, diversity collapses. The sample obeys the prompt but leaves the data manifold. Production default w ~ 7 balances fidelity and realism. Past w ~ 15 the artifacts dominate.

> [!QA]
> Q: What is flow matching, and why does Stable Diffusion 3 use it?
> A: Instead of diffusion's curved noising path, flow matching draws straight lines: x_t = (1 - t) x_0 + t eps, and trains the network to predict the velocity v = eps - x_0. Straight paths need fewer integration steps to follow accurately. SD3 uses rectified flow (the same straight-line idea) because few-step sampling matters for a product: straighter paths mean better images at 28 steps than curved paths give.
> Follow-up: Is flow matching still diffusion?
> A: It is the ODE cousin. Diffusion learns the score of a stochastic process. Flow matching learns the velocity of a deterministic one. Both train by plain regression, both sample by integrating a learned field. The SDE trio plate shows the family: same score/velocity idea, different integrators.

> [!QA]
> Q: Applied: your text-to-image product gets "ignores the prompt" complaints at 20 DDIM steps. What do you change?
> A: Raise the guidance scale first: w from 7 toward 10-12 strengthens prompt adherence without touching the sampler. If faces still melt, the steps are too few for the curvature: switch the scheduler to DPM-Solver (fewer steps, better accuracy) or raise to 30 steps. If latency forbids it, distill the model into a few-step student. The interview signal: diagnose in order (guidance, then sampler, then distillation), and name what each costs: artifacts, then latency, then training.
> Follow-up: Why not just train a bigger model?
> A: A bigger model does not fix a sampler problem: 20 curved steps still undershoot. Guidance and the integrator are the levers that act on few-step quality. Capacity helps the per-step prediction, not the path error.

![Chapter plate: DDPM noise is score in disguise](assets/plate-l06.png "Chapter plate. DDPM noise is score in disguise: s = -eps / sqrt(1 - alpha-bar). Source: original plate for Stanford Frontier AI.")

## Recap: the whole lesson on one screen

1. **The question, through a new lens.** Learn the rule. Draw fresh samples. Learn the slope of log-probability and walk uphill.
2. **The score, concretely.** s(x) = grad log p(x). For N(0,1): s(x) = -x. At 2.3, the slope is -2.3 toward the mean.
3. **Density-first breaks, demonstrated.** A 30% wobble in learned density turns true score -3 into -1.793. Differentiation amplifies error, worst in empty regions.
4. **The key question.** What if we learn the score directly by denoising, never estimating a density?
5. **Denoising score matching, by hand.** x = 2.0, sigma = 0.5, epsilon = 0.6: x-tilde = 2.3, target -1.2. Averages to the perturbed score.
6. **Langevin, by hand.** delta = 0.1, z = -0.4: 2.3 -> 2.059. Score pulls in, noise explores. Annealed schedule equals DDPM's chain.
7. **The SDE trio.** Forward SDE destroys, reverse SDE restores, probability-flow ODE walks straight. One score drives all three.
8. **DDPM predicts scores.** s = -0.42/0.1 = -4.2 on the L05 toy. One machine, two notations.
9. **DDIM.** 1,000 steps -> 50 jumps, same network, eta = 0 deterministic. 20x speedup.
10. **Guidance.** Classifier vs classifier-free. eps-tilde = 0.42 + 3 x 0.18 = 0.96. w ~ 7 in production.
11. **Flow matching.** Straight lines, velocity targets. SD3's rectified flow.
12. **Prices and payments.** No likelihoods (constant discarded). Step-size fragility: delta = 1.0 overshoots 2.3 -> 1.15.
13. **In production.** Latent diffusion plus DDIM-family samplers plus classifier-free guidance. Nobody ships raw DDPM.

## Official sources and further reading

**Official:**
- W9L35: DDPMs as score-predictors (video 2Sp0BqAWWXY) and W9L38: DDIMs (video qiMJBB8chzI): the lectures this chapter follows. [uncertain]: the exact toy numbers in the lectures are not verified. The worked values here are original toys built to the lectures' topics.
- Song & Ermon (2019): https://arxiv.org/abs/1907.05600 (score-based modeling with annealed Langevin).
- Song et al. (2021): https://arxiv.org/abs/2011.13456 (the SDE unification of DDPM and score matching).
- Song, Meng & Ermon, DDIM (2020): https://arxiv.org/abs/2010.02502 (deterministic few-step sampling).

**Further reading:**
- Ho & Salimans (2022), classifier-free guidance: https://arxiv.org/abs/2207.12598 (the epsilon_u + w(epsilon_c - epsilon_u) formula).
- Lipman et al., Flow Matching (2022): https://arxiv.org/abs/2210.02747 (straight lines, velocity targets).
- Yang Song's blog on score matching: https://yang-song.net/blog/2021/score/ (the long-form reference).

**Caveats from these sources.** The 1-D toys are illustrative. Real score models estimate high-dimensional score fields with U-Nets. The denoising target equals the perturbed score only in expectation over corruptions. Single targets are noisy. DDIM's quality at 50 steps depends on the data and the schedule. The 20x figure is representative, not guaranteed.

## Connections to the other courses

- **L05 (this course):** the same machine in DDPM notation. The 0.846 toy reused and re-read as a score.
- **CS229 L11:** annealed Langevin as the conceptual parent of the reverse chain.
- **math-genai (sibling):** guided diffusion (W9 lectures): classifier and classifier-free guidance in the lecture's own framing.
- **CS336:** the denoiser architecture: the U-Net/transformer that predicts epsilon at every scale, and how t conditions it.
- **L04 (this course):** flow matching reunites the warper with the restorer: the velocity field is a continuous flow's drift.
