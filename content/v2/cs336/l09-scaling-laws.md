---
page_id: cs336-l09
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 9
nav: "L09 · Scaling Laws"
title: "Lecture 9: Scaling Laws, the Basics"
summary: "The engineering view of scaling: power laws from data to Chinchilla, critical batch size, the Kaplan correction, and why you overtrain for serving."
date: "2026-04-27"
instructor: "Tatsunori Hashimoto"
offering: "Spring 2026"
duration: "1:17:47"
video_id: Q15rhEWZPQ4
video_title: "Stanford CS336 Spring 2026 Lecture 9: Scaling Laws Basics"
video_caption: "Original lecture. Tatsunori Hashimoto builds the scaling-law toolkit from statistics to Chinchilla."
concepts: [scaling-laws, power-laws, chinchilla, kaplan, critical-batch-size, data-mixtures, upstream-downstream]
sources:
  - tag: video
    label: "Lecture 9 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=Q15rhEWZPQ4
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: slides
    label: "lecture_9.pdf, official course slides"
---

## How to read this lesson

Your wealthy friend hands you 10,000 B200s for a month. How do you
spend them without wasting millions? **Level 1 (Core):** the
engineering view, data scaling laws, and their uses. **Level 2
(Deep):** architecture scaling, critical batch size, the
Kaplan-Chinchilla saga, and the overtraining turn.

## Level 1: The engineering view

Scaling laws are predictive rules that connect small-scale behavior to
large-scale behavior [02:48](ts:02:48). The naive approach tunes
hyperparameters on the big run. The scaling approach optimizes at small
scale and extrapolates with a simple rule. It works only if the
small-to-large connection is regular, and making it regular is
engineering: pick the right x-axis, set hyperparameters right, and
predictability appears [38:28](ts:38:28).

```ascii
naive approach  : tune hyperparameters on the big run
scaling approach: optimize small, extrapolate with a rule
works when     : the small-to-large connection is regular
```

## Level 1: A long history

![History](assets/l09-history.svg "1993 Cortes and Vapnik, 2001 Banko and Brill, 2012 Kolachina, 2017 Hestness, 2020 Kaplan, 2022 Chinchilla.")

Generalization bounds were the theorists' scaling laws: error bounds
that decay with sample size [04:55](ts:04:55). Cortes and Vapnik (1993)
fit error decay on small samples to predict large ones: a data scaling
law, literally. Banko and Brill showed data beats algorithms. Hestness
(2017) found neural scaling across speech, translation, and language
modeling, and already noted emergence: accuracy is discontinuous where
loss is smooth [08:37](ts:08:37). The ideas are old. The scale is new.

## Level 1: Data scaling laws

Fix the model (bigger than the dataset, ~10x), grow the data, plot
log-log. A straight line means error decays polynomially: error ~
1/n^alpha plus the noise floor [14:34](ts:14:34).

![Log-log](assets/l09-loglog.svg "Slope is the exponent. Mean estimation gives -1. Neural nets give about -0.1.")

The stats interlude: estimating a Gaussian mean gives sigma^2/n, slope
-1. Neural nets show -0.1 to -0.3: much slower. The model that fits:
nonparametric regression in D dimensions has rate n^(-1/D). Neural
nets learn like smooth functions in ~10 dimensions [19:25](ts:19:25).
The exponent tells you how fast the network really learns.

> [!QA]
> Q: Why is the neural exponent (-0.1) so much worse than the parametric rate (-1)?
> A: Parametric estimators assume the truth lives in a small, known family: every sample refines a few numbers. A neural net makes far weaker assumptions, so it must learn the shape of the function too, like a nonparametric smoother spreading samples across a high-dimensional space. The exponent is the price of flexibility. That is also why the slope is stable across interventions: it reflects the model class, not the data.
> Follow-up: When does the log-log line break?
> A: Near the noise floor. The line is the far-from-asymptote regime. As error approaches the irreducible level, the curve tapers. That is why the setup keeps the model much bigger than the dataset: it stays in the power-law regime instead of the saturated one.

## Level 1: Data engineering with scaling laws

![Data uses](assets/l09-data-uses.svg "Mixtures shift intercepts. Repetition survives 4 epochs. Filtering loosens with scale.")

- **Mixtures.** Data composition shifts intercepts, not slopes. Fit
  mixtures at small scale, pick the best, scale it up. Often no scaling
  law is even needed: the best small mix is the best large mix
  [24:48](ts:24:48).
- **Repetition.** Up to ~4 epochs, no harm. Past it, the realized curve
  falls below the fresh-data projection. At infinite compute: stop
  repeating, start ensembling [25:42](ts:25:42).
- **Filtering.** Scale-dependent. Small compute: filter hard, keep the
  best. Big compute: loosen the filter, or you run out of data and
  repeat yourself [27:34](ts:27:34).

The recurring lesson: interventions move intercepts, rarely slopes
[27:05](ts:27:05).

## Level 1: Architecture and optimizer scaling

Transformers vs LSTMs: different intercepts, possibly different
slopes. The plot justifies the transformer [31:40](ts:31:40). Tay et
al.'s T5 studies caught the winners early: GLU scales well, Performer
does not, Switch Transformer does [33:18](ts:33:18). Rule of thumb:
if it does not show up in the scaling law, it is not a good
intervention [34:35](ts:34:35).

| Change | Intercepts | Slopes |
|---|---|---|
| Transformer vs LSTM | different | transformer wins |
| GLU | better | scales well |
| Performer | worse | scales poorly |
| SGD vs Adam | different | nearly identical |

SGD vs Adam (Hestness): different intercepts, nearly identical slopes.
Even that big a change leaves the slope alone [34:57](ts:34:57).
Aspect ratio is scale-invariant: ~100 dims per layer, minima stable
across sizes, deeper slightly favoring smaller ratios
[36:47](ts:36:47). One layer is hopeless. Beyond that, more layers win
at fixed compute.

Caution: not all parameters are equal. Kaplan excluded embeddings and
got clean laws. For MoEs, Apple/MIT show the right axes are total vs
active parameters, with sparsity growing at scale and inactive
parameters still helping [39:09](ts:39:09).

## Level 2: Critical batch size

![Critical batch](assets/l09-critical-batch.svg "Noise-limited: variance reduction pays. Bias-limited: the local view disagrees with the global optimum.")

Two regimes [42:24](ts:42:24). Noise-limited: extra examples cut
gradient variance, perfect returns. Bias-limited: you see only the
local objective, and the local direction disagrees with the global
optimum, so extra examples buy little. Critical batch size is the
crossover: B_crit = min examples / min steps, fit from sweeps
[46:01](ts:46:01).

It scales: as the target loss drops, B_crit grows as a power law.
Closer to the minimum, noise matters more, so big runs earn big
batches [47:36](ts:47:36).

Learning rate, the other knob: width scaling suggests LR ~ 1/width.
The alternative school, muP, reparameterizes so the optimal LR stays
fixed across scales. Both have succeeded at scale. The advanced
lecture goes deeper [48:34](ts:48:34).

## Level 2: Upstream is not downstream

![Upstream downstream](assets/l09-upstream-downstream.svg "Perplexity: linear and clean. Downstream tasks: noisy. NL12 won perplexity, NL32XL won tasks.")

Perplexity vs parameters: a beautiful line. Downstream: the best
perplexity model (NL12) lost to a worse-perplexity one (NL32XL)
[51:37](ts:51:37). Fit scaling laws on perplexity, where variance is
tiny (singletons, second-decimal noise). Then verify transfer
separately. The post-training team inherits pre-training's sins
[52:13](ts:52:13).

The procedure: train small models, fit log-log lines, predict the big
run's numbers before spending the money [53:04](ts:53:04).

## Level 2: The Kaplan-Chinchilla saga

The question: fixed FLOPs, more data or bigger model? FLOPs ~ N x D.
Tiny model plus huge data wastes compute (flat purple line)
[57:32](ts:57:32). Kaplan and Rosenfeld fit joint laws L(N, D). Kaplan
said N ~ C^0.27, D ~ C^0.73: train giants. The GPT-3 era listened
[60:21](ts:60:21).

![Kaplan vs Chinchilla](assets/l09-kaplan-vs-chinchilla.svg "Kaplan: N^0.27, train giants. Chinchilla: N^0.5, 20 tokens per param. The details decided.")

Chinchilla (2022) said both exponents ~0.5: 20 tokens per parameter,
smaller models trained longer. Three methods agreed: lower envelope,
IsoFLOP, parametric fit [62:30](ts:62:30).

![Three methods](assets/l09-three-methods.svg "Envelope, IsoFLOP, parametric fit. IsoFLOP is the default you reach for.")

Why Kaplan lost: they excluded the unembedding parameters (big effect
at small scale), ran warmup too short for small models to converge,
and fixed a batch size that was suboptimal for small models
[67:55](ts:67:55). Minor details, big shifts. Epoch AI later showed
Chinchilla's method 3 was underfit: refit, it agrees with methods 1
and 2 [73:15](ts:73:15).

> [!QA]
> Q: If Chinchilla is right, why does nobody train at exactly 20 tokens per parameter?
> A: Chinchilla optimizes training compute. Production optimizes serving: most FLOPs go to R&D and inference, not the training run. Serving wants small, capable models, so labs overtrain far past 20:1. The ratio answers "how do I spend a training budget," not "what model should I ship." Different objective, different optimum.
> Follow-up: What is the single most reliable scaling-law tool?
> A: IsoFLOP. Fix the FLOP budget, sweep the N-vs-D tradeoff, take the minima, draw the line through them. It makes the fewest modeling assumptions of the three Chinchilla methods, and it generalizes: fix any budget, sweep any tradeoff. The lecture calls it the default for a reason.

## Level 2: Overtrain for serving

![Overtrain](assets/l09-overtrain.svg "GPT-3: 3 tokens per param. Chinchilla: 20. Modern serving: 100+, deliberately overtrained.")

GPT-3: 3 tokens per parameter, undertrained. Chinchilla: 20, the
research optimum. Modern: overtrained on purpose, because the model
that minimizes training cost is big and expensive to serve
[75:25](ts:75:25). Chinchilla's importance is not the number 20. It is
the method: how to fit scaling laws and how to think about them.

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l09-loglog.svg" alt="Log-log power law">
<div class="rc-body">
<strong>1. Slope is the exponent</strong>
<p>Log-log line means polynomial decay. Mean estimation: -1. Neural
nets: about -0.1, like nonparametric regression in ~10 dims.</p>
<p class="rc-num">Key: slope = learning speed</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-history.svg" alt="History">
<div class="rc-body">
<strong>2. Old idea, new scale</strong>
<p>Cortes and Vapnik 1993, Banko and Brill, Hestness 2017, Kaplan
2020, Chinchilla 2022. Same power law, bigger machines.</p>
<p class="rc-num">Key: credit the lineage</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-data-uses.svg" alt="Data engineering">
<div class="rc-body">
<strong>3. Slopes are stubborn</strong>
<p>Mixtures, repetition, filtering: intercepts move, slopes stay.
Pick the best mix small, scale it. 4 epochs safe. Loosen filters at
scale.</p>
<p class="rc-num">Key: intercepts move</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-critical-batch.svg" alt="Critical batch size">
<div class="rc-body">
<strong>4. Size the batch to the noise</strong>
<p>Noise-limited: perfect returns. Bias-limited: diminishing. B_crit
grows as loss drops. Big runs earn big batches.</p>
<p class="rc-num">Key: B_crit = E_min / S_min</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-kaplan-vs-chinchilla.svg" alt="Kaplan vs Chinchilla">
<div class="rc-body">
<strong>5. Details decide</strong>
<p>Kaplan said train giants (N^0.27). Chinchilla said 20 tok/param
(N^0.5). Param counting, warmup, batch size: the devil, in the
details.</p>
<p class="rc-num">Key: 20 tokens per param</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-three-methods.svg" alt="Three methods">
<div class="rc-body">
<strong>6. Fit three ways</strong>
<p>Lower envelope, IsoFLOP, parametric fit. IsoFLOP is the reliable
default. Epoch AI: method 3 was underfit, agrees on refit.</p>
<p class="rc-num">Key: IsoFLOP by default</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-overtrain.svg" alt="Overtrain">
<div class="rc-body">
<strong>7. Serve small, train long</strong>
<p>Training-optimal is not serving-optimal. GPT-3 was undertrained
at 3:1. Modern models overtrain deliberately.</p>
<p class="rc-num">Key: optimize the deployment</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l09-upstream-downstream.svg" alt="Upstream vs downstream">
<div class="rc-body">
<strong>8. Perplexity is not the task</strong>
<p>Scaling laws are clean for loss, noisy for downstream. Fit on
perplexity, verify transfer. Post-training inherits your sins.</p>
<p class="rc-num">Key: measure both</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 9 video and slides (lecture_9.pdf).
- Kaplan et al. 2020, "Scaling Laws for Neural Language Models."
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language
  Models" (Chinchilla).

**Further reading:**
- Hestness et al. 2017: the under-cited origins.
- "Resolving Discrepancy to Compute-Optimal Scaling" (Porian et
  al.): why Kaplan and Chinchilla disagree.
- Epoch AI refit of Chinchilla method 3.
- Tay et al. architecture scaling studies.

**Caveats from these sources.** Exponents are fit, not derived: treat
them as recipe lower bounds, not physics. The 20:1 ratio is
training-optimal, not serving-optimal. Downstream transfer is the
weakest link in every claim here.

## Connections to the other courses

- **CS336 Lecture 11:** advanced scaling: muP, initialization,
  optimizers.
- **CS336 Lecture 2:** the 6ND compute model that the joint laws
  optimize.
- **CS229:** generalization bounds, the theoretical ancestor.

> [!CHEAT]
> **Scaling laws cheatsheet.** View: optimize small, extrapolate big. Log-log line = power law, slope = exponent. Stats: mean est -1, nets -0.1 (~nonparametric, 10 dims). History: 1993 Cortes/Vapnik, 2017 Hestness, 2020 Kaplan, 2022 Chinchilla. Data: mixtures move intercepts, slopes stay. Repetition: 4 epochs safe. Filtering loosens with scale. Arch: GLU good, Performer bad, Switch good. SGD vs Adam: same slopes. Aspect ratio ~100, scale-invariant. Critical batch: noise vs bias limited, B_crit = E_min/S_min, grows as loss drops. LR: 1/width or muP. Upstream vs downstream: fit ppl, verify tasks. Joint: Kaplan N^0.27 (giants), Chinchilla N^0.5 (20 tok/param). Fit: envelope, IsoFLOP (default), parametric. Kaplan lost on: unembedding count, warmup, batch size. Serve: overtrain, small and capable.

> [!MEMORY]
> **Slopes are stubborn, intercepts move.** Fit the law small, spend the budget big, and serve the overtrained model.
