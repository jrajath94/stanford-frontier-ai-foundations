---
page_id: cs336-l09
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 9
nav: "L09 · Scaling Laws I"
title: "Lecture 9: Scaling Laws, Part 1"
summary: "The basics of scaling laws: data scaling and its statistical roots, scaling laws for model engineering, critical batch size, and the Kaplan versus Chinchilla story of compute-optimal training."
date: "2026-04-27"
instructor: "Tatsunori Hashimoto"
offering: "Spring 2026"
duration: "1:17:57"
video_id: Q15rhEWZPQ4
video_title: "Stanford CS336 Spring 2026 Lecture 9: Scaling Laws (Part 1)"
video_caption: "Original lecture. Timestamps link to exact moments."
concepts: [scaling laws, kaplan, chinchilla, compute optimal, loss prediction]
papers: []
sources:
  - tag: video
    label: "Lecture 9 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=Q15rhEWZPQ4
  - tag: slides
    label: "lecture_09.pdf, official lecture slides"
    url: https://github.com/stanford-cs336/lectures
  - tag: notes
    label: "Official subtitle transcript (en-orig)"
---

## The B200 thought experiment

A wealthy friend gives you 10,000 B200s for a month and asks for a good open-source language model [00:53](ts:00:53). The infra team is assembled. The dataset is ready. Now you must run the big training run. Wide or deep? How many heads? Which nonlinearity? What learning rate?

Tuning hyperparameters on the big run is terrifying and wasteful. **Scaling laws** offer the alternative: simple predictive rules that map small-scale behavior to large-scale behavior [02:17](ts:02:17). Do all your optimization at small scale, fit a regularity, and extrapolate to the big run with confidence.

This is the first of two scaling lectures. Today covers the basics and the classics. A later lecture goes advanced: modern tech reports, new parameterizations, and optimizers.

## Scaling laws have a long history

Scaling laws feel new. They are not. Theorists have asked "how will my model perform?" for decades, and their answer was **generalization bounds**: upper bounds on error as a function of sample size [04:32](ts:04:32). A data scaling law is the empirical cousin of a sample complexity bound.

| Year | Work | Contribution |
|---|---|---|
| 1993 | Cortes and Vapnik, Bell Labs | First data scaling law. Fit classifiers on small samples, fit a curve to error decay, estimate large-data performance. |
| 2001 | Banko and Brill | Log-linear scaling with data. Asked whether money is better spent on data than on algorithms. |
| 2012 | Collobert and colleagues | Functional forms for scaling. Power laws relating data to downstream performance. |
| 2017 | Hestness et al. | First large-scale neural scaling study. Predictable scaling across speech, translation, and language modeling. |

Hestness et al. was far ahead of its time [08:09](ts:08:09). It discussed emergence (accuracy is discontinuous while loss is smooth), scaling by compute, and the idea that speed and systems optimization turn into accuracy. Much of today's discourse was foreseeable in 2017.

> [!PROF] When a student asks whether scaling laws are derived or just found empirically, the answer is blunt: they are pure curve-fitting exercises. Theory and physics suggest candidate functional forms, but there is no golden rule that these are the only ones [10:29](ts:10:29).

## Data scaling: the basic object

Fix the model and training procedure. Keep the model bigger than the dataset, so you stay in the power-law regime rather than the irreducible-error regime where the model has already fit the data as well as it can [13:35](ts:13:35). Increase the dataset size and watch the error fall.

Expect a monotone, sigmoidal-ish curve: from random guessing down toward the entropy noise floor. In practice, plotting log test loss against log dataset size gives a strikingly straight line [14:18](ts:14:18).

A straight line on a log-log plot is a **power law**: error decays polynomially. It also means you are far from the asymptote. Near the noise floor the curve would taper.

```mermaid
flowchart LR
    A[More data] --> B[Log-log plot<br>is linear]
    B --> C[Error decays<br>polynomially]
    C --> D[Far from<br>noise floor]
```

The usual x-axes are all resources: log compute, log data, log parameters. Each linearizes log test loss. Downstream benchmarks give sigmoid curves instead, and even dates on the x-axis can look linear for capability envelopes [12:01](ts:12:01).

## Toy example: estimating a mean

Why should error decay polynomially? Take the simplest estimation problem [15:43](ts:15:43). Draw samples from a Gaussian and estimate the mean with the empirical average:

\[ E[(\hat{\mu} - \mu)^2] = \frac{\sigma^2}{n} \]

Take logs:

\[ \log(\text{error}) = -\log n + 2\log\sigma \]

That is a scaling law. Anything of the form \(1/n^{\alpha}\) plus a constant becomes linear on a log-log plot once you subtract the constant. Classical parametric models give \(\alpha = 1\).

## The exponent mystery

Here is the puzzle [17:07](ts:17:07). Classical models predict slope −1. Neural scaling laws show exponents around −0.1 to −0.3 for translation, speech, and language modeling. Neural nets learn much more slowly than mean estimation.

A detour into nonparametric statistics suggests why. Suppose you estimate an arbitrary smooth function on a 2D unit box by cutting it into boxes of side length \(n^{-1/4}\) [17:51](ts:17:51). You get \(\sqrt{n}\) boxes, each holding about \(\sqrt{n}\) samples, and error around \(1/\sqrt{n}\). In d dimensions the same argument gives:

\[ \text{error} \approx n^{-1/d} \]

So the log-log slope is \(-1/d\). A slope of −0.1 looks like a nonparametric regressor in about 10 dimensions. Some researchers (Bahri et al. 2021) argue the exponent literally reveals the intrinsic dimensionality of the data.

> [!CAVEAT] The lecturer does not fully buy the intrinsic-dimension story. Estimators of intrinsic dimension are sketchy, and the argument is not airtight. Treat the exponent as a useful measurement of how fast the model learns, not as a proven fact about data geometry [19:59](ts:19:59).

## Composition, repetition, and filtering

Data scaling laws so far map dataset *size* to error. Three engineering questions go further.

**Data mixture.** Think of data scaling laws as empirical generalization bounds. Classical reasoning says dataset composition changes the *offset* of the scaling law, not its slope, which is set by the model class [21:21](ts:21:21). So to pick a news-versus-Wikipedia mixture, fit the mixture's effect on small models and extrapolate. In practice, people skip the curve fitting: train small models on candidate mixtures and scale up the winner. The DataDecide study found this works, which is consistent with slopes being mixture-independent [24:25](ts:24:25).

**Repetition.** Compute grows faster than data, so data gets repeated. Up to about 4 epochs, standard recipes show no harm. Past that, realized scaling falls well below the fresh-data projection. Modified functional forms with effective data size capture this [25:11](ts:25:11). Pushed to the extreme of infinite compute, you cannot keep repeating or growing the model. Gains come from elsewhere, like ensembling. Notably, the slopes stay similar. Interventions usually move the intercept, not the slope [26:01](ts:26:01).

**Filtering.** Filtering is scale-dependent [27:30](ts:27:30). With little compute you filter aggressively, keeping only the highest-quality data, because you cannot train on everything anyway. With huge compute you loosen the filters, because you need volume and do not want to repeat the small high-quality set. Optimal filtering changes with scale.

> [!WARN] Axes can lie. One plot in the lecture looks linear in both loss and data size, but the x-axis is doubling, so it is secretly log scale, and the y range is so narrow that linear and log look identical. With only a tiny slice of compute range, polynomial and exponential are indistinguishable. Always be skeptical of the functional form [28:52](ts:28:52).

## Scaling laws for model engineering

Beyond data, scaling laws answer design questions: architecture, optimizer, depth and width, batch size, learning rate. The procedure is always the same [53:04](ts:53:04):

1. Train several smaller models.
2. Establish a scaling law (e.g., Adam versus SGD curves).
3. Pick the hyperparameter the scaling law predicts is best at scale.

**Architecture.** Are transformers really better than LSTMs? The brute-force answer costs tens of millions. The scaling-law answer trains small LSTMs and transformers across compute ranges and compares curves [32:01](ts:32:01). LSTMs show a worse intercept and possibly a worse slope. A worse slope is disqualifying: it means the gap grows with scale. Tay et al. 2022 ran this analysis across T5-style architectures. Efficient attention (Performer) scaled poorly, which is why frontier models do not use it. GLU scaled well, which is why they do [32:46](ts:32:46).

> [!PAPER] Kaplan et al. 2020, "Neural Scaling Laws for Language Models" (OpenAI). The canonical study: power-law fits for loss versus compute, data, and parameters, plus ablations of architecture, depth, and aspect ratio. Referenced throughout this lecture.

**Optimizer.** Hestness 2017 compared Adam and SGD. Different intercepts, strikingly similar slopes [34:21](ts:34:21). Even an intervention as large as changing the optimizer leaves the slope nearly unchanged. This pattern repeats everywhere: interventions move intercepts, not slopes.

**Depth and width.** One layer versus two is a huge difference. Beyond that, more layers show diminishing returns. The aspect ratio (depth to width) is a scale-invariant quantity: the optimum sits around 100 d_model per layer and barely shifts as models grow, so fixing the aspect ratio while scaling is safe [35:52](ts:35:52).

**Not all parameters are equal.** Kaplan found that including embedding parameters produced bizarre scaling curves, so they counted only non-embedding parameters. They also excluded the final layer's parameters, since embeddings and the output matrix are duals with the same shape. These bookkeeping choices have non-trivial consequences, as the Chinchilla story will show [37:17](ts:37:17).

**MoE parameters.** With mixtures of experts, total and active parameters decouple. An Apple and MIT analysis shows that as total parameters grow, the loss-minimizing design gets sparser, and inactive parameters still reduce loss [38:42](ts:38:42). The "value of a parameter" becomes a real question.

## Critical batch size

Batch size is special for systems reasons: data parallel needs large batches. The question is how large you can go before suffering [40:54](ts:40:54).

Two regimes. In the **noise-limited** regime, each added batch element reduces gradient variance, and returns are near-perfect. In the **bias-limited** regime, you have hit the noise scale of the gradients. Your descent direction disagrees with the true direction to the optimum no matter how low the noise gets, so extra batch elements give diminishing returns [42:26](ts:42:26).

The **critical batch size** is the crossover point: the largest batch worth using. The estimation procedure [44:47](ts:44:47):

1. Pick a target loss.
2. Sweep batch sizes. Record steps needed (S) and examples needed (E) to reach the target.
3. Fit for the minima S_min and E_min, then set B_crit = E_min / S_min.

This balances the two costs: it uses slightly more steps than optimal and slightly more examples than optimal. It is also close to the ratio of the gradient covariance trace to the squared gradient norm.

Why does this belong in a scaling lecture? Because the critical batch size itself scales [47:02](ts:47:02). As the target loss drops, the critical batch size grows, following a power law. Closer to the minimum, you optimize finer-grained structure, and variance reduction matters more. Large runs can use very large batches.

## Learning rates

Naive width scaling says the learning rate should shrink as the model grows, roughly as 1 over width: more parameters moving at once means each should move less [48:33](ts:48:33). The alternative school rescales initialization and per-layer step sizes so the optimal learning rate stays fixed across scales. This is **muP** (maximal update parameterization).

Two philosophies for picking the big-run learning rate [49:18](ts:49:18): predict how the optimum shifts with scale, or reparameterize so the optimum does not shift. Both have succeeded at scale. Anecdotally, more practitioners now favor the scaling-law approach. Full details come in the advanced scaling lecture.

## Upstream versus downstream

Scaling laws are cleanest on perplexity. Perplexity transfers to downstream tasks, but unreliably [50:52](ts:50:52). In one architecture study, the best-perplexity model was far from the best downstream model. Perplexity scaling is regular and predictable. Downstream scaling is much less certain.

> [!PROF] Former students working in post-training complain that pretraining teams hand over models saying "the perplexity is good, it is all your problem now." The problems often start in pretraining. Think about transfer, not just perplexity [52:24](ts:52:24).

Two measurement notes. Scaling-law points are usually single runs, not averages, because perplexity has tiny variance: reruns differ in the second decimal place [53:44](ts:53:44). And most plots mix train and validation loss freely, because one-pass SGD makes the generalization gap negligible. Some pretraining codebases do not even track validation loss [55:22](ts:55:22).

## Compute-optimal training: Kaplan versus Chinchilla

The most famous use of scaling laws: given fixed FLOPs, should you train a bigger model or train longer? A tiny model drowned in data wastes compute. Joint data-model scaling laws quantify the tradeoff [56:59](ts:56:59).

Rosenfeld et al. 2020 proposed a simple additive form:

\[ \text{error} = n^{-\alpha} + m^{-\beta} + C \]

where n is data and m is model size. Kaplan et al. 2020 proposed a similar joint form. Sanity-check by taking limits: infinite data leaves a pure model-size law, infinite model leaves a pure data law [57:43](ts:57:43). Fit on small models and small data, extrapolate far. It works surprisingly well.

Then optimize under the FLOPs constraint (FLOPs scale roughly as data times parameters). Kaplan's optimum [60:00](ts:60:00):

\[ N_{opt} = C^{0.73}, \quad D_{opt} = C^{0.27} \]

Tokens per parameter *decrease* with compute. This drove the era of giant models: GPT-3 trained on about 2 to 3 tokens per parameter, and labs raced toward ever-larger models.

> [!PAPER] Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla, DeepMind). Argued Kaplan's fits were far off and that models were systematically undertrained. Proposed 20 tokens per parameter and three fitting methods that mostly agree.

Chinchilla's three methods [61:39](ts:61:39):

1. **Lower envelope.** Take all training curves, trace the lower envelope (best loss ever achieved at each FLOP count), and read off the model size at each envelope point. Predicts a 67B model for their budget.
2. **IsoFLOPs.** Fix several FLOP budgets, sweep the parameter-to-data ratio at each, fit the convex loss curve, and connect the minima. Predicts 63B. The lecturer's favorite: easy and reliable.
3. **Joint fits.** Fit a hypothesized loss surface over the size-data grid by least squares. Disagreed with the other two (exponents 0.46/0.54 instead of ~0.5/0.5).

Methods 1 and 2 say parameters and data should scale together, hence the 20:1 rule. So why did Kaplan and Chinchilla disagree so much when both fit joint scaling laws?

Three explanations, each a lesson in how fragile scaling laws are [66:57](ts:66:57):

- **Parameter counting and tuning** (Yi et al., "Resolving Discrepancies in Compute Optimal Scaling of Language Models"). Replicating Kaplan's setup, then changing one thing at a time: excluding the final-layer parameters shifted the curve, overlong warmup at small compute left models under-converged, and a fixed batch size was suboptimal for small models. Fix all three and you land on Chinchilla.
- **Compute scale** (Pearson and Song). Simulating Kaplan-style fits from Chinchilla's curves with no new training: Kaplan operated at much lower compute, where small changes (like the non-embedding nonlinearity) have outsized effects.
- **Fitting error** (Besiroglu et al. 2024, Epoch AI). Method 3 never agreed with methods 1 and 2, and the reason was mundane: the original fit underfit. Re-extracting the data from the paper's plots and refitting recovered the 20:1 rule. The authors were more right than they knew.

> [!KEY] Scaling laws are lower bounds for a given recipe, not laws of nature. A scaling law fit on a bad recipe (wrong warmup, wrong batch size, wrong parameter counting) predicts bad outcomes. The details are the result.

## Overtrain for serving

One more correction: train-compute-optimal is probably not what you want [73:48](ts:73:48). Most of a lab's compute goes to R&D and serving, not training. Serving favors small, capable models over big, cheap-to-train ones. The right move is to **overtrain**: spend extra training compute to shrink inference cost.

| Model | Tokens per parameter |
|---|---|
| GPT-3 | ~2 |
| Chinchilla | 20 |
| LLaMA 65B | 22 |
| Llama 2 70B | 29 |
| Mistral 7B | 110 |
| Llama 3 70B | 215 |

Chinchilla matters less as a golden ratio than as a lesson in how to fit and think about scaling laws. And IsoFLOPs endures as a research tool: fix a FLOP budget, sweep the free parameters, read off the surface. It has been reused for diffusion models and MoEs [75:58](ts:75:58).

## Assignment connection

Assignment 3 is the scaling assignment. You will fit your own data scaling law, reproducing the log-log linearity from this lecture on small models [14:18](ts:14:18). The hyperparameter-transfer ideas (aspect ratio, batch size, learning rate) feed the prediction tasks: choose the big-model settings from small-model fits, the way Kaplan and Chinchilla did.

> [!INTERVIEW] Scaling-law questions test engineering judgment under uncertainty: how would you pick a model size for a fixed budget, why would you overtrain for serving, and what breaks when you extrapolate. Know the Kaplan versus Chinchilla disagreement cold, including the three mundane causes. Interviewers use it to separate people who memorized 20:1 from people who understand why.
