---
page_id: math-genmodels-l07
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 7
nav: "L07 · Energy-Based Models"
title: "Lecture 7: Energy-Based Models — The Critic"
summary: "The critic answers the one question with valleys: learn an energy function, low for real outcomes, high for the rest, and sample by rolling downhill. The partition-function bill demonstrated, contrastive divergence worked by hand."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
concepts: [energy-based-model, partition-function, contrastive-divergence, mcmc, langevin-dynamics, negative-sampling]
sources:
  - tag: paper
    label: "Hinton, Training Products of Experts by Minimizing Contrastive Divergence (2002)"
    url: https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
  - tag: paper
    label: "LeCun et al., A Tutorial on Energy-Based Learning (2006)"
    url: https://www.cs.nyu.edu/~yann/research/ebm.pdf
---

## The question, for valleys

Restate the one question: learn the rule behind samples, draw
fresh samples from it. The critic's answer: assign every outcome
an **energy**, a single number. Low energy for real-looking
outcomes, high energy for the rest. Real faces sit in valleys. Noise sits on hills. To sample, start from noise and roll
downhill toward low energy. The model never writes a probability
table. It sculpts a landscape.

## First attempt: energies into probabilities

Energies become probabilities through the Boltzmann formula:

```ascii
p(x) = exp(-E(x)) / Z,    Z = sum over all x of exp(-E(x))
```

**Z**, the **partition function**, normalizes: it is the sum that
makes the probabilities add to 1. Work it on 4 faces with
energies E = [2, 1, 1, 0]:

```ascii
exp(-E) = [0.1353, 0.3679, 0.3679, 1.0000]
Z = 0.1353 + 0.3679 + 0.3679 + 1.0000 = 1.8711
p = [0.0723, 0.1966, 0.1966, 0.5345]
```

Face 4 has the lowest energy and the highest probability,
0.5345. The machine works: energies in, probabilities out. To
sample, roll downhill: start at a random face, step toward lower
energy, add a little noise so you explore.

## Where it breaks: the partition bill

Z sums over all outcomes. For the 4-face toy that is 4 terms.
For a 32x32 binary image it is 2^1024 terms, each needing an
energy evaluation. At a billion evaluations per second, 2^1024
evaluations take about 10^290 years. The partition function is
the exploding table from L01 wearing a sum sign.

The bill infects training too. Maximum likelihood wants the
gradient of log p(x) = -E(x) - log Z. Differentiate:

```ascii
d/dtheta log p(x) = - dE(x)/dtheta + E_model[dE/dtheta]
                    ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
                    positive phase:  negative phase:
                    lower energy     raise energy of what
                    of real data     the model likes
```

The positive phase is easy: one data point, one gradient. The
negative phase is an expectation under the model itself, which
needs Z. You cannot train without Z, and you cannot compute Z.
The naive critic is stuck.

## The key question

What if we train the energy without ever computing Z, by
comparing real data against the model's own fantasies?

## The new idea: contrastive divergence

Read the gradient again. It is a contrast: the energy slope on
data minus the energy slope on model samples. Push energy down
where data sits. Push it up where the model currently puts mass.
If the model's fantasies match the data, the two terms cancel
and learning stops. At that point the energy landscape's valleys
coincide with the data, with Z never computed.

The remaining problem is the model expectation: sampling from
the model needs MCMC, and MCMC needs many steps to mix.
**Contrastive divergence** (Hinton, 2002) cuts the chain short:
start the chain at a data point, run k steps (often just 1), and
use the landing spot as the negative sample. The chain does not
reach the model's true distribution, but it reaches the model's
*current* fantasies near the data, which is where the contrast
matters.

Work CD-1 by hand. Four faces, energies E_i = -theta_i, data
probs [0.1, 0.2, 0.3, 0.4]. Start theta = [0,0,0,0], so the model
is uniform: p = [0.25, 0.25, 0.25, 0.25].

Take data point x = 4. Positive phase: lower E(4), i.e. raise
theta_4. Run one MCMC step from x = 4 under the uniform model.
suppose it lands on face 2. Negative phase: raise E(2), i.e.
lower theta_2. With step size eta = 0.1:

```ascii
theta: [0, 0, 0, 0] -> [0, -0.1, 0, 0.1]

new unnormalized: [1.0000, 0.9048, 1.0000, 1.1052],  Z = 4.0100
new p = [0.2494, 0.2256, 0.2494, 0.2756]
```

One contrastive step moved mass from the fantasy (face 2: 0.25
-> 0.2256) to the data (face 4: 0.25 -> 0.2756). Repeat over the
dataset and the valleys migrate onto the data, one fantasy at a
time. The full-batch gradient behind it, for the record: with
theta = 0, the data term E_data[dE/dtheta] = -0.3 and the model
term E_model[dE/dtheta] = -0.5, giving gradient -(-0.3 + 0.5) =
-0.2 on the shared slope parameter, pushing the model away from
over-producing face 2's region. CD-1 approximates that model
term with the one-step fantasy.

Sampling uses the same downhill walk as L06's Langevin dynamics,
because the score is minus the energy gradient: s(x) = -grad
E(x). The critic's sampler and the score model's sampler are one
algorithm. Score models (L06) are EBMs that skipped learning E
and learned its gradient directly.

## Mapping back: what each property fixes

| Naive-critic failure | Contrastive-divergence answer | How |
|---|---|---|
| Z needs 2^1024 terms; training needs Z | Gradient as a contrast | Positive phase on data, negative phase on fantasies; Z cancels out of the difference |
| Model expectation needs perfect MCMC | CD-k: k-step chains from data | Short chains reach the model's current fantasies near data, where the contrast bites |
| No sampling rule | Downhill Langevin walk | x <- x - (delta/2) grad E + sqrt(delta) z; valleys attract |

## The honest price: biased, blind, and stuck

**Price 1: the gradient is biased, demonstrated.** CD-1's
negative sample came from a 1-step chain, not the model's true
distribution. A single fantasy estimates the model expectation
with huge variance: on the 2-outcome toy the one-sample estimate
of E_model[dE/dtheta] is either 0 or -1 against the true -0.5, a
variance of 0.25 per sample. Short chains plus few samples mean
noisy, biased gradients. The theory says CD-k is not even the
gradient of any fixed objective. It works in practice the way a
compass works in a storm.

**Price 2: no normalized probabilities.** Z was never computed,
so p(x) is unknown. The critic samples beautifully and cannot
answer "how likely is this image?" The warper (L04) keeps the
exact-density crown. The critic trades it away.

**Price 3: the sampler gets stuck, demonstrated.** The downhill
walk explores one valley at a time. If the landscape has two
valleys separated by a high ridge, the chain sits in one valley
for thousands of steps before crossing. Multimodal data means
multimodal valleys, and Langevin mixes slowly across them. The
negative samples then come from one mode only, and the contrast
teaches the energy about that mode alone. This is the sampler's
version of mode collapse, and tempering (running hot and cold
chains) is the standard, fiddly fix.

> [!QA]
> Q: What is the partition function, and why is it the whole problem?
> A: Z = sum_x exp(-E(x)) normalizes energies into probabilities. On the 4-face toy it is 1.8711, computed exactly. On a 32x32 binary image it has 2^1024 terms: at a billion energy evaluations per second, about 10^290 years. Training needs Z through the negative phase of the gradient, so the naive maximum-likelihood critic cannot even start.
> Follow-up: If Z is intractable, how does anything about EBMs work at all?
> A: The gradient splits into data term minus model term, and Z cancels in the difference. Contrastive divergence approximates the model term with short MCMC chains started at data points. You never need Z's value, only the direction in which data and fantasies disagree. The price is bias and variance, not impossibility.

> [!QA]
> Q: What is contrastive divergence, mechanically?
> A: For each data point: lower its energy (positive phase), run k MCMC steps starting from it (often k = 1), raise the landing spot's energy (negative phase). The worked CD-1 step moved theta from [0,0,0,0] to [0,-0.1,0,0.1], shifting probability from the fantasy face 2 (0.25 to 0.2256) to the data face 4 (0.25 to 0.2756). Learning stops when fantasies match data and the two phases cancel.
> Follow-up: Why start the chain at the data instead of from scratch?
> A: A chain from scratch needs thousands of steps to mix into the model's distribution. A chain from data starts where the contrast matters and one step already produces a useful fantasy. The bias this introduces is the method's known price: CD-k does not follow the gradient of any fixed objective, but the short-chain fantasies point the right way in practice.

> [!QA]
> Q: How are EBMs and score models related?
> A: The score is minus the energy gradient: s(x) = -grad E(x). Langevin sampling on energies (x <- x - (delta/2) grad E + sqrt(delta) z) is identical to L06's score Langevin. A score model is an EBM that skipped learning E and learned grad E directly, dodging Z the same way. The EBM keeps the energy (useful for composing models: energies add). The score model keeps only the slope.
> Follow-up: When would you prefer the energy over the score?
> A: When models must compose. Energies of independent constraints add: E_total = E_face + E_smiling gives a "smiling face" model with no retraining. Scores compose too (gradients add), but only energies give a (unnormalized) joint density to reason about. Compositionality is the critic's surviving superpower.

![Chapter plate: compare data against fantasies, the partition function cancels](assets/plate-l07.png "Chapter plate. Compare data against fantasies; the partition function cancels out. Source: original plate for Stanford Frontier AI.")

## Recap: the whole lesson on one screen

1. **The question, for valleys.** Learn the rule. Draw fresh samples. Sculpt an energy landscape: valleys for real, hills for noise.
2. **Energies to probabilities, by hand.** E = [2,1,1,0], Z = 1.8711, p = [0.0723, 0.1966, 0.1966, 0.5345]. Face 4 wins.
3. **The partition bill, demonstrated.** Z over 2^1024 image outcomes: ~10^290 years at a billion evals per second. Training's negative phase needs Z.
4. **The key question.** What if we train without Z, contrasting data against the model's fantasies?
5. **Contrastive divergence, by hand.** CD-1: theta [0,0,0,0] -> [0,-0.1,0,0.1]. Mass moves fantasy face 2 -> data face 4 (0.2256 vs 0.2756).
6. **The gradient behind it.** Data term -0.3, model term -0.5, net -0.2: push away from over-producing the fantasy region.
7. **Sampling.** Downhill Langevin. The score is -grad E, so L06's sampler is this sampler.
8. **The price.** Biased noisy gradients (variance 0.25 per one-sample estimate). No normalized probabilities. Chains get stuck in one valley.

## Official sources and further reading

**Official:**
- Hinton (2002), Training Products of Experts by Minimizing Contrastive Divergence: https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf — the CD algorithm.
- LeCun et al. (2006), A Tutorial on Energy-Based Learning: https://www.cs.nyu.edu/~yann/research/ebm.pdf — the energy framing this chapter follows.

**Further reading:**
- Du & Mordatch, Implicit Generation and Generalization in EBMs (2019): modern EBM image modeling with Langevin sampling.
- Grathwohl et al., Your Classifier is Secretly an Energy Based Model (2020): energies from classifiers.

**Caveats from these sources.** [uncertain]: energy-based models do not appear in the Prathosh playlist's topic sequence (weeks 1-12). This lesson is grounded in Hinton 2002 and LeCun et al. 2006. The toy chains are illustrative. Real CD runs k-step Langevin in high dimensions with persistent chains. CD-k's bias is analyzed in the literature. The "compass in a storm" characterization is this chapter's, not the papers'.

## Connections to the other courses

- **L06 (this course):** the score is -grad E. Annealed Langevin is the critic's sampler. Score models are EBMs without the energy.
- **L04 (this course):** the exact-density alternative: the warper computes Z's analog (the determinant) exactly. The critic gives up and samples.
- **math-genai (sibling):** the divergence view: CD minimizes the KL between data and model distributions approximately, one fantasy at a time.
- **CS229 L09:** mixture models as the tractable cousin: exact posteriors where the critic needs MCMC.
