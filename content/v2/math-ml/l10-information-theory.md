---
page_id: math-ml-l10
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 10
nav: "L10 · Information Theory"
title: "Lecture 10: Information Theory Basics"
summary: "How many bits does a prediction cost? Entropy, KL divergence, and cross-entropy worked by hand on coins and classifiers, ending at the loss every neural net minimizes."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [entropy, kl-divergence, cross-entropy, bits, surprise]
sources:
  - tag: supplement
    label: "MacKay, Information Theory, Inference, and Learning Algorithms, ch. 1-2 (free)"
    url: http://www.inference.org.uk/itila/
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 6.6"
    url: https://mml-book.github.io
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 48-53 (Probability) as prerequisite"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
---

## [uncertain] A supplement beyond the playlist

The playlist has no dedicated information theory lecture: its
probability block (L48-L53) stops at joint distributions and
covariance. This lesson is a synthesis from standard references,
marked [uncertain] throughout. It earns its place because one
quantity, **cross-entropy**, is the loss function of logistic
regression (L08), softmax classifiers, and every language model in
this system (CS229 L08, CS336). You cannot read ML without it.

## The task: price a prediction in bits

A model predicts the next word. How do you score the prediction?
"Right or wrong" is too coarse: a model that assigned 49% to the
right word is better than one that assigned 1%. You need a score
that rewards good probabilities, not just good guesses.

**Information theory** prices uncertainty in **bits**. One bit is
the information in the answer to one yes/no question. The core
idea: surprising events carry more information than expected ones.
A fair coin flip tells you 1 bit. A double-headed coin flip tells
you 0 bits: you knew the answer.

## Surprise, then entropy

Define the **surprise** of an event with probability p as
-log2(p). A fair coin's heads: -log2(0.5) = 1 bit. A 1-in-1024
event: 10 bits. An impossible event would be infinitely
surprising, which is why models must never assign probability
exactly zero to anything possible.

**Entropy** is the expected surprise: the average bits per outcome.

```ascii
H(X) = -sum of p(x) * log2 p(x)

fair coin:  H = -(0.5*log2 0.5 + 0.5*log2 0.5) = -(-0.5 - 0.5) = 1 bit
```

A biased coin, p(heads) = 0.25, worked by hand:

```ascii
log2 0.25 = -2,  log2 0.75 = -0.415
H = -(0.25 * -2 + 0.75 * -0.415) = -(-0.5 - 0.311) = 0.811 bits
```

Less than 1 bit: the bias makes outcomes more predictable, so each
flip teaches you less. Entropy is maximized by the uniform
distribution and minimized (zero) by certainty. It measures how
much you do not know.

## KL divergence: the price of the wrong distribution

You believe the coin is fair: q = [0.5, 0.5]. It is actually
biased: p = [0.25, 0.75]. How much does your wrong belief cost?
The **KL divergence** measures the extra bits you pay for coding
data from p with q's expectations:

```ascii
KL(p || q) = sum of p(x) * log2(p(x) / q(x))

p = [0.25, 0.75], q = [0.5, 0.5]:
KL = 0.25 * log2(0.25/0.5) + 0.75 * log2(0.75/0.5)
   = 0.25 * (-1) + 0.75 * 0.585
   = -0.25 + 0.439 = 0.189 bits
```

Your wrong model costs 0.189 extra bits per flip. Two properties
matter. First, KL >= 0 always, and KL = 0 only when p = q: you
cannot beat the truth. Second, it is **not symmetric**: KL(p||q)
!= KL(q||p). It measures "wrongness of q when the world is p",
direction included. Treating it as a distance is the classic
mistake.

## Cross-entropy: the loss function

**Cross-entropy** is entropy plus the KL price:

```ascii
H(p, q) = H(p) + KL(p || q) = -sum of p(x) * log2 q(x)
```

Read it: the true distribution is p, you predict q, and you pay
the average surprise of your predictions on real data. In
classification, p is the one-hot truth ([1, 0] for "cat") and q is
the model's probabilities ([0.7, 0.3]):

```ascii
loss = -(1 * log2 0.7 + 0 * log2 0.3) = -log2 0.7 = 0.515 bits
```

Confident and right ([0.99, 0.01]): -log2 0.99 = 0.014 bits.
Confident and wrong ([0.01, 0.99]): -log2 0.01 = 6.64 bits. The
loss explodes for confident errors: that is the behavior you want
in a loss. This is exactly L08's logistic loss, now in bits. Every
softmax classifier and every language model minimizes
cross-entropy between the true next token and the predicted
distribution (CS336, CS229 L08).

## Where it breaks: zero probabilities and infinite bills

If q assigns 0 to an event that occurs, -log2(0) is infinite: one
impossible prediction bankrupts the whole average. Real systems
defend with **smoothing**: mix a little uniform distribution into
q so no probability hits exactly zero. Laplace smoothing in naive
Bayes (CS229 L05) is this defense. The second crack: cross-entropy
only cares about the probability assigned to the true class. A
model can be perfectly calibrated on truth-class probabilities and
wild elsewhere; calibration metrics exist to catch that.

| Idea | Formula | Meaning |
|---|---|---|
| Surprise | -log2 p | bits in one outcome; rarer = more bits |
| Entropy | -sum p log2 p | expected surprise; fair coin = 1, biased 0.25-coin = 0.811 |
| KL divergence | sum p log2(p/q) | extra bits for believing q when truth is p; toy: 0.189 |
| Cross-entropy | -sum p log2 q | the loss: truth p, prediction q; confident-wrong = 6.64 bits |
| Smoothing | mix in uniform | never let q hit exactly zero |

> [!QA]
> Q: What is entropy, in plain words?
> A: The average surprise per outcome, in bits. A fair coin has entropy 1: each flip teaches you one bit. A coin with p(heads) = 0.25 has entropy 0.811: the bias makes flips more predictable, so each teaches less. Entropy is highest for uniform distributions and zero when the outcome is certain. It measures how much you do not know.
> Follow-up: Why log base 2?
> A: Because the unit is the bit: one yes/no question. log2(0.5) = -1, so a 50/50 event carries exactly 1 bit. Natural log gives nats, base 10 gives dits: same idea, different ruler. ML uses nats (natural log) in code and bits in explanations; the math is identical up to a constant.

> [!QA]
> Q: Why is cross-entropy the standard classification loss?
> A: Because it prices the model's predicted probabilities on real data: -sum of p log q. Truth [1, 0], prediction [0.7, 0.3]: loss 0.515 bits. Prediction [0.01, 0.99]: loss 6.64 bits. Confident wrong answers pay explosively, which is exactly the incentive you want. Minimizing cross-entropy is minimizing KL(p||q), since H(p) is constant: it drives the model's distribution toward the truth.
> Follow-up: What is the difference between KL divergence and cross-entropy?
> A: Cross-entropy = entropy of truth + KL divergence: H(p,q) = H(p) + KL(p||q). Since H(p) does not depend on the model, minimizing one minimizes the other. KL isolates the model's wrongness (0.189 bits in the coin toy); cross-entropy is the total bill including irreducible uncertainty.

> [!QA]
> Q: Is KL divergence a distance?
> A: No. It is nonnegative and zero only when the distributions match, which tempts the word "distance", but it is not symmetric: KL(p||q) differs from KL(q||p). It has a direction: "cost of believing q when the world is p". Symmetric versions exist (Jensen-Shannon), but plain KL's asymmetry is a feature: in variational inference the direction you choose changes the answer.
> Follow-up: What happens with a zero predicted probability?
> A: Infinite loss: -log2(0) is undefined. One impossible prediction bankrupts the average. Defend with smoothing: mix a small uniform component into q. This is why naive Bayes uses Laplace smoothing and why softmax never outputs exact zeros.

## Recap: the whole lesson on one screen

1. **The task.** Score predicted probabilities, not just right/wrong guesses.
2. **Surprise.** -log2(p): rarer events carry more bits.
3. **Entropy.** Expected surprise: fair coin 1 bit; 0.25-biased coin 0.811 bits, computed term by term.
4. **KL divergence.** Extra bits for the wrong belief: 0.189 bits for q = fair when p = [0.25, 0.75]. Nonnegative, asymmetric, not a distance.
5. **Cross-entropy.** H(p) + KL: the loss. Confident-right 0.014 bits, confident-wrong 6.64 bits. This is logistic regression's loss and every LM's loss.
6. **Where it breaks.** Zero predicted probability means infinite loss: smooth. Cross-entropy ignores non-truth classes: check calibration.
7. **[uncertain]** The playlist has no info-theory lecture; this lesson synthesizes MacKay and the MML book.
8. **The bridge.** The math arc is complete: vectors (L01) hold data, matrices (L02) transform it, eigen/SVD (L03-L04) compress it, probability (L05-L06) models uncertainty, calculus (L07) differentiates, convexity (L08) guarantees, least squares (L09) fits, information theory prices predictions.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 48-53 (probability prerequisite):
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- MacKay, "Information Theory, Inference, and Learning Algorithms", ch. 1-2 (free):
  http://www.inference.org.uk/itila/ — entropy, KL, cross-entropy from zero.
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 6.6 (free):
  https://mml-book.github.io — information theory for ML.

**Caveats.** This entire lesson is [uncertain] with respect to the playlist: no information-theory lecture exists in it. All numbers are the lesson's own, verified by hand. Content follows MacKay's standard treatment.

## Connections to the other courses

- **CS229 L08:** softmax + cross-entropy: the loss every neural classifier minimizes.
- **CS336:** language modeling IS next-token cross-entropy minimization; perplexity is 2^(cross-entropy).
- **CS229 L05:** logistic regression's loss is binary cross-entropy (L08 of this course).
