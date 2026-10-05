---
page_id: math-ml-l06
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 6
nav: "L06 · Distributions and Covariance"
title: "Lecture 6: Distributions and Covariance"
summary: "The three distributions ML cannot live without, each with hand-worked numbers: Bernoulli, Binomial, Gaussian. Then covariance on three points, and why correlated features inflate variance."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [bernoulli, binomial, gaussian, normal-distribution, joint-distribution, covariance, mle]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 51 (Discrete probability distributions), 52 (Continuous probability distributions), 53 (Joint probability distribution and covariance)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Bishop, Pattern Recognition and Machine Learning, ch. 1.2, 2"
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 6"
    url: https://mml-book.github.io
---

## The task: name the shapes uncertainty takes

L05 gave the rules of probability. Now the vocabulary: the specific
distributions that appear in nearly every ML model. A
**distribution** assigns a probability to every possible value of a
random variable. Discrete variables get a **probability mass
function** (a list of probabilities that sum to 1). Continuous
variables get a **probability density function** (a curve; areas
under it are probabilities).

Three distributions cover most of ML. Each gets a worked example.

## Distribution 1: Bernoulli, the single yes/no

One trial, two outcomes. A coin flip, a click, spam or not spam.

```ascii
X ~ Bernoulli(p):   P(X=1) = p,   P(X=0) = 1-p

fair coin: p = 0.5.  biased ad click: p = 0.03.
E[X] = p.  Var(X) = p(1-p).
```

The variance formula is worth reading: p(1-p) is largest at p =
0.5 (maximum uncertainty) and zero at p = 0 or 1 (certainty). For
the ad with p = 0.03: variance = 0.03 * 0.97 = 0.0291. Rare events
have small variance because the answer is almost always "no".

## Distribution 2: Binomial, the count of yeses

Repeat n Bernoulli trials. Count the successes.

```ascii
K ~ Binomial(n, p):  P(K=k) = C(n,k) * p^k * (1-p)^(n-k)

10 emails, each spam with p = 0.2.  P(exactly 3 spam)?
C(10,3) = 120.  P = 120 * 0.2^3 * 0.8^7
       = 120 * 0.008 * 0.2097 = 0.201
```

About a 20% chance of exactly 3 spam emails. E[K] = np = 2,
Var(K) = np(1-p) = 1.6. The Binomial is the distribution behind every "k out of n" question: k conversions out
of n visitors, k defects out of n parts.

## Distribution 3: Gaussian, the bell curve

The **Gaussian** or **normal** distribution is the continuous
workhorse: measurement noise, heights, test scores, and the
initialization of every neural network weight.

```ascii
X ~ N(mu, sigma^2):  bell centered at mu, width set by sigma

68-95-99.7 rule: P(|X - mu| < sigma)   = 0.68
                 P(|X - mu| < 2 sigma) = 0.95
                 P(|X - mu| < 3 sigma) = 0.997
```

Human heights: mu = 170 cm, sigma = 10 cm. About 95% of people fall
between 150 and 190 cm. About 0.3% fall outside 140-200 cm. The
rule turns "two standard deviations" into a plain-English rarity
claim, which is why anomaly detectors threshold at 2 or 3 sigma.

Why is the Gaussian everywhere? The **central limit theorem**: a
sum of many small independent effects looks Gaussian no matter
what the pieces look like. Measurement error is the sum of many
tiny disturbances, so it is Gaussian. This is the deep reason the
normal distribution models noise across science and ML.

## Maximum likelihood: fit the distribution to data

You observe data. Which parameters make the data most probable?
That is **maximum likelihood estimation**, MLE: pick the parameters
that maximize P(data | parameters).

Bernoulli MLE, worked by hand. Five ad impressions, clicks on the
1st and 4th: data = [1, 0, 0, 1, 0]. The likelihood of p is

```ascii
L(p) = p * (1-p) * (1-p) * p * (1-p) = p^2 (1-p)^3
```

Try p = 0.4: L = 0.16 * 0.216 = 0.0346. Try p = 0.5: L = 0.25 *
0.125 = 0.0313. Try p = 0.3: L = 0.09 * 0.343 = 0.0309. The winner
near 0.4 is no accident: the MLE for Bernoulli is the sample mean,
2/5 = 0.4. Counting is estimating. This one fact underlies naive
Bayes training (CS229 L05): estimate each likelihood by counting.

## Joint distributions and covariance: when variables move together

A **joint distribution** P(X, Y) describes two variables at once.
From it you can read each variable alone (**marginal**) or one
given the other (**conditional**). The number that summarizes how
two variables move together is the **covariance**:

```ascii
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
```

Positive: above-average X comes with above-average Y. Negative:
they move oppositely. Zero: no linear relationship.

Three points, by hand: (1, 2), (2, 4), (3, 6). Means: E[X] = 2,
E[Y] = 4.

```ascii
Cov = [(1-2)(2-4) + (2-2)(4-4) + (3-2)(6-4)] / 3
    = [( -1)(-2) + 0 + (1)(2)] / 3
    = (2 + 0 + 2) / 3 = 1.33
```

Positive, as expected: the points lie on a rising line. Now the
L05 follow-up made concrete: Var(X + Y) = Var(X) + Var(Y) + 2
Cov(X, Y). Here Var(X) = 2/3, Var(Y) = 8/3, so Var(X+Y) = 2/3 +
8/3 + 2*1.33 = 6. The covariance term contributed nearly half.
Correlated features inflate the variance of their sum. Portfolio
managers and ML regularizers both fight this term.

| Distribution | Models | Key numbers |
|---|---|---|
| Bernoulli(p) | one yes/no | mean p, var p(1-p) |
| Binomial(n, p) | count of yeses in n trials | mean np, var np(1-p); toy: P(3 spam of 10) = 0.201 |
| Gaussian(mu, sigma^2) | continuous noise, measurements | 68-95-99.7 rule; toy: 95% of heights in 150-190 cm |
| Covariance | joint movement | toy: 1.33 on three collinear points |

> [!QA]
> Q: Why do ML courses keep assuming Gaussian noise?
> A: Two reasons. The central limit theorem: sums of many small independent effects look Gaussian, and measurement error is such a sum. And the math is kind: the Gaussian is fully described by mean and variance, its MLE has a closed form (sample mean, sample variance), and Gaussians stay Gaussian under linear maps. In the toy, heights N(170, 100) give the 95% interval 150-190 cm directly from the 68-95-99.7 rule.
> Follow-up: What breaks if the noise is not Gaussian?
> A: Least squares stops being the MLE. Squared error is the MLE only under Gaussian noise; with heavy-tailed noise (outliers), it chases the outliers. That is why robust regression swaps the square for the absolute value. The loss function encodes a noise assumption: always ask which one.

> [!QA]
> Q: What is maximum likelihood estimation, concretely?
> A: Pick the parameters that make the observed data most probable. Five ad impressions with 2 clicks: the likelihood p^2 (1-p)^3 peaks at p = 0.4, the sample mean. For Bernoulli, MLE is counting. For Gaussian, MLE is the sample mean and sample variance. Training a generative model is usually MLE wearing fancier clothes.
> Follow-up: Is the MLE always the sample mean?
> A: No. It is the sample mean for Bernoulli and Gaussian means because their likelihoods peak there. For other distributions it differs: the MLE of a uniform [0, theta] distribution's endpoint is the maximum observed value, not any mean. Always derive it from the likelihood; never guess.

> [!QA]
> Q: Covariance is positive. So what?
> A: It means above-average X travels with above-average Y, linearly. In the toy, three points on a rising line give Cov = 1.33. The practical bite: Var(X+Y) includes +2 Cov(X,Y), so correlated features inflate variance. That inflation is why L01's dependent features are dangerous and why regularization (CS229 L08) penalizes large weights on correlated inputs.
> Follow-up: Does zero covariance mean independence?
> A: No. Zero covariance means no *linear* relationship. Y = X^2 with symmetric X has zero covariance but perfect dependence: knowing X determines Y exactly. Independence implies zero covariance; the reverse fails. Interviewers love this trap.

## Recap: the whole lesson on one screen

1. **The task.** Name the shapes uncertainty takes: mass functions for discrete, densities for continuous.
2. **Bernoulli.** One yes/no: P(X=1) = p. Ad click p = 0.03, variance 0.0291.
3. **Binomial.** Count of yeses: P(exactly 3 spam of 10) = 0.201, computed term by term.
4. **Gaussian.** The bell: 68-95-99.7 rule; 95% of N(170,100) heights fall in 150-190 cm. Central limit theorem explains its ubiquity.
5. **MLE.** Parameters that maximize P(data | params). Bernoulli toy: p^2(1-p)^3 peaks at the sample mean 0.4. Counting is estimating.
6. **Covariance.** E[(X-mean)(Y-mean)]: 1.33 on three collinear points. Correlated features inflate Var(X+Y) through the +2Cov term.
7. **The price.** Every distribution is an assumption. Gaussian MLE gives least squares; wrong noise model gives wrong loss.
8. **The bridge.** Distributions describe data; calculus (L07) optimizes over parameters; together they train models.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 51-53:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Bishop, "Pattern Recognition and Machine Learning", ch. 1.2, 2 — Bernoulli, Binomial, Gaussian, MLE.
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 6 (free):
  https://mml-book.github.io — distributions, covariance.

**Caveats.** The lecture titles promise discrete, continuous, and joint distributions with covariance; the specific worked examples (spam emails, heights, ad clicks) are the lesson's own. Exact lecture examples are [uncertain].

## Connections to the other courses

- **CS229 L05:** naive Bayes trains by MLE counting; Gaussian discriminant analysis fits Gaussians per class.
- **CS229 L02:** least squares is Gaussian MLE; the 68-95-99.7 rule justifies sigma-based anomaly thresholds.
- **CS336:** weight initialization samples Gaussians; the central limit theorem explains why sums of weights stay controlled.
