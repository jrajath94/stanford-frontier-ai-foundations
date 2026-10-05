---
page_id: math-ml-l05
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 5
nav: "L05 · Probability and Bayes"
title: "Lecture 5: Probability and Bayes' Theorem"
summary: "Uncertainty as arithmetic: conditional probability, a medical-test Bayes update worked to 8.7 percent, and expectation and variance on a loaded die."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [probability, conditional-probability, bayes-theorem, expectation, variance, random-variable]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 47 (Operations on Sets), 48 (Review on Probability), 49 (Bayes' theorem and Random variables), 50 (Expectation and Variance)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Bishop, Pattern Recognition and Machine Learning, ch. 1"
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 6"
    url: https://mml-book.github.io
---

## The task: reason under uncertainty

Every ML prediction is a bet. The model sees symptoms and bets on
the disease. It sees an email and bets on spam. It sees pixels and
bets on "cat". The mathematics of betting correctly is
**probability**: numbers between 0 and 1 that obey three rules.

A **random variable** is a quantity whose value is uncertain: the
outcome of a die roll, tomorrow's temperature, whether an email is
spam. Write X for the variable and x for a value it takes. P(X = x)
is the probability it takes that value. The three rules:

```ascii
1. P(anything) is between 0 and 1.
2. P(all possibilities) = 1.  Something happens.
3. P(A or B) = P(A) + P(B) when A and B cannot both happen.
```

That is the whole foundation. Everything in this lesson is built
on these three.

## First attempt: trust the test result

A patient tests positive for a rare disease. The test is 95%
accurate. How likely is the patient sick?

The gut answer: 95%. This is the trap the lesson exists to fix.
The test's accuracy is P(positive | sick): the chance of a positive
* given * sickness. The question asks the reverse: P(sick |
positive). Confusing the two is the **base-rate fallacy**, and it
is the most expensive probability mistake in applied ML.

**Conditional probability** P(A | B) means the probability of A
given that B happened: restrict the world to B, then measure A.

```ascii
P(A | B) = P(A and B) / P(B)
```

## Where the gut breaks: the base rate dominates

Run the numbers. 10,000 people. 1% carry the disease: 100 sick,
9,900 healthy.

```ascii
sick and test positive:    100 * 0.95 = 95
healthy but test positive: 9,900 * 0.10 = 990   (10% false alarm rate)

total positives: 95 + 990 = 1,085
P(sick | positive) = 95 / 1,085 = 0.0876 = 8.8%
```

A positive result on a 95%-accurate test means an 8.8% chance of
being sick. The 990 false alarms drown the 95 true hits because
the disease is rare. The base rate, the 1%, dominated the test's
95% accuracy. Anyone deploying a classifier on rare events who
ignores this ships a false-alarm machine.

## The key question

How do you flip a conditional probability: turn P(evidence |
hypothesis), which tests and models give you, into P(hypothesis |
evidence), which is what you actually want?

## The new idea: Bayes' theorem

**Bayes' theorem** is the flip:

```ascii
P(H | E) = P(E | H) * P(H) / P(E)

P(H):       prior.      Belief before the evidence. (1% sick)
P(E | H):   likelihood. How data looks if H is true. (95%)
P(E):       evidence.   Total chance of this evidence.
P(H | E):   posterior.  Belief after. (8.8%)
```

Name each piece every time you use it. The prior is your starting
belief. The likelihood is the model's job: it predicts what data
each hypothesis produces. The posterior is the updated belief.
Learning, in the Bayesian view, is prior times likelihood,
renormalized.

In ML this is the engine of **naive Bayes** (CS229 L05): the prior
is how common each class is, the likelihood is how each class
generates features, and classification picks the class with the
highest posterior. "Naive" because it assumes features are
independent given the class, which factorizes one big likelihood
into a product of small ones.

## Expectation and variance: summarize a random variable in two numbers

A random variable's full distribution can be complex. Two numbers
summarize most of what you need.

**Expectation**, E[X], is the long-run average. For values x_i with
probabilities p_i: E[X] = sum of x_i * p_i.

**Variance**, Var(X), is the average squared distance from the
mean: E[(X - E[X])^2]. It measures spread. Its square root is the
**standard deviation**.

A loaded die, worked by hand. Faces 1-5 are fair-ish, face 6 is
heavy:

```ascii
P(1..5) = 0.1 each,  P(6) = 0.5

E[X] = 1*.1 + 2*.1 + 3*.1 + 4*.1 + 5*.1 + 6*.5
     = 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 3.0 = 4.5

Var(X) = E[(X-4.5)^2]
  = .1*(1-4.5)^2 + .1*(2-4.5)^2 + .1*(3-4.5)^2
    + .1*(4-4.5)^2 + .1*(5-4.5)^2 + .5*(6-4.5)^2
  = .1*12.25 + .1*6.25 + .1*2.25 + .1*0.25 + .1*0.25 + .5*2.25
  = 1.225 + 0.625 + 0.225 + 0.025 + 0.025 + 1.125 = 3.25
```

Mean 4.5 (above the fair die's 3.5, pulled by the heavy 6),
variance 3.25. Two numbers capture the loading.

Three expectation facts the ML courses use constantly:

```ascii
E[aX + b] = a E[X] + b            (linearity: constants pass through)
E[X + Y] = E[X] + E[Y]            (sums split, always, no independence needed)
Var(X) = E[X^2] - (E[X])^2        (the computational shortcut)
```

Linearity of expectation needs no independence assumption. That
is why it appears in every other proof in ML: you can always split
an expectation of a sum.

| Idea | Formula | Role |
|---|---|---|
| Conditional probability | P(A\|B) = P(A,B)/P(B) | restrict the world, then measure |
| Bayes' theorem | P(H\|E) = P(E\|H)P(H)/P(E) | flip likelihood into posterior |
| Prior / likelihood / posterior | belief before / evidence model / belief after | the vocabulary of learning |
| Expectation | sum of x_i * p_i | long-run average; splits over sums |
| Variance | E[(X - mean)^2] | spread; sqrt is standard deviation |

> [!QA]
> Q: A test is 95% accurate and I tested positive. Am I 95% likely to be sick?
> A: No: about 8.8% in the worked example. The 95% is P(positive | sick); you want P(sick | positive). With 1% prevalence, 100 sick people yield 95 true positives but 9,900 healthy people yield 990 false alarms, so 95 out of 1,085 positives are truly sick. The base rate dominates. This is the base-rate fallacy, and it bites every rare-event classifier.
> Follow-up: How do you fix it in a real classifier?
> A: Two levers. Raise the decision threshold so fewer healthy cases trigger alarms, or gather more evidence (a second independent test multiplies the likelihoods). Naive Bayes handles it automatically: the prior P(spam) is small, so the posterior stays small unless the likelihood ratio is overwhelming.

> [!QA]
> Q: What do prior, likelihood, and posterior actually mean?
> A: The prior is what you believed before seeing data: 1% disease prevalence. The likelihood is how each hypothesis generates data: 95% of sick patients test positive. The posterior is the updated belief: 8.8% after a positive test. Bayes' theorem multiplies prior by likelihood and renormalizes. Training a generative classifier is estimating likelihoods; predicting is computing posteriors.
> Follow-up: Where does P(E), the denominator, come from?
> A: It is the total probability of the evidence under all hypotheses: P(E) = sum over H of P(E|H) P(H). In the toy: 95 true positives + 990 false alarms, over 10,000 people. It just rescales so the posteriors sum to 1. When comparing two hypotheses you can often ignore it, since it is the same for both.

> [!QA]
> Q: Why is linearity of expectation such a big deal?
> A: Because E[X + Y] = E[X] + E[Y] holds with zero assumptions: no independence needed. Most quantities in ML are sums or averages: the loss over a dataset, the gradient over a batch, the return over an episode. Linearity lets you push expectations inside sums everywhere, which is why proofs about stochastic gradient descent and generalization start with it.
> Follow-up: Does Var(X + Y) = Var(X) + Var(Y)?
> A: Only if X and Y are uncorrelated. In general Var(X+Y) = Var(X) + Var(Y) + 2 Cov(X,Y). The extra term is why L06 exists: correlated features inflate variance, and covariance measures exactly that inflation.

## Recap: the whole lesson on one screen

1. **The task.** Bet correctly under uncertainty: three rules, random variables, probabilities.
2. **First attempt.** Trust the 95%-accurate test: a positive means 95% sick.
3. **Where it breaks.** Base rate: 990 false alarms drown 95 true hits. Posterior is 8.8%, not 95%.
4. **The key question.** How to flip P(evidence | hypothesis) into P(hypothesis | evidence)?
5. **The new idea.** Bayes' theorem: posterior = likelihood times prior, renormalized. Name the pieces every time.
6. **The ML engine.** Naive Bayes (CS229 L05): priors over classes, factorized likelihoods over features, classify by highest posterior.
7. **Summarize in two numbers.** Expectation (loaded die: 4.5) and variance (3.25), computed by hand.
8. **The price and the bridge.** Bayes needs a prior and a likelihood; bad priors or wrong likelihoods give confident wrong answers. L06 builds the distributions those likelihoods come from.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 47-50:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Bishop, "Pattern Recognition and Machine Learning", ch. 1 — probability rules, Bayes, expectation.
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 6 (free):
  https://mml-book.github.io — discrete and continuous probability.

**Caveats.** The medical-test numbers and the loaded die are the lesson's own worked examples, in the standard textbook style. The lecture's exact examples are [uncertain] (transcripts for L47-L50 were not recovered). The lecture-1 transcript confirms the course frames probability as underpinning all ML algorithms.

## Connections to the other courses

- **CS229 L05:** naive Bayes classifier: priors, likelihoods, the independence assumption, Laplace smoothing.
- **CS229 L08:** the Bayesian view of regularization: priors over weights become penalty terms.
- **CS329H:** choice theory and Bayesian decision theory build directly on posterior reasoning.
