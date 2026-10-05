---
page_id: cs229-l03
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 3
nav: "L03 · Logistic Regression"
title: "Lecture 3: Maximum Likelihood and Logistic Regression"
summary: "MLE as the bedrock framework, Gaussian noise recovering least squares, the sigmoid, and Newton's method becoming weighted least squares."
date: "2026-04-13"
instructor: "Chris Ré"
offering: "Spring 2026"
duration: "1:02:06"
video_id: uJF_gL3jhxI
video_title: "Lecture 3: Weighted Least Squares"
video_caption: "Original lecture. Chris Ré builds the MLE framework, derives logistic regression, and shows Newton's method is iteratively reweighted least squares."
concepts: [MLE, maximum-likelihood, Gaussian-noise, logistic-regression, sigmoid, Bernoulli, Newton-method, IRLS, weighted-least-squares, logits]
sources:
  - tag: video
    label: "Lecture 3 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=uJF_gL3jhxI
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** contains what you need to
understand everything that follows in CS229 and the courses that build on
it. **Level 2 (Deep)** contains what you need for correct, interview-grade
understanding. Read Level 1 straight through. Return to Level 2 when you
want depth.

No prerequisites are assumed. Every term is defined at first use. Loss
functions were defined in [lecture 2](l02-linear-regression.html); the loss
chip is reused, not re-explained.

## Level 1: Maximum likelihood, the bedrock

Last lecture picked the squared loss by hand. This lecture derives it. The
framework is **maximum likelihood estimation** (MLE). Chris Ré calls it
the bedrock [02:44](ts:02:44). The recipe: write a probabilistic story of
how the data was generated, then choose the parameters that make the
observed data most probable.

**Likelihood** L(theta) is the probability the model assigns to the actual
training data. For independent examples, independence means product: the
joint probability is the product of the per-example probabilities
[25:11](ts:25:11), [26:00](ts:26:00). **Log likelihood** takes the log of
that product. Logs turn products into sums, which are stable and easy to
differentiate [27:39](ts:27:39). "We love logarithms." Maximizing the log
likelihood is the same as maximizing the likelihood, because log is
monotone.

The payoff is immediate. Assume the truth is linear plus Gaussian noise:
y = theta^T x + epsilon, with epsilon drawn from a normal distribution.
Write the likelihood, take the log, and maximize. The result is exactly
least squares [02:58](ts:02:58). The loss from lecture 2 was not a guess.
It is what MLE produces under Gaussian noise.

![MLE recovers least squares](assets/svg/l03-mle.svg "Model y = theta^T x + noise. Likelihood = product. Max log-likelihood = least squares. Original plate.")

> [!QA]
> Q: What is maximum likelihood estimation?
> A: Pick the parameters that make the observed data most probable. Write p(data; theta), take the log to turn the product into a sum, and maximize. It generalizes across continuous and discrete outputs and underlies modern AI training. The lecture calls it the bedrock because so many losses are MLE in disguise.
> Follow-up: Why take the log instead of maximizing the product directly?
> A: Three reasons. Products of many small probabilities underflow to zero in floating point. Sums differentiate term by term. And log is monotone, so the maximizer is unchanged. You lose nothing and gain stability.

## Level 1: From regression to classification

Regression predicts numbers. **Classification** predicts categories: spam
or not, cat or dog. Fitting a line to category labels is crazy. A line
predicts 7 or -40 for a 0-or-1 label, and then you need outside code to
round the nonsense back into classes.

Keep the linear score theta^T x, but pass it through a **link function**
g that squeezes it into (0, 1). Requirements: monotone and smooth. A step
function would give the right shape but it is not differentiable
[39:21](ts:39:21), so gradient methods die on it. The smooth choice is the
**sigmoid**:

g(z) = 1 / (1 + e^(-z))

![The sigmoid](assets/svg/l03-sigmoid.svg "S-shaped curve from 0 to 1. Predict 1 when theta^T x >= 0. Original plate.")

**Logistic regression** sets h_theta(x) = g(theta^T x) and reads the
output as a probability: P(y=1 | x) [39:06](ts:39:06). Height is
confidence. The model never says exactly 1; it says 0.999. Predict class
1 when h >= 0.5, which happens exactly when theta^T x >= 0. The quantity
theta^T x also has a name: **logits**, the raw scores before squashing.
You will meet logits again in every neural network.

The name is a historical accident. Logistic regression is classification,
not regression. Remember that when an interviewer asks why it is called
regression. The answer is history, not mathematics.

> [!QA]
> Q: Why not just fit a line and round the output for classification?
> A: A line extrapolates past 0 and 1, so its outputs are not probabilities and cannot be trusted as confidences. Rounding needs an arbitrary threshold with no probabilistic meaning. The sigmoid keeps every output in (0, 1) with a clean reading: the height is P(y=1|x). You get decisions and calibrated confidence from one model.
> Follow-up: Where does logistic regression appear in modern models?
> A: Everywhere. Chris Ré: logistic regression is the last layer of every model you have interacted with [01:22](ts:01:22). ChatGPT's final layer is a softmax, which is logistic regression generalized to many classes. Lecture 4 proves that connection.

## Level 1: Fitting it with Newton

The log likelihood for logistic regression has no closed form. Two
optimizers are on the table. **Gradient ascent** steps uphill on the log
likelihood (ascent because we maximize now) [44:16](ts:44:16). **Newton's
method** is faster and needs no step size.

Newton's idea: approximate the function locally by its tangent line, then
jump to where the tangent hits zero. In one dimension, theta := theta -
f(theta)/f'(theta). In many dimensions, division becomes inversion of the
**Hessian**, the matrix of second derivatives. No alpha. The method reads
the curvature and tells you exactly how far to go.

![Newton vs gradient descent](assets/svg/l03-newton.svg "GD takes many small steps. Newton reads curvature and jumps. No step size. Original plate.")

Applied to the logistic log likelihood, Newton's method has a beautiful
identity: each Newton step solves a **weighted least squares** problem.
That is **iteratively reweighted least squares** (IRLS), and it is why
the video is titled "Weighted Least Squares." The weights change every
iteration, emphasizing the examples the model currently gets wrong.

> [!QA]
> Q: Newton or gradient descent: which do you pick?
> A: Newton converges quadratically once close: correct digits roughly double per step. But each step inverts the Hessian, which costs cubic time in the number of parameters. Small parameter counts: Newton wins. Millions of parameters: gradient methods win, because you cannot afford the Hessian. That tradeoff decides the optimizer for every model in this course.
> Follow-up: Why does Newton's method need no learning rate?
> A: The second-order Taylor approximation already encodes how far to step. The curvature tells the method the distance to the bottom of the local quadratic. Gradient descent only knows the slope, so it needs alpha to guess the distance.

## Level 2: The probabilistic story, carefully

MLE has three moving parts. The **forward model** p(y | x; theta) says how
data is generated given parameters. The **iid assumption** says examples
are independent and identically distributed, which justifies the product.
The **estimator** argmax_theta log L(theta) picks the winner.

Each part can fail. If the forward model is wrong, MLE converges to the
wrong answer confidently. If examples are not independent, the product
overcounts evidence. If the likelihood has many local maxima, the
optimizer may not find the global one. Lecture 2's SGD assumptions were a
special case of this: the training set must reflect the world, and
minibatches must reflect the training set.

For logistic regression the forward model is the **Bernoulli**
distribution: y is 1 with probability h_theta(x), 0 otherwise. The log
likelihood becomes a sum over examples of y log h + (1-y) log (1-h).
Negate it and you get the **cross-entropy** loss, derived properly in
lecture 4. The loss chip from lecture 2 gets a new formula. Same chip,
new contents.

## Level 2: Why the sigmoid and not something else

The sigmoid is not arbitrary. It is the canonical link for binary
outcomes, and lecture 4 shows it falls out of the exponential family
automatically. For now, the lecture's justification is pragmatic:
monotone, smooth, maps the real line to (0, 1), differentiable everywhere
with a derivative that is easy to compute: g'(z) = g(z)(1 - g(z)).

That derivative matters. In backpropagation (lecture 8), every sigmoid in
the network contributes this factor. Its maximum is 0.25, at z = 0.
Products of many such factors shrink toward zero in deep networks. That
shrinkage has a name, the vanishing gradient problem, and it is one
reason lecture 7 prefers ReLU. The seed is planted here.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/svg/l03-mle.svg" alt="MLE recovers least squares">
<div class="rc-body">
<strong>1. MLE is the bedrock</strong>
<p>Write p(data; theta), take the log, maximize. Products become sums.
Log is monotone, so the maximizer is unchanged. Generalizes across
output types.</p>
<p class="rc-num">Key: argmax log L(theta)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-mle.svg" alt="Gaussian noise gives least squares">
<div class="rc-body">
<strong>2. Gaussian noise gives least squares</strong>
<p>Assume y = theta^T x plus normal noise. MLE recovers exactly the
squared loss of lecture 2. The loss was derived, not guessed.</p>
<p class="rc-num">Key: Gaussian + MLE = squares</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-sigmoid.svg" alt="The sigmoid">
<div class="rc-body">
<strong>3. Classification needs a link function</strong>
<p>Keep theta^T x, squash with g. Monotone and smooth. A step function is
not differentiable, so gradients die on it.</p>
<p class="rc-num">Key: squash (0,1), stay smooth</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-sigmoid.svg" alt="Logistic regression">
<div class="rc-body">
<strong>4. Logistic regression outputs probabilities</strong>
<p>h = 1/(1+e^(-theta^T x)) = P(y=1|x). Predict 1 when theta^T x >= 0.
The name is history. The last layer of every model you use.</p>
<p class="rc-num">Key: height = confidence</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-newton.svg" alt="Newton vs GD">
<div class="rc-body">
<strong>5. Newton reads curvature</strong>
<p>Theta := theta - f/f'. No step size. Quadratic convergence near the
answer. Each step inverts the Hessian: cubic cost.</p>
<p class="rc-num">Key: fast, needs the Hessian</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-newton.svg" alt="IRLS">
<div class="rc-body">
<strong>6. Newton on logistic loss is IRLS</strong>
<p>Each Newton step solves a weighted least squares problem. Weights
update each round, emphasizing current mistakes. Hence the video
title.</p>
<p class="rc-num">Key: iteratively reweighted least squares</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-mle.svg" alt="IID means product">
<div class="rc-body">
<strong>7. Independence means product</strong>
<p>IID justifies multiplying per-example probabilities. Logs turn the
product into a sum. Break independence and you overcount evidence.</p>
<p class="rc-num">Key: iid -> product -> log -> sum</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l03-sigmoid.svg" alt="Logits">
<div class="rc-body">
<strong>8. Logits are the raw scores</strong>
<p>Theta^T x before the sigmoid. Every neural network ends in logits.
Sigmoid derivative g(1-g) peaks at 0.25: the seed of vanishing
gradients.</p>
<p class="rc-num">Key: logits in, probabilities out</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 3 video: MLE bedrock [02:44](ts:02:44), product from independence [25:11](ts:25:11), sigmoid [39:06](ts:39:06), Newton [01:56](ts:01:56).
- CS229 Spring 2026 official course notes: the MLE and logistic regression chapters.

**Further reading:**
- Murphy, Probabilistic Machine Learning: An Introduction, Chapter 2: MLE with worked Bernoulli and Gaussian examples.
- Nelder and Wedderburn (1972): the original generalized linear models paper, for the historical arc into lecture 4.

**Caveats from these sources.** Newton's method can diverge far from the
optimum; practical implementations add line search or trust regions. IRLS
weights can blow up on perfectly separable data; that is a feature (the
likelihood is unbounded) that needs regularization, covered in lecture 6.
The "last layer of every model" claim is about the functional form, not a
claim that every deployment literally runs logistic regression code.

## Connections to the other courses

- **CS336:** the cross-entropy loss of language modeling is the multiclass MLE derived here; logits and softmax carry over unchanged.
- **CS224N:** binary sentiment classifiers are logistic regression on word features.
- **CS329H:** Bradley-Terry preference models are logistic regression on pairs.

> [!CHEAT]
> **MLE and logistic regression cheatsheet.** MLE: argmax log L(theta); iid gives product; log gives sum. Gaussian noise + MLE = least squares. Classification: keep theta^T x, squash with sigmoid g(z) = 1/(1+e^-z); h = P(y=1|x); predict 1 iff theta^T x >= 0. Logits = raw scores. Fit: gradient ascent or Newton; Newton = IRLS = weighted least squares per step; no step size; cubic cost per step. Name is historical: it classifies.

> [!MEMORY]
> **Bedrock.** Whenever a loss looks arbitrary, ask what probabilistic story makes it MLE. Squared loss is Gaussian noise. Cross-entropy is Bernoulli noise. The story tells you when the loss is the right one.
