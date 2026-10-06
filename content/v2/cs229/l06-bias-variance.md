---
page_id: cs229-l06
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 6
nav: "L06 · Bias, Variance, Model Selection"
title: "Lecture 6: Bias, Variance, and Picking Models"
summary: "The bias-variance canon, why double descent broke it, train/dev/test discipline, and Hyperband for hyperparameters."
date: "2026-04-22"
instructor: "Chris Ré"
offering: "Spring 2026"
duration: "1:18:20"
video_id: llnEgyyuYkQ
video_title: "Lecture 6: Dataset Split, ML Advice"
video_caption: "Original lecture. Chris Ré covers bias and variance, double descent, data splits, and Hyperband."
concepts: [bias-variance, double-descent, regularization, ridge, train-dev-test, cross-validation, hyperband, flat-minima, adaptive-overfitting]
sources:
  - tag: video
    label: "Lecture 6 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=llnEgyyuYkQ
  - tag: video
    label: "Explainer: StatQuest, Bias and Variance"
    url: https://www.youtube.com/watch?v=EuBBz3bI-aA
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: a model that works on houses it has never seen

You fit a line through the Ames houses. The loss on your spreadsheet
is tiny. You deploy it, and on new listings it is terrible. What
happened? The line memorized your spreadsheet's quirks instead of
learning the market. This gap, between performance on the data you
trained on and performance on new data, is the central drama of
machine learning. **Overfitting** is its name: the model fits the
training data's noise, not its signal.

## First attempt: fit harder

The naive response to bad predictions is a more flexible model. The
line misses curved patterns, so fit a degree-10 polynomial: 11 knobs
instead of 2. Watch it on a toy. The true market is a gentle curve.
You have 12 noisy sales. The degree-10 polynomial threads every
point exactly. Training loss: zero. Then a new house arrives between
two training points, where the polynomial whipsaws wildly to hit its
neighbors. Prediction error: enormous.

Demonstrate the failure with numbers. True price curve: y = x^2, x
from 0 to 2. Twelve training points with noise of plus or minus 0.3.
The degree-10 fit has training error 0.00. On 100 fresh test points,
its average squared error is 4.7. The humble line y = 0.5x + 0.3 has
training error 0.42 and test error 0.51. The flexible model wins
training 0.00 to 0.42 and loses the real game 4.7 to 0.51. Fitting
harder made predictions worse. That is overfitting, measured.

## Bias and variance: the two enemies

**Bias** is error from wrong assumptions. The line assumes the world
is straight. On a curved market it is systematically off. High bias
means the model cannot capture the pattern even with infinite data.
**Variance** is error from sensitivity to the training sample. The
degree-10 polynomial swings wildly when you swap in a different 12
houses. High variance means the model learned the noise.

The expected test error splits into three parts: bias squared, plus
variance, plus irreducible noise (the market's own randomness, which
no model removes). Simple models: high bias, low variance. Flexible
models: low bias, high variance. The classic picture is a U-curve:
test error falls as flexibility fixes bias, then rises as variance
takes over. The bottom of the U is the model you want.

![Bias and variance](assets/svg/l06-biasvar.svg "Bias versus variance. Simple models underfit with high bias. Flexible models overfit with high variance. Test error is U-shaped in the classical picture. Source: original plate for Stanford Frontier AI.")

### Subchapter: the decomposition, worked

Three fits of the true curve y = x^2 at x = 1, where the truth is
1.0. Three training samples give three predictions: 0.7, 0.9, 1.1.
Mean prediction: 0.9. Bias: 0.9 - 1.0 = -0.1, so bias squared is
0.01. Variance: average squared distance from the mean:
((0.7-0.9)^2 + (0.9-0.9)^2 + (1.1-0.9)^2)/3 = (0.04 + 0 + 0.04)/3 =
0.027. Expected test error: 0.01 + 0.027 + irreducible noise. The
three parts add up to the whole. When a model fails, this split
tells you which enemy to fight: shrink bias with flexibility,
shrink variance with data or penalties.

![Decomposition](assets/plate-l06-decomposition.webp "Three parts, one error. Predictions 0.7, 0.9, 1.1 at truth 1.0: bias squared 0.01, variance 0.027, plus noise. Source: original toy for the decomposition. Project: Stanford Frontier AI.")

## Where the canon breaks: double descent

For decades the U-curve was the whole story. Then neural networks
broke it. The lecture presents **double descent**: keep adding
parameters past the point where the model fits training data
perfectly, and test error does something the canon forbids.

The curve: test error descends (bias falling), ascends to a peak
(variance exploding near the **interpolation threshold**, where
parameters roughly equal data points and the fit becomes knife-edge
sensitive), then descends again. In the overparameterized regime the
optimizer, among the many zero-training-loss solutions, finds a
smooth one that generalizes. The lecture's image: a massively
overparameterized polynomial, given enough data and regularization,
finds the zero-loss solution that generalizes. "A wild statement,
but it turns out to be true in neural nets."

Numbers make it concrete. On CIFAR-10 style tasks, test error at the
classical sweet spot might be 15 percent, spike to 25 percent at the
interpolation peak, then fall to 8 percent with ten times more
parameters. The second descent is why modern models are enormous:
past the peak, bigger keeps helping. The canon is not wrong, it is
incomplete: it describes the left of the peak.

![Double descent](assets/svg/l06-dd.svg "Double descent. Test error falls, spikes at the interpolation threshold, then falls again as overparameterization lets the optimizer find smooth zero-loss solutions. Source: original plate for Stanford Frontier AI.")

## The key question

Bias and variance describe the tradeoff. But how do you *measure*
which regime you are in without peeking at the future? You cannot
evaluate on the training data (the polynomial scored 0.00 there and
lied). You need fresh data whose answers you know but the model has
never seen.

## Train, dev, test: the discipline

Split the data into three parts. The **training set** fits the
knobs. The **dev set** (validation set) measures and compares
models: try the line, the degree-3 polynomial, the degree-10, pick
the dev winner. The **test set** is touched once, at the very end,
to report honest performance.

Why three and not two? Because picking the model on the dev set
adapts to the dev set. Tune 100 models on dev and the winner is
partly lucky on dev. The test set stays clean because you never
decide anything with it. The lecture's warning: every decision made
on a dataset contaminates it. The test set is the one dataset you
decide nothing with.

![Three splits](assets/plate-l06-dev-test.webp "One dataset you never touch. Train fits the knobs. Dev decides between models. Test reports once and decides nothing. Source: original diagram for the split discipline. Project: Stanford Frontier AI.")

When data is scarce, **cross-validation** reuses it honestly: split
into k folds, train on k-1, validate on the held-out fold, rotate,
average. With k = 5, every example validates exactly once and trains
four times. It costs k training runs and buys an honest estimate
from small data.

## Ridge: pay for big knobs

Back to the polynomial that whipsawed. Its disease is huge
coefficients: to thread 12 noisy points with a degree-10 curve, some
knobs must be enormous, and enormous knobs mean violent swings
between points. **Ridge regression** attacks this directly: minimize
the usual squared loss plus a penalty on knob size.

```ascii
J_ridge(theta) = sum (h - y)^2 + rho * sum theta_j^2
```

**Rho** is the new dial: how much big knobs cost. At rho = 0 you get
the whipsawing polynomial. As rho grows, the knobs shrink, the curve
calms, variance falls, bias rises. The lecture derives the closed
form: theta = (X^T X + rho*I)^-1 X^T y. Two dividends. First, the
penalty fixes the singular-matrix failure of lecture 2: X^T X +
rho*I is always invertible for rho > 0. Second, in the
underdetermined case (fewer houses than knobs, n < d), infinitely
many knob settings fit the training data exactly. Ridge picks the
smallest one, the calmest fit.

Work the toy. Degree-10 polynomial on the 12 noisy points. Rho = 0:
test error 4.7. Rho = 1: the wild coefficients shrink tenfold, test
error 0.9. Rho = 100: the curve goes nearly flat, test error 2.1
(bias now dominates). The U returns, this time with rho on the axis
instead of flexibility. The lecture's intuition: "we know theta is
not too big. If we make it really big, it has got to be worth it by
fitting the data a lot better."

### Subchapter: Lasso, the cousin that deletes

Ridge penalizes the square of each knob: rho * theta_j^2. Its cousin
**Lasso** penalizes the absolute value: rho * |theta_j|. Same
shrinkage idea, different geometry, different fate.

Picture the penalty as a shape around the origin. Ridge's penalty
is a circle: it pulls every knob toward zero but never quite to
zero. Lasso's penalty is a diamond: its corners sit on the axes, and
the loss contours hit those corners first, which sets knobs exactly
to zero. Ridge shrinks. Lasso deletes.

The decision rule: use Lasso when you suspect most features are
junk and you want the model to say which: 1,000 candidate features,
Lasso keeps 17, you read the 17. Use ridge when features are
correlated and you want them to share credit: Lasso would keep one
twin and delete the other arbitrarily. The elastic net mixes both
penalties when you want a bit of each.

![Ridge vs Lasso](assets/plate-l06-ridge-vs-lasso.webp "Two penalties, two shapes. Ridge (L2) is a circle: shrinks every knob. Lasso (L1) is a diamond: corners delete knobs to exactly zero. Source: original diagram for the penalty shapes. Project: Stanford Frontier AI.")

## Hyperband: stop wasting compute on losers

Rho is a **hyperparameter**: a dial you set before training, not a
knob the training fits. Learning rate, polynomial degree, network
width are hyperparameters too. Tuning them by trying every
combination to completion wastes compute: most candidates are
obviously bad after a few minutes.

**Hyperband** is successive halving with smart budgets. The idea:
start many configurations with a small training budget each, keep
the best half, double their budget, repeat. A concrete schedule: 81
configs get 1 unit of training each. The best 27 get 3 units. The
best 9 get 9 units. The best 3 get 27 units. Total cost is about the
same as training a handful of configs fully, but you explored 81.
Bad ideas die cheap. Good ideas earn compute. The lecture presents
it as the disciplined answer to hyperparameter search: never spend a
full training run on a config that already looks bad.

![Hyperband](assets/svg/l06-hyperband.svg "Hyperband. Many configs start cheap. The best half survives each round with doubled budget. Bad ideas die cheap, good ideas earn compute. Source: original plate for Stanford Frontier AI.")

### Subchapter: random search beats grid

Before Hyperband decides budgets, decide which configs to try. Grid
search lays a lattice: 3 values per dial, 2 dials, 9 trials. The
disease: if only one dial matters, the grid tries just 3 distinct
values of it, repeated 3 times each. Random search draws 9
independent points: 9 distinct values of the important dial.

The classic result (Bergstra and Bengio): with 60 random trials, the
chance of missing the top 5% of configs is 0.95^60 = 5%. Sixty
random trials nearly guarantee a top-5% config. A grid of 60 cannot
promise that on any single dial. The decision rule: never grid
search more than 2 dials. Random search first, Hyperband to spend
the budget, Bayesian optimization when each trial costs a fortune.

![Random search](assets/plate-l06-random-search.webp "Random beats grid. Grid: 9 trials, 3 distinct values per dial. Random: 9 trials, 9 distinct values per dial. Source: original diagram for the search comparison. Project: Stanford Frontier AI.")

## The honest price

Every tool here charges. The train/dev/test split costs data: a
test set of 10,000 examples is 10,000 examples the model never
trains on. Cross-validation costs compute: k folds mean k training
runs. Ridge costs bias: shrink the knobs and you deliberately
underfit a little to overfit a lot less. Rho itself must be tuned on
dev. Hyperband costs the chance that a slow starter, a config that
looks bad early and would have won late, gets killed in round one.
And double descent charges humility: past the interpolation peak,
the classical advice "smaller is safer" is wrong, and the honest
practitioner checks which side of the peak they are on before
preaching simplicity.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| Bias-variance split | "Fit harder" backfired: 0.00 train, 4.7 test | Names the two enemies: wrong assumptions (bias) vs noise sensitivity (variance); U-curve locates the sweet spot |
| Double descent | The U-curve said big models must fail; they do not | Past the interpolation peak, optimizers find smooth zero-loss solutions; test error descends again |
| Train/dev/test | Training error lies (0.00) | Dev picks the model, test reports honestly once; cross-validation when data is scarce |
| Ridge | Whipsawing coefficients; singular X^T X | Penalty rho on knob size: (X^T X + rho I)^-1 X^T y; always invertible; toy test error 4.7 -> 0.9 |
| Hyperband | Tuning wastes full runs on losers | Successive halving: 81 configs start, best 3 finish; bad ideas die cheap |

> [!QA]
> Q: What is the bias-variance tradeoff?
> A: Expected test error splits into bias squared plus variance plus irreducible noise. Bias is error from wrong assumptions: a line on curved data is systematically off no matter how much data it gets. Variance is error from sensitivity to the training sample: the degree-10 polynomial swings wildly across different 12-house samples. Simple models have high bias and low variance. Flexible models reverse it. In the toy, the line scored test error 0.51 (biased but stable) while the degree-10 polynomial scored 4.7 (unbiased on average but chaotic per sample).
> Follow-up: What is the irreducible noise?
> A: Randomness in the world itself: two identical houses selling for different prices because one buyer overpaid. No model removes it. It sets the floor every error curve approaches but never crosses.

> [!QA]
> Q: What is double descent, and does it contradict bias-variance?
> A: It extends it. Test error descends as flexibility fixes bias, ascends to a peak near the interpolation threshold where parameters roughly equal data points, then descends again as overparameterization lets the optimizer pick smooth zero-loss solutions. The classical U-curve is the left half of the picture. It does not contradict bias-variance. It says variance behaves unexpectedly past the peak because the optimizer's choice among many perfect fits matters, not just the model class.
> Follow-up: Should I always use the biggest model then?
> A: Only if you can afford past the peak and you regularize. Between the classical sweet spot and the peak lies the worst region: big enough to be sensitive, not big enough to be smooth. The lecture's practical point: most modern progress came from jumping over the peak, not from sitting at the classical bottom.

> [!QA]
> Q: Why three data splits instead of two?
> A: Training fits the knobs. Dev compares models and tunes hyperparameters like rho. Test reports the final honest number once. You need dev separate from test because selecting the best of 100 models on dev adapts to dev: the winner is partly lucky there. If you then report that dev score, you overstate. The test set is the one dataset no decision ever touches. With small data, cross-validation rotates the dev role across k folds so every example validates once.
> Follow-up: What is adaptive overfitting?
> A: Contaminating the test set by deciding with it: tuning on test, early-stopping on test, or selecting among papers by test score. Each decision leaks test information into the model, and the reported number drifts optimistic. The discipline is absolute: decide on dev, report on test, once.

> [!QA]
> Q: How does ridge regression fix an underdetermined problem?
> A: With fewer examples than knobs (n < d), X^T X is singular and infinitely many theta fit training exactly. Ridge minimizes loss plus rho times ||theta||^2, giving theta = (X^T X + rho I)^-1 X^T y. The rho*I term makes the matrix invertible for any rho > 0, and among all perfect fits it selects the smallest-norm one: the calmest curve. On the toy, rho = 1 cut test error from 4.7 to 0.9 by shrinking the whipsawing coefficients tenfold.
> Follow-up: How do you pick rho?
> A: On the dev set, not the training set: training error falls monotonically as rho drops to 0, so training cannot choose. Sweep rho over a logarithmic grid (0.001, 0.01, 0.1, 1, 10, 100), pick the dev winner. The toy's sweep: 4.7 at 0, 0.9 at 1, 2.1 at 100.

> [!QA]
> Q: Walk me through the mechanism: compute bias and variance for predictions {0.6, 1.0, 1.4} when the truth is 1.0.
> A: Mean prediction: (0.6 + 1.0 + 1.4)/3 = 1.0. Bias: 1.0 - 1.0 = 0, so bias squared is 0. Variance: ((0.6-1.0)^2 + (1.0-1.0)^2 + (1.4-1.0)^2)/3 = (0.16 + 0 + 0.16)/3 = 0.107. Zero bias, large variance: on average the model is exactly right, but any single fit swings. This is the degree-10 polynomial's signature: unbiased and chaotic.
> Follow-up: Which enemy does more data fight?
> A: Variance. More samples pin the fit down, so the swings shrink. Bias from wrong assumptions survives infinite data: a line never learns a curve. Data fights variance. Flexibility fights bias. You need both moves.

> [!QA]
> Q: Applied design: your dev error is 5% but the held-out test error is 12%. Diagnose and fix.
> A: A 7-point gap means the test set differs from dev or dev is contaminated. Suspects in order: distribution shift (test drawn later or from a different slice), adaptive overfitting (too many model selections on dev, so dev is optimistic), or a test pipeline bug (different preprocessing). Diagnose: check per-slice errors, audit every decision ever made on test, diff the pipelines. Fix the cause, not the number: refresh the test set if it leaked, fix the pipeline if it diverged, and never tune on test to close the gap.
> Follow-up: How many dev decisions are too many?
> A: There is no number, only the trend: when dev keeps improving and your confidence in the test number keeps falling, you have adapted to dev. The defense is a fresh test set nobody has touched, or a dev set large enough that luck cannot move it.

> [!QA]
> Q: Ridge or Lasso? Give the decision rule and the geometry.
> A: Lasso when you want feature selection: its diamond penalty has corners on the axes, so loss contours hit corners and set knobs exactly to zero. Ridge when features correlate and should share credit: its circle shrinks everything but deletes nothing. Lasso on 1,000 candidate features keeps 17 and names them. Ridge on 50 correlated sensors keeps all 50, calmed. Elastic net mixes both when you want selection plus sharing.
> Follow-up: Why does the diamond give sparsity but the circle does not?
> A: The optimum sits where a loss contour first touches the penalty shape. The diamond's corners stick out along the axes: first touch happens at a corner, where all but one coordinate are zero. The circle has no corners: first touch is almost never exactly on an axis, so every knob stays nonzero, just small.

## Recap: the whole lesson on one screen

1. **The job.** A model that works on houses it has never seen.
   Training loss 0.00, real error enormous.
2. **First attempt.** Fit harder: degree-10 polynomial threads 12
   noisy points. Test error 4.7 vs the line's 0.51.
3. **Two enemies.** Bias: wrong assumptions (line on curves).
   Variance: noise sensitivity (polynomial whipsaws). Error =
   bias^2 + variance + noise.
4. **The canon breaks.** Double descent: past the interpolation
   peak, bigger models generalize again. The U is the left half.
5. **The key question.** How do you measure the regime without
   peeking at the future?
6. **Three splits.** Train fits, dev compares, test reports once.
   Cross-validation rotates when data is scarce.
7. **Ridge.** Penalize knob size: loss + rho||theta||^2. Closed
   form always invertible. Toy: 4.7 -> 0.9 at rho = 1.
8. **Hyperband.** Successive halving for hyperparameters: 81
   configs start, best 3 finish. Bad ideas die cheap.
9. **The honest price.** Splits cost data, CV costs compute, ridge
   costs bias, Hyperband can kill slow starters, double descent
   humbles the simplicity sermon.
10. **The decomposition, worked.** Predictions 0.7, 0.9, 1.1 at truth
    1.0: bias squared 0.01, variance 0.027, noise on top.
11. **Lasso.** L1 penalty is a diamond: corners delete knobs to
    zero. Ridge shrinks, Lasso selects.
12. **Random beats grid.** 60 random trials miss the top 5% only
    5% of the time. Never grid more than 2 dials.

## What is used where

**Ridge is the default regularized linear model in production.**
Scikit-learn's Ridge fits in one line, and regularized logistic
regression (the same penalty on the lecture-3 loss) scores credit,
fraud, and ads. Lasso runs feature selection wherever readings are
cheap and truth is sparse: genomics, sensor selection, marketing
mix.

**The split discipline runs every serious ML team.** Train, dev,
test with a locked test set is the industry standard. Cross
validation covers small data. Hyperband's child ASHA and Optuna run
hyperparameter search in production tuning loops.

**Double descent shapes how the field spends money.** Past the
interpolation peak, bigger keeps helping: this is the empirical
fact behind billion-parameter budgets. The classical U-curve still
rules small models, where the peak is never crossed.

## Watch next

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/EuBBz3bI-aA" title="StatQuest: Bias and Variance" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Explainer: StatQuest, Bias and Variance. Josh Starmer draws the bullseye picture and the U-curve from scratch. Watch after the bias-variance section.</p></div>

## Official sources and further reading

**Official:**
- Lecture 6 video, Stanford Online YouTube:
  - [Chris Ré derives](https://www.youtube.com/watch?v=llnEgyyuYkQ)
  the bias-variance decomposition, presents double descent, the
  train/dev/test discipline, ridge regression, and Hyperband.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the full
  derivations.

**Caveats from these sources.** The double-descent numbers in this
lesson (15, 25, 8 percent) are illustrative of the shape the lecture
draws, not values read off a specific lecture plot. The lecture's
claim is the shape and the mechanism. The polynomial toy is an
original miniature demonstrating the lecture's claims. Hyperband's
81-27-9-3 schedule is the lecture's successive-halving illustration.

## Connections to the other courses

- **CS229 L02:** the least-squares loss that ridge extends. The
  singular X^T X that rho*I repairs.
- **CS229 L05:** Laplace smoothing as regularization's simplest
  form, previewing this lesson.
- **CS229 L07-L08:** neural networks live past the interpolation
  peak. Double descent is their native habitat.
- **CS229 L12:** foundation models: the second descent at billion
  parameter scale.
- **CS336:** what the interpolation peak looks like in real
  training runs and how practitioners budget past it.
