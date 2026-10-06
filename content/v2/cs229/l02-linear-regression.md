---
page_id: cs229-l02
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 2
nav: "L02 · Linear Regression"
title: "Lecture 2: Supervised Learning and Linear Regression"
summary: "The supervised setup, the loss chip J(theta), gradient descent, SGD, and the normal equations, all on the Ames housing data."
date: "2026-04-08"
instructor: "Chris Ré"
offering: "Spring 2026"
duration: "1:18:04"
video_id: cmNIMjPYdgM
video_title: "Lecture 2: Supervised Learning Setup"
video_caption: "Original lecture. Chris Ré builds linear regression from the Ames housing data: loss, gradient descent, SGD, normal equations."
concepts: [supervised-learning, hypothesis, regression, loss-function, least-squares, gradient-descent, SGD, normal-equations, learning-rate, MFU]
sources:
  - tag: video
    label: "Lecture 2 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=cmNIMjPYdgM
  - tag: video
    label: "Explainer: StatQuest, Linear Regression Clearly Explained"
    url: https://www.youtube.com/watch?v=nk2CQITm_eo
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
  - tag: supplement
    label: "Ames Housing dataset documentation"
    url: https://jse.amstat.org/v19n3/decock.pdf
---

## The job: price a house in Ames, Iowa

A buyer asks you: "This house has 1,800 square feet of living area.
What should it sell for?" You have a spreadsheet of past sales in
Ames, Iowa: each row is one house, with its living area and the price
it sold for. This is the most canonical dataset in statistics, and the
lecture uses it to build the entire machinery of supervised learning.

Look at the data first. The lecture insists on this: look at your
data. Plot price against living area. The points slope upward: bigger
houses sold for more. The cloud of points is noisy but the trend is
clear. A line through the cloud would let you price any new house by
reading off the line.

So the job is: find the best line through the points. "Best" needs a
definition, "find" needs an algorithm. This chapter builds both from
zero, and every later course reuses what is built here.

## The supervised setup, piece by piece

Supervised learning has four characters. Meet each one with the
housing example.

An **example** is one labeled pair: the input features and the
correct answer. Write the i-th example as (x^(i), y^(i)). The input
x^(i) is a vector of **features**: measurable facts about the house.
For now, one feature: living area in square feet. The answer y^(i) is
the **target**: the sale price, the thing we want to predict. The
lecture's first dataset has hundreds of such pairs.

A **hypothesis** is the machine's guess at the rule mapping inputs to
answers. Write it h(x). It takes a feature vector and returns a
predicted price. The hypothesis is a function with knobs. The knobs
are the **parameters**, written theta. A line has two knobs: the
intercept theta_0 (where the line crosses the price axis) and the
slope theta_1 (dollars per square foot). The hypothesis is:

```ascii
h_theta(x) = theta_0 + theta_1 * x_1
```

Read it as: predicted price equals a base price plus slope times
living area. With theta_0 = 50,000 and theta_1 = 100, a 1,800 sq ft
house is predicted at 50,000 + 100 * 1,800 = $230,000. Different
knob settings give different lines. Learning means finding the knob
settings that fit the data best.

### Subchapter: the hypothesis, worked

Two houses: 1,000 sq ft sold for $200,000, 2,000 sq ft sold for
$400,000. Try knobs theta_0 = 0, theta_1 = 200. Predictions:
h(1000) = 200,000, h(2000) = 400,000. Both exact. Now try
theta_0 = 100,000, theta_1 = 100. Predictions: h(1000) = 200,000,
h(2000) = 300,000. The second house misses by $100,000. Same
function shape, different knobs, different fit. The hypothesis is a
family of lines. Learning is the search over the family. Every knob
setting is one candidate line, and the loss will score them.

**Regression** is the name for this job: predicting a number. The
target is continuous: prices, temperatures, wait times. (Predicting a
category, like spam or not spam, is **classification**. It comes in
lecture 3.) The Ames job is regression.

![The loss chip](assets/svg/l02-loss-chip.svg "Shell 1. The loss chip scores how badly the line misses. The loss chip J(theta). Prediction minus truth, squared, averaged over all examples. Smaller means the line fits better. Source: original plate for Stanford Frontier AI.")

Now define "best". The **loss** is a single number that scores one
knob setting. For each house, compute the prediction error: how far
the line's prediction missed the true price. Square it, so misses in
both directions count and big misses count a lot. Average over all
houses. The lecture's loss, with m examples, is:

```ascii
J(theta) = 1/(2m) * sum over i of (h_theta(x^(i)) - y^(i))^2
```

This is the **loss chip**: the symbol this course owns and every
later course reuses. Three things to notice. The square makes the
loss always positive and punishes a $100,000 miss one hundred times
more than a $10,000 miss. The 1/m averages over houses so the number
does not grow with the dataset. The 1/2 is cosmetic: it cancels a 2
that appears when you differentiate. The lecture calls this out
explicitly.

Minimizing J(theta) is the whole game of supervised learning. Pick
theta, score it with J, adjust theta to lower J, repeat.

### Subchapter: three losses, one toy

Why squares and not something else? Three contenders on one toy:
a line misses three houses by $1,000, $2,000, and $10,000. Squared
loss: 1 + 4 + 100 = 105 (in millions of dollars squared). Absolute
loss: 1 + 2 + 10 = 13 (in millions). The $10,000 miss is 10 times
the $1,000 miss under absolute loss, but 100 times under squared
loss. Squares punish big misses quadratically: one huge error
dominates the sum, which is exactly what you want when huge errors
are the expensive ones.

Absolute loss has a kink at zero: it is not differentiable there,
so gradient descent stalls at the kink. Squared loss is smooth
everywhere. The price of squares: outliers. One $1,000,000 miss
contributes 10^12 and drags the whole line toward itself. Lecture 3
gives squares a deeper meaning (Gaussian noise), and lecture 6
shows what outliers cost. For now: smooth, differentiable, and
big-miss-phobic.

![Chapter plate: the supervised setup](assets/plate-l02-chap-setup.svg "Chapter plate L02-C1. Left: no score: any hand-picked line through the cloud, knobs with no judge. Center: the loss chip J(theta) = 1/(2m) sum of squared errors, where the square punishes big misses 100-fold. Right: the 3-house toy scores J = 6.33, then 3.32 after one step; squared 105 beats absolute 13. Bottom: one $1M miss contributes 10^12 and drags the line. Dense chapter plate. Source: original synthesis of the lesson. Project: Stanford Frontier AI.")

## First attempt: solve it with calculus

The oldest idea: find where the loss bottoms out. At the bottom of a
smooth bowl, the slope is zero. Compute the derivative of J with
respect to each theta_j, set it to zero, solve for theta. For a line
through points, this gives a closed-form answer called the **normal
equations**, derived at the end of this chapter.

But there is a problem with the calculus-first approach, and the
lecture leads with the other route for a reason. The closed form
needs a matrix inverse. It works for lines. It does not work for
neural networks, where the loss surface is rugged with no closed
form. The field's real workhorse is iterative: start somewhere, walk
downhill, and stop at the bottom. That workhorse is **gradient
descent**, and it trains nearly every model you will ever meet,
including the giant language models of lectures 14 through 17.

## Gradient descent, by hand

A **gradient** is the direction of steepest uphill. At any point on
the loss surface, the gradient is a vector of partial derivatives,
one per knob. It points uphill. So to go downhill, step the opposite
way. The update rule, applied to every knob at once, is:

```ascii
theta_j := theta_j - alpha * (partial derivative of J / partial theta_j)
```

The new knob value equals the old one minus the step size times the
slope. **Alpha** is the **learning rate**, also called the step size:
how big each step is. Subtract the gradient, and you move downhill.
Repeat.

For linear regression the derivative has a clean form. Watch the
pieces: the derivative of the squared error brings down the error
term (prediction minus truth) times the feature:

```ascii
theta_j := theta_j - alpha * (1/m) * sum over i of (h_theta(x^(i)) - y^(i)) * x_j^(i)
```

The lecture highlights the shape: **error times feature, summed**.
The error (h - y) says how wrong the prediction was and in which
direction. The feature x_j says how much knob j contributed. Big
error on a house with big living area pushes the slope knob hard.
This error-times-something pattern recurs in every learning rule in
the course. Learn to spot it.

### Subchapter: the vectorized update

The per-knob update has a matrix form that implementations use.
Stack the errors into the vector (X theta - y) and write the full
gradient at once: theta := theta - alpha (1/m) X^T (X theta - y).
One line replaces the loop over j. On the 3-house toy: X theta -
y = (-2, -3, -5), X^T times that = (-10, -23), divided by 3 =
(-3.33, -7.67), the same gradients as the hand computation.

The vectorized form is not just notation. It is what BLAS
(Basic Linear Algebra Subprograms) libraries execute: one
matrix-vector multiply instead of n scalar loops, which is where the
hardware speed comes from. Every production gradient step is this
line. The per-knob form is for understanding. The matrix form is
for running.

### Subchapter: when to stop

Gradient descent needs a stopping rule: the loss never announces
it has arrived. Three rules, in order of honesty. The gradient
norm: stop when ||gradient|| < epsilon (say 10^-6). On the toy,
the gradient shrinks 6.33-scale to near zero as theta approaches
(1/3, 3/2). The loss plateau: stop when J falls less than delta
(say 10^-8) for k consecutive steps. The budget: stop after T
steps, no questions.

The gradient norm is the principled one: at the exact bottom the
gradient is zero, so a tiny gradient means you are close. The
plateau rule is the practical one on noisy losses where the
gradient never quite vanishes (SGD). The budget is the honest one
when compute is the constraint. In practice: run the plateau rule
with a budget cap, and log the gradient norm to check you actually
converged instead of just timing out.

![Error times feature](assets/plate-l02-error-times-feature.webp "Shell 2. Error times feature is the update shape everywhere. The error-times-feature pattern. Linear regression: error times feature, summed. Logistic regression: same shape, sigmoid inside. Backprop: same shape, chain rule outside. One pattern, three lessons. Source: original plate for Stanford Frontier AI.")

Now work it by hand. Three houses, one feature, and the knobs start
at zero.

```ascii
houses:  (x=1, y=2)   (x=2, y=3)   (x=3, y=5)
knobs:   theta_0 = 0, theta_1 = 0        alpha = 0.05
start:   predictions all 0.  J = 1/6 * (4 + 9 + 25) = 6.33
errors:  (0-2) = -2,  (0-3) = -3,  (0-5) = -5
grad_0 = (1/3) * (-2 - 3 - 5) = -3.33
grad_1 = (1/3) * (-2*1 - 3*2 - 5*3) = (1/3) * (-23) = -7.67

step:    theta_0 := 0 - 0.05 * (-3.33) = 0.167
         theta_1 := 0 - 0.05 * (-7.67) = 0.383

after:   h(1) = 0.55,  h(2) = 0.93,  h(3) = 1.32
         J = 1/6 * (2.10 + 4.28 + 13.54) = 3.32
```

One step cut the loss from 6.33 to 3.32. The line is still terrible,
but it moved in the right direction on every knob. Run this update
hundreds of times and the knobs settle near theta_0 = 0.33,
theta_1 = 1.5, the best line through the three points. That is
gradient descent: many small downhill steps, each one cheap.

### Subchapter: the descent trace, three iterations

Continue the toy and watch J fall. After step 1: theta = (0.167,
0.383), J = 3.32. Step 2: predictions 0.55, 0.93, 1.32 give errors
-1.45, -2.07, -3.68. Gradients: grad_0 = -2.40, grad_1 = -5.54.
New theta: (0.287, 0.660). New predictions: 0.95, 1.61, 2.27. New
J: 1.75. Step 3 continues the pattern and J lands near 1.0.

The trace: 6.33, 3.32, 1.75, ~1.0, ... , 0.028. Each step roughly
halves the remaining distance. The loss never rises: every step
moves opposite the gradient, so J falls monotonically when alpha is
small enough. The destination is the normal-equations answer
theta = (1/3, 3/2), J = 0.028. Gradient descent does not know the
destination. It walks downhill until the slope is zero, and for
this bowl the slope is zero only at the bottom.

![Gradient descent](assets/svg/l02-gd.svg "Shell 3. Alpha sets each step against the gradient. Gradient descent. Each knob steps opposite its gradient. Alpha sets the step size. Small alpha crawls, large alpha overshoots. Source: original plate for Stanford Frontier AI.")

### Subchapter: feature scaling, the hidden dial

One more dial hides inside gradient descent: the scale of the
features. Take two features: living area (1,000 to 3,000 sq ft) and
bedrooms (1 to 5). One gradient step moves each knob by alpha times
its gradient, and the living-area gradient is about 1,000 times
larger than the bedroom gradient. With one shared alpha, the area
knob leaps while the bedroom knob crawls.

Picture the loss surface. Unscaled, it is a long narrow valley:
steep across the area direction, flat along bedrooms. Gradient
descent zigzags across the valley walls, taking hundreds of steps to
drift down the flat floor. Scale both features to the range 0 to 1
(subtract the mean, divide by the range), and the valley becomes a
round bowl. The same alpha now works for every knob, and descent
walks straight to the bottom in a fraction of the steps.

The decision rule: always scale features before gradient descent.
The normal equations do not care about scale, but every iterative
method does. A model that will not converge at any alpha is often a
model with unscaled features.

![Feature scaling](assets/plate-l02-feature-scaling.webp "Shell 4. Scaling turns the narrow valley into a round bowl. Feature scaling. Unscaled features make a narrow valley and zigzag descent. Scaled features make a round bowl and straight descent. Source: original plate for Stanford Frontier AI.")

## Where it breaks: the learning rate

Alpha is the one dial you must set, and both extremes fail. Watch
each failure on the simplest bowl: J(theta) = theta^2. The gradient
is 2*theta. Start at theta = 4. The bottom is at 0.

Alpha too small: alpha = 0.01. Each step moves 2 percent of the
distance. Step 1: theta = 4 - 0.01*8 = 3.92. After 100 steps you are
at about 0.53. After 1,000 steps, about 0.000000007, effectively
zero. It converges, but you paid 1,000 steps for a one-dimensional
problem. On a real model with
millions of knobs, tiny alpha means training never finishes.

Alpha too large: alpha = 1.5. Step 1: theta = 4 - 1.5*8 = -8. Step 2:
theta = -8 - 1.5*(-16) = 16. Step 3: theta = 16 - 1.5*32 = -32. The
steps explode: 4, -8, 16, -32, 64, -128. Each step overshoots the bottom and lands
farther out than it started. The loss does not fall. It diverges to
infinity. The lecture shows this live: with alpha set too high, the
loss bounces around in a widening ball and never settles.

The working range is narrow and problem-dependent. In practice you
try a few values, watch the loss curve, and pick the largest alpha
whose loss falls smoothly. The lecture's rule of thumb: when the loss
bounces instead of falling, your alpha is too high. Turn it down.

![Learning rate traces](assets/plate-l02-alpha-traces.webp "Shell 5. Alpha too small crawls, too large explodes. Three learning rates on the bowl J = theta squared from theta = 4. Alpha 0.01 crawls: 100 steps reach 0.53. Alpha 0.1 converges smoothly. Alpha 1.5 explodes: 4, minus 8, 16, minus 32, 64. Source: original plate for Stanford Frontier AI.")

![Chapter plate: gradient descent and alpha](assets/plate-l02-chap-gd.svg "Chapter plate L02-C2. Left: calculus only: derivative set to zero, needs a matrix inverse, dies on neural networks. Center: error times feature summed over examples: the toy moves from (0,0) to (0.167, 0.383), J from 6.33 to 3.32. Right: alpha 0.01 crawls to 0.53 in 100 steps, alpha 0.1 converges, alpha 1.5 explodes to infinity. Bottom: when the loss bounces instead of falling, alpha is too high. Dense chapter plate. Source: original synthesis of the lesson. Project: Stanford Frontier AI.")

## The key question

Each gradient step in the toy summed over all three houses. That is
**batch** gradient descent: every step scans the full dataset. Now
scale up. The lecture asks: what if the dataset is the entire
internet and the job is next-word prediction? One step would need to
score every page on earth before moving a single knob. That cannot
work. What if each step used just one example instead of all of
them?

### Subchapter: batch cost accounting

Count in example-visits: one batch step visits all m examples. With
m = 2 billion ad impressions and 40 features, one step does 80
billion multiply-adds. A modern GPU does ~10 trillion per second,
so the arithmetic is 0.008 seconds per step. Roughly 1,000 steps to
converge: 8 seconds of arithmetic per fit. The arithmetic is cheap.
But every step also needs a full pass over the data, and the model
must retrain hourly to track drift. The arithmetic fits. The
full-pass schedule does not.

Worse, every step waits on the slowest part: reading 2 billion rows
from storage. Batch descent is a full data pass per step. On
internet-scale data, one step costs one eternity, and you need
hundreds of steps. The cost is not the FLOPs. It is the passes.

## Stochastic gradient descent

**Stochastic gradient descent** (SGD) uses one training example per
step, picked at random. The update becomes:

```ascii
theta_j := theta_j - alpha * (h_theta(x^(i)) - y^(i)) * x_j^(i)
```

No sum over m. No 1/m. Just one example's error times its features.
Each step is m times cheaper. With m = 1,000,000 houses, one batch
step costs 1,000,000 error computations. One SGD step costs 1.

The price is noise. One example's gradient is a jittery estimate of
the true downhill direction. The knobs wiggle. The lecture shows the
trace: batch descent glides smoothly to the bottom. SGD bounces
around it, sometimes stepping the wrong way. But it bounces *while
moving*: in the time batch descent takes one exact step, SGD takes
one million noisy steps and gets much closer. For huge datasets the
noisy fast method beats the exact slow method, every time. This is
the algorithm that trains modern neural networks. The "stochastic"
part is not a hack. It is the reason training at internet scale is
possible at all.

One practical rule from the lecture: shuffle the data before each
pass (**epoch**), so the machine does not learn the order of your
spreadsheet. And a common trick when alpha is large: average the
knobs over the bouncing trajectory to simulate a steadier estimate.

### Subchapter: SGD's unbiased noise, worked

"Unbiased" has a precise meaning: the average single-example
gradient equals the batch gradient. Check it on a 2-house toy:
(x=1, y=2), (x=2, y=3), theta = 0. Batch gradient for the slope:
(1/2)[(0-2)(1) + (0-3)(2)] = (1/2)(-8) = -4. Single-example
gradients: house 1 gives (0-2)(1) = -2, house 2 gives (0-3)(2) =
-6. Pick one at random: the expected gradient is (-2 + -6)/2 = -4.
Exactly the batch gradient.

Each SGD step is wrong, but the errors average to zero. Over many
steps the noise cancels and the signal accumulates. This is why the
bouncing still arrives: the bounces are symmetric around the true
downhill direction. Shuffling each epoch keeps the sampling honest,
so no example's noise dominates.

![Batch vs SGD](assets/svg/l02-sgd.svg "Shell 6. Batch is exact and slow, SGD noisy and fast. Batch gradient descent uses all m examples per step: exact direction, slow steps. SGD uses one example: noisy direction, fast steps. Shuffle every epoch. Source: original plate for Stanford Frontier AI.")

### Subchapter: mini-batch, the middle path

Batch descent and SGD are two ends of a dial. The dial is the batch
size B: how many examples each step uses. Batch is B = m. SGD is
B = 1. The middle is **mini-batch** SGD: B = 32, 64, 128, or 256.

Count the tradeoff on m = 1,000,000 houses. A batch step costs
1,000,000 error computations and gives one exact direction. An SGD
step costs 1 and gives one noisy direction. A mini-batch step with
B = 64 costs 64 and gives a direction 64 times less noisy than SGD.
In the time batch descent takes one step, mini-batch takes about
15,000 steps, each nearly as good as the exact one.

Hardware adds a second reason. Modern chips compute 32 or 64
examples almost as fast as 1, because the arithmetic runs in
parallel. B = 32 costs barely more wall-clock time than B = 1 but
cuts the noise by a factor of 32. This is why every production
training loop uses mini-batches: the batch size is the one number
that trades noise against hardware at the same time.

![Mini-batch](assets/plate-l02-minibatch.webp "Shell 7. Mini-batches split the difference for real hardware. Mini-batch SGD. Batch size 1 is noisy and cheap. Batch size m is exact and slow. Batch size 32 to 256 is the production middle: parallel hardware eats 32 examples almost as fast as 1. Source: original plate for Stanford Frontier AI.")

![Chapter plate: stochastic gradient descent](assets/plate-l02-chap-sgd.svg "Chapter plate L02-C3. Left: full passes: one step scores all m examples, 80B multiply-adds at 2B examples, and the schedule fails. Center: one example per step with no sum, an unbiased estimate of the batch gradient: (-2 + -6)/2 = -4. Right: m times cheaper per step and arriving sooner, with mini-batch 32 to 256 as the production middle. Bottom: noise is the price of speed; shuffle every epoch. Dense chapter plate. Source: original synthesis of the lesson. Project: Stanford Frontier AI.")

## The normal equations: the exact answer for lines

For linear regression there is also a one-shot answer. Stack all
examples into a matrix X (m rows, one per house. N+1 columns, one per
knob including the intercept column of ones) and all targets into a
vector y. The loss in matrix form is J(theta) = (1/2m)(X theta -
y)^T (X theta - y). Set the gradient to zero and solve:

```ascii
theta = (X^T X)^(-1) X^T y
```

These are the **normal equations**. No alpha, no iterations, no
bouncing. One formula, exact bottom of the bowl.

### Subchapter: the normal equations, derived

Derive them from zero in four steps. Step 1: write the loss in
matrix form. J(theta) = (1/2m)(X theta - y)^T (X theta - y). Step 2:
expand. (X theta - y)^T (X theta - y) = theta^T X^T X theta - 2
theta^T X^T y + y^T y. Step 3: differentiate with respect to
theta. The derivative of theta^T A theta is 2 A theta (for symmetric
A), and the derivative of theta^T b is b. So the gradient is (1/m)
X^T X theta - (1/m) X^T y, or (1/m) X^T (X theta - y). Step 4: set
to zero. X^T X theta = X^T y. Multiply by the inverse: theta =
(X^T X)^-1 X^T y.

The 1/m and 1/2 vanish in step 4: scaling the loss does not move
its minimum. What remains is a linear system: X^T X theta = X^T y.
The normal equations are the statement "the gradient is zero",
solved for theta. Everything before the inverse is bookkeeping.

Solve the toy by hand. Three houses, X has rows [1,1], [1,2], [1,3],
y = [2, 3, 5].

```ascii
X^T X = [[3, 6], [6, 14]]        X^T y = [10, 23]
det = 3*14 - 6*6 = 42 - 36 = 6
(X^T X)^(-1) = (1/6) * [[14, -6], [-6, 3]]

theta_0 = (1/6) * (14*10 - 6*23) = (1/6) * (140 - 138) = 1/3
theta_1 = (1/6) * (-6*10 + 3*23) = (1/6) * (-60 + 69) = 3/2
```

The best line is h(x) = 1/3 + 1.5x. Predictions: 1.83, 3.33, 4.83.
Errors: -0.17, -0.33, 0.17. Squared errors sum to 0.167, so
J = 0.167/6 = 0.028. This is the exact minimum. Gradient descent was crawling
toward this point. The normal equations jump straight to it.

### Subchapter: when the inverse fails, worked

The formula needs (X^T X)^-1 to exist. Watch it fail on redundant
features. Two features: living area in square feet and living area
in square meters. One square foot is 0.0929 square meters, so column
2 is 0.0929 times column 1: the columns are linearly dependent.

Toy: X^T X = [[3, 0.279], [0.279, 0.0259]]. Determinant:
3(0.0259) - 0.279^2 = 0.0777 - 0.0778 = -0.0001, essentially zero.
The matrix is singular: no inverse exists. Geometrically, the loss
bowl has a flat valley floor: infinitely many knob pairs (theta_1,
theta_2) give the same predictions, because the two features carry
identical information. The equation cannot choose among them.

The fixes, in order: drop the redundant feature (cheapest), collect
more varied data, or add a ridge penalty (lecture 6), which makes
(X^T X + rho I) invertible for any rho > 0. The singular case is
not rare: one-hot encodings with a dropped-nothing column and
text features with duplicate columns hit it constantly.

![Normal equations](assets/svg/l02-normaleq.svg "Shell 8. The normal equations solve theta in one shot. Closed form: theta = (X^T X)^-1 X^T y. Exact in one shot. Needs an invertible X^T X and costs O(n^3). Source: original plate for Stanford Frontier AI.")

![Chapter plate: the normal equations](assets/plate-l02-chap-normaleq.svg "Chapter plate L02-C4. Left: hundreds of steps: alpha tuned by hand, the trace crawls from 6.33 to 0.028. Center: theta = (X'X)^-1 X'y in one shot: det 6, theta (1/3, 3/2), exact J = 0.028. Right: O(n^3) makes n = 10,000 cost 10^12 ops, and redundant features make the inverse fail. Bottom: no alpha and no iterations, but only for small n, full rank, and linear models. Dense chapter plate. Source: original synthesis of the lesson. Project: Stanford Frontier AI.")

## Mapping back: three ways to fit a line

| Approach | How it finds theta | Cost per answer | When it wins |
|---|---|---|---|
| Gradient descent | Many small downhill steps, alpha dialed by hand | O(m*n) per step, many steps | Huge m, or loss with no closed form (neural nets) |
| SGD | One example per step, noisy but fast | O(n) per step | Internet-scale m; the only option when one pass over data is expensive |
| Normal equations | One matrix inverse, exact | O(n^3) to invert, needs X^T X invertible | Small n (hundreds of features), exact answer wanted |

Each answers a different pain. Batch descent is too slow per step on
huge data. SGD fixes the per-step cost. Iterative methods need alpha
tuning and many steps. The normal equations skip both when the
problem is small and linear. Nothing here works on neural networks
except the gradient idea itself, which is why lectures 7 and 8 exist.

## What is used where

**Linear regression runs in production more than any other model.**
[uncertain: no public census of production models exists.]
It is the default baseline for pricing, forecasting, and
calibration layers. Ad systems fit linear models on billions of
impressions because a linear model scores in microseconds and
retrains in minutes [uncertain: internal serving details, not
public]. Every serious team fits linear regression first:
if a fancier model cannot beat it, the fancier model ships nothing
[uncertain: team lore, not sourced].

**The normal equations never run at scale.** No production system
inverts X^T X on millions of rows [uncertain: a universal negative,
not verifiable from public sources]. The formula survives in
textbooks and in small-data statistics, where n is hundreds and the
exact answer is free. Scikit-learn's LinearRegression uses an SVD
(singular value decomposition) solver under the hood (scikit-learn
docs, checked Oct 2026), which is the numerically stable cousin of the
normal equations, and it is the right call up to about n = 10,000
features [uncertain: rule of thumb, not sourced].

**SGD and mini-batch SGD run everything else.** Every neural
network in production trains on a variant of the mini-batch update
from this lesson [uncertain: stated as field consensus, not sourced]:
PyTorch and JAX loops are mini-batch SGD with
adaptive step sizes (Adam) [uncertain: framework internals vary by
team]. The step you hand-computed on three
houses is the great-grandfather of every LLM training run.

Sources (checked Oct 2026): scikit-learn LinearRegression docs, SVD-based lstsq solver on dense data.

## Watch next

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/nk2CQITm_eo" title="StatQuest: Linear Regression, Clearly Explained" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Explainer: StatQuest, Linear Regression Clearly Explained. Josh Starmer fits the line by least squares with the same error-times-feature update. Watch after the gradient-descent section.</p></div>

## The honest price

The normal equations have two bills. First, inverting X^T X costs
O(n^3) in the number of features. With n = 10,000 features, that is
10^12 operations: the "exact" answer is slower than gradient descent.
Second, the inverse exists only if X^T X is invertible. If two
features are redundant (living area in sq ft and in sq meters), or if
m < n (fewer houses than knobs), the matrix is singular and the
formula breaks: infinitely many lines fit equally well, and the
equation cannot choose. Gradient descent has its own bill: the
learning rate. Too small crawls, too large diverges, and the right
value is found by trial, not theory. Every optimizer in this course
is a negotiation with these prices.

> [!QA]
> Q: What is the loss function for linear regression, and why the 1/2?
> A: J(theta) = 1/(2m) times the sum of squared prediction errors over all m examples. The square punishes big misses more than small ones and keeps the loss positive. The 1/m averages so the number does not grow with the dataset. The 1/2 is cosmetic: differentiating the square brings down a factor of 2, and the 1/2 cancels it, leaving clean update rules. In the hand toy, three houses gave J = 6.33 at theta = 0 and J = 3.32 after one gradient step.
> Follow-up: Why square the errors instead of taking absolute values?
> A: Squares are smooth and differentiable everywhere, so gradient descent applies directly. Absolute values have a kink at zero. Squares also punish large errors quadratically, which matches the Gaussian-noise story of lecture 3. The price: squared loss is sensitive to outliers, since one huge error dominates the sum.

> [!QA]
> Q: Derive the gradient descent update for linear regression.
> A: Start from J(theta) = 1/(2m) sum (h_theta(x^(i)) - y^(i))^2 with h = theta^T x. Differentiate with respect to theta_j: the 1/2 cancels the 2 from the square, leaving (1/m) sum (h - y) * x_j^(i). The update is theta_j := theta_j - alpha * (1/m) sum_i (h_theta(x^(i)) - y^(i)) x_j^(i). The lecture stresses the shape: error times feature, summed over examples. On the 3-house toy with alpha = 0.05, one step moved theta from (0, 0) to (0.167, 0.383) and cut J from 6.33 to 3.32.
> Follow-up: Why must all theta_j update simultaneously from the old values?
> A: Because the gradient is computed at the current point. If you update theta_0 first and then use the new theta_0 to compute theta_1's update, you are stepping along a stale-mixed gradient, not the true downhill direction. Simultaneous update keeps the step honest. Implementations compute all new values into a temporary copy, then assign.

> [!QA]
> Q: When would you use the normal equations instead of gradient descent?
> A: When n is small (hundreds of features) and you want the exact answer with no learning-rate tuning. The normal equations cost O(n^3) for the inverse, so at n = 10,000 features they lose to gradient descent. They also fail when X^T X is singular: redundant features or fewer examples than features give infinitely many solutions and no inverse. Gradient descent and SGD are the only options for neural networks, where no closed form exists at all.
> Follow-up: What do you do when X^T X is singular?
> A: Remove redundant features, get more data, or add a ridge penalty (lecture 6): minimize J(theta) + lambda * ||theta||^2, which makes the matrix (X^T X + lambda*I) invertible. The penalty also shrinks the knobs, which fights overfitting.

> [!QA]
> Q: Why does SGD work if each step uses a noisy gradient?
> A: Each single-example gradient is an unbiased estimate of the true gradient: wrong per step, right on average. The noise averages out over many steps while each step costs O(n) instead of O(m*n). On m = 1,000,000 examples, SGD takes a million cheap steps in the time batch descent takes one expensive step. The lecture's trace shows SGD bouncing around the optimum while batch glides. The bouncing still arrives far sooner. Shuffling each epoch keeps the noise honest.
> Follow-up: Why not use a huge batch instead, getting exact gradients with parallelism?
> A: That is mini-batch SGD, the practical middle ground: batches of 32 to 256 examples. Big enough to use hardware well and smooth the noise, small enough to step often. Full-batch on internet-scale data is still one step per eternity.

> [!QA]
> Q: Walk me through one gradient-descent step on a new toy: houses (x=2, y=4) and (x=4, y=6), theta = (0, 0), alpha = 0.1.
> A: Predictions are both 0. Errors: (0-4) = -4, (0-6) = -6. J = 1/4 * (16 + 36) = 13. Grad_0 = (1/2)(-4 - 6) = -5. Grad_1 = (1/2)(-4*2 - 6*4) = (1/2)(-32) = -16. Step: theta_0 := 0 - 0.1*(-5) = 0.5, theta_1 := 0 - 0.1*(-16) = 1.6. New predictions: h(2) = 0.5 + 3.2 = 3.7, h(4) = 0.5 + 6.4 = 6.9. Errors: -0.3 and 0.9. J = 1/4 * (0.09 + 0.81) = 0.225. One step cut the loss from 13 to 0.225.
> Follow-up: Why did this toy converge in one step while the lesson's toy did not?
> A: Alpha was larger relative to the curvature, and the two points nearly determine the line. It is luck of the numbers, not a property of the method. Do not generalize from one toy: the lesson's toy needed hundreds of steps at alpha = 0.05.

> [!QA]
> Q: Applied design: you have 2 billion logged ad impressions, 40 features, and a model that must retrain hourly. Batch, SGD, mini-batch, or normal equations?
> A: Mini-batch SGD, batch size 256 to 1024. Normal equations are dead: X^T X is 40 by 40, which is fine, but forming it over 2 billion rows every hour is the expensive part, and streaming mini-batches adapt to drift. Full-batch GD takes one step per hour, which cannot track a moving target. Pure SGD at B = 1 wastes the hardware: 256 examples cost nearly the same wall-clock as 1 on a GPU. Mini-batch is the only option that is fast, parallel, and adaptive.
> Follow-up: What breaks first as the feature count grows from 40 to 40,000?
> A: The O(n) per-step cost grows linearly, so steps get 1,000 times more expensive. At 40,000 dense features you need sparse representations or feature hashing, or the hourly retrain misses its window. The algorithm stays the same. The data plumbing changes.

> [!QA]
> Q: Why does feature scaling matter for gradient descent but not for the normal equations?
> A: Gradient descent uses one alpha for every knob. If living area spans 1,000 to 3,000 and bedrooms span 1 to 5, the area gradient is ~1,000 times larger, so one alpha cannot suit both: it either crawls on bedrooms or explodes on area. Scaling both to 0-1 makes the loss bowl round and one alpha works everywhere. The normal equations solve the linear system exactly in one shot. No steps means no step size, so scale is irrelevant. The price of scale-freedom is the O(n^3) inverse.
> Follow-up: What is the cheapest correct scaling?
> A: Subtract the mean and divide by the range or standard deviation, per feature, computed on the training set only. Apply the same transform at prediction time. Never compute scaling statistics on the test set: that leaks test information into training.

## Coverage map: every lecture claim and where it lives

| Lecture claim | Covered in | File line |
|---|---|---|
| Ames housing job: price from past sales | The job: price a house in Ames, Iowa | L34 |
| Example, features, target, hypothesis, parameters, regression | The supervised setup, piece by piece | L52 |
| Hypothesis worked on two houses | the hypothesis, worked | L81 |
| Loss J(theta) = 1/(2m) sum of squared errors | The supervised setup, piece by piece | L52 |
| Squared vs absolute loss, outlier price | three losses, one toy | L120 |
| Calculus-first vs the iterative workhorse | First attempt: solve it with calculus | L139 |
| GD update: theta_j := theta_j - alpha x error x feature | Gradient descent, by hand | L156 |
| Vectorized update theta := theta - alpha(1/m)X^T(Xtheta - y) | the vectorized update | L187 |
| Stopping rules: gradient norm, plateau, budget | when to stop | L202 |
| Three-iteration trace: J 6.33 -> 3.32 -> 1.75 -> 0.028 | the descent trace, three iterations | L248 |
| Feature scaling: narrow valley to round bowl | feature scaling, the hidden dial | L266 |
| Alpha too small crawls; too large diverges (4, -8, 16, -32, 64, -128) | Where it breaks: the learning rate | L290 |
| Batch cost: full data pass per step | batch cost accounting | L327 |
| SGD: one example per step, noisy but m times cheaper | Stochastic gradient descent | L343 |
| E[single-example gradient] = batch gradient, proved | SGD's unbiased noise, worked | L372 |
| Mini-batch B = 32-256, the production middle | mini-batch, the middle path | L390 |
| Normal equations theta = (X^T X)^-1 X^T y | The normal equations | L412 |
| Four-step matrix derivation | the normal equations, derived | L427 |
| Singular X^T X from redundant features (det ~ 0) | when the inverse fails, worked | L461 |

Lecture video cmNIMjPYdgM verified real via YouTube search (Lecture 2:
Supervised Learning Setup, Stanford Online). oEmbed 401 = embedding
disabled by owner, linked not embedded. Explainer embed nk2CQITm_eo
verified via oEmbed.

## Recap: the whole lesson on one screen

1. **The job.** Price Ames houses from past sales. Find the best
   line through the points.
2. **The setup.** Example (x, y). Hypothesis h_theta(x), a line with
   knobs theta. Regression predicts a number.
3. **The loss chip.** J(theta) = 1/(2m) sum of squared errors.
   Smaller is better. Owned by this course, reused everywhere.
4. **Gradient descent.** Step each knob opposite its gradient:
   theta_j := theta_j - alpha * error * feature, summed. Hand toy:
   J fell 6.33 to 3.32 in one step.
5. **The learning rate.** Too small crawls (1,000 steps for a
   1-D bowl). Too large diverges (4, -8, 16, -32, 64). Watch the loss
   curve. Bouncing means turn alpha down.
6. **The key question.** What if one step used one example instead
   of the whole internet?
7. **SGD.** One example per step: m times cheaper, noisy, wins on
   huge data. Shuffle every epoch.
8. **Normal equations.** theta = (X^T X)^-1 X^T y. Hand toy: the
   best line is 1/3 + 1.5x, J = 0.028, exact. Costs O(n^3), needs
   invertibility.
9. **The honest price.** Exact answers need inverses. Inverses need
   small n and full rank. Iterative answers need alpha tuning.
10. **Feature scaling.** Unscaled features make a narrow valley.
    Scale to 0-1 and one alpha works for every knob.
11. **Mini-batch.** B = 32 to 256: 64 times less noise than SGD at
    nearly the same hardware cost. The production default.
12. **The pattern.** Error times feature, summed, recurs in logistic
    regression and backprop. Learn to spot it.

## Official sources and further reading

**Official:**
- Lecture 2 video, Stanford Online YouTube:
  - [Chris Ré builds the](https://www.youtube.com/watch?v=cmNIMjPYdgM)
  supervised setup on the Ames housing data, derives gradient
  descent and SGD, and sketches the normal equations.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the full
  rigorous treatment, including the matrix derivation of the normal
  equations.
- Ames Housing dataset documentation:
  - [the real dataset behind](https://jse.amstat.org/v19n3/decock.pdf)
  the lecture's example.

**Caveats from these sources.** The lecture presents the normal
equations briefly "just for notation" and points to the course notes
for rigor. The full derivation with matrix calculus lives there. The
lecture's gradient descent demo uses the Ames data with many
features. The 3-house toy in this lesson is an original miniature
with the same update rule. "Look at your data" is the lecture's
standing advice and applies before any fitting.

## Connections to the other courses

- **CS229 L03:** the same loss derived from probability: Gaussian
  noise makes least squares the maximum-likelihood answer.
- **CS229 L06:** what goes wrong when the line is too flexible:
  bias, variance, and ridge regression fixing the singular matrix.
- **CS229 L08:** the gradient idea scaled up: backpropagation
  computes gradients for millions of knobs at the cost of one
  forward pass.
- **CS336:** gradient descent at scale: how the update actually
  runs on GPUs, and MFU (Model FLOPs Utilization), the fraction of
  peak FLOPs actually achieved.
- **CS224N L01:** the same SGD loop training word vectors and
  language models.
