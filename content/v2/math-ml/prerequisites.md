# prerequisites.md, math-ml prerequisite graph

Date: 2026-10-06. Shared 24-module bridge (P01-P24) is not built yet. This file gives the course-level graph plus self-contained remediation
for the P01/P02 essentials that U01 depends on.

## Module index (from the prerequisite atlas)

P01 numeracy/algebra/notation. P02 Python and scientific software.
P03 vectors/geometry/linear maps. P04 spectral and numerical linear algebra.
P05 scalar and multivariable calculus. P06 probability.
P07 statistical estimation and uncertainty. P08 information theory.
P09 optimization and constrained problems. P10 ML foundations and evaluation.
P11 neural networks and autodiff. P12 PyTorch, tensors, numerical stability.
P13 language and sequence modelling. P14 transformer mechanics.
P15 hardware and computer architecture. P16 distributed systems.
P17 reinforcement learning. P18 Bayesian inference, latent variables, sampling.
P19 retrieval. P20 tools, APIs, agent state. P21 security and privacy.
P22 experimental method. P23 economics. P24 production ML.

## Dependency chains into this course

- U01 needs P01, P02.
- U02 needs P03 (needs P01), P04 (needs P03).
- U03 needs P05 (needs P01, P03), P09 (needs P04, P05).
- U04 needs P06 (needs P01), P07 (needs P04, P06), P08 (needs P06, P07).
- U05 needs P07, P09, P10 (needs P02, P07, P09).
- U06 needs P03, P07, P09.
- U07 needs P06, P07, P10.
- U08 needs P11 (needs P02, P03, P05, P09, P10), P12 (needs P02, P11),
  P13 (needs P06, P08, P10), P14 (needs P11, P12, P13).
- U09 needs P04, P07, P18 (needs P06, P07, P08, P09).
- U10 needs P08, P18, P22 (needs P07, P10).

## Local remediation: P01 essentials used by U01

### R1, arithmetic and fractions

Concrete: 1/2 + 1/3 = 5/6. Rule: add fractions through a common
denominator, never across numerators alone. Check: 5/6 = 0.8333.... Direct float sum gives 0.8333333333333333 (computed 2026-10-06).

### R2, powers and roots

Concrete: 2^10 = 1024. The 3rd root of 8 = 2. Rule: x^(a*b) = (x^a)^b.
Check: 2^(2*5) = 4^5 = 1024.

### R3, indices versus exponents

An index picks an item: x_2 is the second item of x.
An exponent repeats multiplication: x^2 = x * x.
Concrete: for x = [10, 20, 30], x_2 = 20. X_2^2 = 400.

### R4, function input and output

A function maps each input to exactly one output.
Concrete: f(x) = 2x + 3. F(4) = 11.
Misconception: the letter x has no meaning by itself. It names the input
slot of the function where it is defined.

### R5, substitution

Replace the symbol by the value, then compute.
Concrete: with a = 7 and b = 3, a^2 - b^2 = 49 - 9 = 40.

## Local remediation: P02 essentials used by U01

### R6, Python values and names

A name points at a value. Assignment stores. Expression reads.
Concrete: n = 6. M = n * 2. M holds 12.

### R7, lists and loops

A list holds items in order. A loop visits each item once.
Concrete: total = 0. For v in [2, 5, 7]: total = total + v. Total = 14.

### R8, functions and scope

A function takes inputs and returns an output. Names inside the function
do not leak out.
Concrete: def square(t): return t * t. Square(9) = 81.

### R9, NumPy arrays and shapes

An array has a shape: the count of items along each axis.
A vector of 4 items has shape (4,). A 2 by 3 table has shape (2, 3).
Concrete: numpy.arange(6).reshape(2, 3) has shape (2, 3).

### R10, vectorized operations

Operations apply to every item at once, without a written loop.
Concrete: numpy.array([1, 2, 3]) * 2 = [2, 4, 6].

## Diagnostic routing

diagnostics/diagnostic-01-math-language.md probes R1-R10 and the U01
concepts. A score below 70 percent sends the learner back to the
matching R-section before U02.

## Local remediation: P03 essentials used by U02

### R11, dot product from zero

Take v = [3, 4] and w = [1, 0]. The dot product multiplies matching
items and adds: v dot w = 3*1 + 4*0 = 3. Rule: the dot product is a
scalar that measures shared direction. Check: computed 2026-10-06,
numpy 1.26.4, float64.

### R12, norms and unit vectors

The length of v = [3, 4] is sqrt(3^2 + 4^2) = 5. Divide by the length
to get the unit vector [0.6, 0.8], whose length is 1. Rule: a unit
vector keeps direction and forgets scale.

### R13, matrix times vector as column combination

M = [[1, 2], [3, 4]] times x = [5, 6] equals 5 times column one plus
6 times column two: 5*[1, 3] + 6*[2, 4] = [17, 39]. Rule: the product
is always a combination of the columns of M.

### R14, shape rule for matrix multiplication

(m, n) times (n, p) gives (m, p). The inner counts must match. Rule:
(2, 3) times (3, 4) gives (2, 4). A (2, 3) times a (4, 2) is an error,
not a matrix.

### R15, transpose

The transpose flips rows to columns: [[1, 2], [3, 4]]^T =
[[1, 3], [2, 4]]. Rule: (A B)^T = B^T A^T, order reverses.

### R16, orthogonality

Two vectors are orthogonal when their dot product is 0: [1, 0] dot
[0, 1] = 0. Rule: orthogonal means perpendicular, nothing more.

## Local remediation: P04 essentials used by U02

### R17, symmetric matrices

A matrix is symmetric when A^T = A. For E = [[2, 1], [1, 2]] the
eigenvectors are [1, 1]/sqrt(2) and [1, -1]/sqrt(2), and their dot
product is 0. Rule: symmetric matrices have perpendicular eigenvectors.

### R18, eigenpairs from zero

A v = lambda v says v keeps its line through A, stretched by lambda.
For E = [[2, 1], [1, 2]], eigenvalues are 3 and 1 with eigenvectors
[1, 1]/sqrt(2) and [1, -1]/sqrt(2). Computed 2026-10-06, float64.

### R19, SVD exists for every real matrix

Every real matrix factors as U Sigma V^T with Sigma diagonal and
non-negative. For M = [[3, 2, 2], [2, 3, -2]] the singular values are
5 and 3. Rule: singular values are the true stretch factors of M.

### R20, conditioning from zero

A matrix with one huge and one tiny stretch factor blows up small
input errors. For K = [[1, 1], [1, 1.001]] the stretch factors are
2.0005 and 0.0004999, so kappa = 4002.0. Rule: kappa is the error
amplifier.

### R21, solve beats inverse

For D x = b use numpy.linalg.solve(D, b), not inv(D) @ b. The inverse
form does extra work and squares the error risk. Check the residual
norm(D @ x - b), not the formula.

### R22, low-rank idea

Keep the large singular values, drop the small ones. For M above,
keeping only sigma = 5 leaves a reconstruction error of 3.0, exactly
sigma 2. Rule: the dropped value names the price.

## Local remediation: P05 essentials used by U03

### R23, derivatives from zero

The derivative is the slope of the tangent line. For f(x) = 3x^2 at
x = 2, the slope is 12. Rule: d/dx x^n = n x^(n-1). Check:
finite difference with h = 1e-7 gives 11.999999989242838
(computed 2026-10-06).

### R24, partial derivative mechanic

Hold all but one variable fixed, then differentiate in the free one.
For f(x, y) = x^2 + 3xy + y^2, treat y as a constant: df/dx =
2x + 3y. Rule: a partial derivative is an ordinary derivative with
frozen partners.

### R25, scalar chain rule

For y = (2x+1)^3, name u = 2x + 1. Then dy/dx = dy/du * du/dx =
3u^2 * 2. At x = 2: 75 * 2 = 150. Rule: multiply the link slopes.
Check: finite difference gives 150.00000004761205
(computed 2026-10-06).

### R26, gradient as an arrow

The gradient of f(x, y) = x^2 + 3xy + y^2 is the arrow [2x + 3y,
3x + 2y]. At (2, 1) it is [7, 8] and points in the steepest-uphill
direction. Rule: the gradient is the direction of fastest increase.
descent walks the opposite way.

### R27, Taylor idea

Near a point, a smooth function looks like its tangent line, then
like a parabola. e^x near 0: 1 + x, then 1 + x + x^2/2. Rule: each
term fixes the leftover curve of the last. Check: at x = 0.5 the
errors are 0.1487 then 0.0237 (computed 2026-10-06).

### R28, area under a curve

The integral of a density over an interval is the probability of that
interval. For the uniform on [0, 2], the area over [0, 1] is 0.5.
Rule: total area under a density is 1. Area is probability.

## Local remediation: P09 essentials used by U03

### R29, objective and feasible set

An optimization problem names what to minimize and where you may
look. Toy: minimize x^2 over x >= 1. The objective is x^2. The
feasible set is [1, infinity). Rule: the answer must be both low and
allowed.

### R30, gradient descent mechanic

Repeat w = w - eta * slope. For J(w) = (w-3)^2 with eta = 0.1 from
w = 0: step 1 gives w = 0.6. Rule: each step moves against the
slope by eta times its size. Check: after 10 steps w =
2.6778774528 (computed 2026-10-06).

### R31, learning rate regimes

The step size picks the regime. On J(w) = (w-3)^2: eta = 0.05
crawls, eta = 0.5 lands in one step, eta = 1.5 explodes to J =
150994944.0 (computed 2026-10-06). Rule: below the boundary the
walk converges. Above it the walk diverges.

### R32, convexity from zero

f(x) = (x-1)^2 has one bowl: every downhill walk ends at x = 1.
h(x) = x^4 - 3x^2 has two bowls: the walk can end at 1.2247 or
-1.2247 depending on the start. Rule: one bowl means the bottom is
unique. Many bowls mean the start decides.

### R33, constraints and the Lagrangian idea

A fence x >= 1 turns min x^2 into the fenced answer x = 1. The
Lagrangian L = x^2 - lam*(x-1) folds the fence into the objective
with a fine lam. Rule: lam charges for fence violation. At the
answer the fine is 2 per unit (computed 2026-10-06).

### R34, Newton idea

Newton jumps to the bottom of the local parabola instead of stepping
down the slope. On (w-3)^2 one jump from 0 lands at 3. Rule: use
curvature when you can afford the Hessian. Use slopes when you
cannot.

## Diagnostic routing (extended RUN 3)

diagnostics/diagnostic-01-math-language.md probes R1-R10. Learners
entering U03 should self-check R23-R34 above: a miss on R24 or R30
sends the learner back to the matching R-section before the U03
lesson.

## Local remediation: P07/P10 essentials used by U05 (added RUN 4)

### R35, empirical versus population risk

The empirical risk is the average loss on the sample. The population
risk is the expected loss over the truth. Toy: 4 points, 1 error,
R_hat = 0.25. Rule: training minimizes the first. Deployment pays
the second.

### R36, loss versus metric

The loss is what the optimizer minimizes. The metric is what the
business grades. They need not match. Toy: squared loss on labels
0/1 buys the mean 2/3, not a class. Rule: train on a smooth
surrogate when you must, report the true metric on held-out data.

### R37, train/validation/test discipline

Split before fitting. Lock the model. Score once. Toy: train MSE
0.4821, test MSE 0.8653 on the lesson's line toy. Rule: a test set
touched during fitting is a training set.

### R38, cross-validation mechanic

Rotate the held-out part k times, average the k honest scores.
Toy: LOOCV on 4 points picks n lambda = 2 over 0 (1.1806 versus
1.8515). Rule: select on the honest score, never on the train
score.

### R39, bias and variance of an estimator

MSE = bias^2 + variance. Toy: Laplace (k+1)/(n+2) has bias
-0.0667, variance 0.02333, MSE 0.02778, beating the MLE's 0.0525
at n = 4. Rule: a little bias can buy a lot less variance.

### R40, leakage

Information from the future smuggled into training. Toy: a
label-copy feature gives train accuracy 1.0 and deployment
accuracy 0.6. Rule: build every feature from information available
at decision time.

### R41, capacity and overfitting

More flexible models fit training points better and new points
worse, past a point. Toy: degree-3 polynomial on 4 points gets
train MSE ~0 and test MSE 28.56. Rule: capacity is relative to n.

### R42, regularization penalties

A penalty taxes weight size: J = RSS + n lambda w^2. Toy: n
lambda = 2 moves w from 0.9286 to 0.8125. Rule: standardize
features first, or the tax punishes units.

### R43, priors as regularizers

A Gaussian prior is a quadratic penalty in log space. Toy: prior
N(0, 0.5) with noise variance 1 gives w_MAP = 0.8125, the ridge
answer. Rule: lambda = sigma^2/(n tau^2). A tight prior is a
strong belief, not a neutral default.

### R44, distribution shift

Train and deploy distributions differ. Toy: same line, x range
[0,3] then [4,6]: MSE 0.0986 then 0.5548. Rule: monitor inputs,
not just the loss. Reweighting helps covariate shift with
overlapping support. Nothing reweights concept shift.

## Diagnostic routing (extended RUN 4)

Learners entering U05 should self-check R35-R44 above: a miss on
R35 or R39 sends the learner back to the matching R-section before
the U05 lesson. A miss on R40 sends the learner to lesson C11
before any applied work.

## Local remediation: P03/P07/P09 essentials used by U06

### R45, normal equations from zero

RSS = sum (y_i - b - w x_i)^2. Setting both partial derivatives
to zero gives X^T X w = X^T y. Toy: x = [0,1,2,3], y = [0,1,3,2]
gives w_hat = [0.3, 0.8], RSS 1.8. Rule: squares give linear
equations. Absolute values do not.

### R46, Gaussian noise buys squares

If y = Xw + eps with eps IID N(0, sigma^2), the log-likelihood
is a negative sum of squares plus constants. Maximizing it
minimizes RSS. Toy: sigma2_hat = 1.8/4 = 0.45 on the R45 data.
Rule: the loss follows the noise story, not the other way.

### R47, sigmoid and log-odds

sigmoid(z) = 1/(1+e^-z) maps any score to (0, 1). Its inverse is
logit(p) = log(p/(1-p)): the log-odds. Toy: logit(0.7) =
0.8473, sigmoid back 0.7. Rule: linear scores live in
log-odds space. Probabilities live in [0, 1].

### R48, softmax normalizes

p_k = e^{z_k}/sum_j e^{z_j}. The probabilities sum to 1 by
construction. Toy: logits [2,1,0] give [0.6652, 0.2447,
0.0900]. Rule: subtract max(z) before exp, or float64
overflows.

### R49, kernels are dot products elsewhere

k(x, z) = (1 + xz)^2 equals phi(x) . phi(z) for phi(x) = [1,
sqrt(2) x, x^2]. Check: 1 + 12 + 36 = 49. Rule: the sqrt(2)
carries the cross-term coefficient. The naive map [1, x, x^2]
gives 43, not 49.

### R50, Gram matrices must be PSD

K_ij = k(x_i, x_j) must have non-negative eigenvalues, or no
feature space exists. Toy: the RBF Gram on 3 points has
eigenvalues 0.3911, 0.9656, 1.6433. Rule: test PSD before any
dual solve. Watch the condition number too.

### R51, margin geometry

For w . x + b = 0, the distance from a point to the boundary is
|w . x_i + b|/||w||. The street width is 2/||w||. Toy: w = [1,
1] gives width 1.4142. Rule: functional margin scales with w. 
geometric margin does not.

### R52, hinge and the price C

hinge_i = max(0, 1 - y_i(w . x_i + b)). Objective: 0.5||w||^2
+ C sum hinge_i. Toy: one violator with hinge 1.4. C = 1 gives
2.4, C = 100 gives 141.0. Rule: C prices violations. C ->
infinity recovers the hard margin.

### R53, duality in one line

The dual maximizes over alpha_i >= 0 with sum alpha_i y_i = 0. 
w = sum alpha_i y_i x_i. Toy: two symmetric points give a* =
0.25, primal 0.25 = dual 0.25. Rule: drop the balance
constraint and the gap (0 vs 1.0) exposes it.

### R54, scale before you solve

kappa(X^T X) = 3.35e6 raw vs 2.62 standardized on the lesson
toy. The solve is the same math. The digits differ. Rule:
standardize columns before normal equations, kernels, and
k-means alike.

## Diagnostic routing (extended RUN 5)

Learners entering U06 should self-check R45-R54 above: a miss on
R45 or R46 sends the learner back before the U06 lesson. A miss
on R50 or R54 sends the learner to U02-C10/C12 first. A miss on
R51-R53 sends the learner to U03-C09/C10 first.

## Local remediation: P06/P07/P10 essentials used by U07 (added completion run)

### R55, recursive partition idea

A tree splits the input space again and again. Each split
picks one feature and one threshold. The leaf regions form
a partition: every point lands in exactly one leaf.
Concrete: two splits make at most four regions. Rule: the
path from root to leaf is the region's definition. Check:
count leaves, count regions, they match.

### R56, bootstrap from zero

Draw n items from n with replacement. Some items repeat,
some never appear. Concrete: n = 4, seed 7: [2, 2, 3, 3].
Rule: the chance one item misses a replicate is
(1 - 1/n)^n, about 1/e for large n. Check: each replicate
has exactly n draws.

### R57, class scores from counts

Counts become probabilities by division: [4, 1] gives
[0.8, 0.2]. Scores become decisions by threshold or
majority vote. Concrete: majority of [4, 1] is class 0.
Rule: probabilities first, decisions second. Check:
probabilities sum to 1.

## Diagnostic routing (extended completion run)

Learners entering U07 should self-check R55-R57: a miss on
R56 sends the learner to C05 before the U07 lesson. A miss
on R57 sends the learner to U04-C01 first.

## Local remediation: P11 essentials used by U08 (added completion run)

### R58, linear layer and activation

A layer computes z = Wx + b, then a = f(z) per item.
Concrete: W = [[1, 0], [0, 1]], b = [0, -1], x = [1, 2]:
z = [1, 1], relu gives [1, 1]. Rule: the affine map
mixes, the activation bends. Check shapes: (2,2)(2,) ->
(2,).

### R59, reverse-mode differentiation

One forward pass stores intermediates. one backward pass
applies the chain rule to get every parameter gradient.
Concrete: the C02 toy gives six gradients from one
backward walk. Rule: cost is about two forward passes,
not n. Check every backward pass with finite
differences.

### R60, tensor axes and broadcasting (P12)

An axis is one dimension of a tensor. Broadcasting
stretches size-1 axes to match. Concrete: (3,1) + (3,4)
gives (3,4). Rule: align trailing axes. stretch only
size 1. Check: (3,2) + (4,) is an error, not (3,4).

### R61, stable softmax and log-sum-exp (P12)

Softmax overflows when scores are large. Subtract the
max first: the result is identical, the exp is safe.
Concrete: scores [1000, 1001] -> subtract 1001 ->
[0.2689, 0.7311]. Rule: max-subtraction is exact, not
approximate. Check: weights sum to 1.

### R62, autoregressive factorization (P13)

A sequence probability factors left to right: p(x_1..x_T)
= prod_t p(x_t | x_<t). Concrete: three tokens with
conditionals 0.5, 0.4, 0.9 give 0.18. Rule: order is
part of the model. Check: the factors condition on
strictly earlier tokens only.

### R63, Q/K/V projections (P14)

Queries ask, keys advertise, values carry content. All
three are learned linear maps of the same input.
Concrete: the C09 toy: QK^T scores, softmax weights,
weighted V sums. Rule: attention is a differentiable
dictionary. Check: weight rows sum to 1.

### R64, residual paths and norm placement (P14)

A residual block outputs x + sublayer(x): the network
can learn the identity by pushing the sublayer to zero.
Concrete: if sublayer outputs 0, the block copies x.
Rule: the add needs equal shapes. Check: norm before
or after the sublayer is a design choice with
different training behavior.

## Diagnostic routing (extended completion run)

Learners entering U08 should self-check R58-R64: a miss
on R58 or R59 sends the learner back before the U08
lesson. A miss on R61 sends the learner to U06-C04
first. A miss on R63 sends the learner to reread C09
before C10.

## Local remediation: P08/P18/P22 essentials used by U10 (added completion run)

### R70, falsifiable hypothesis (P22)

A hypothesis that an experiment can prove wrong.
Concrete: "two-step lookahead beats greedy at depth 3
on parity tasks" is falsifiable by the accuracy gap
over 20 seeds. Rule: no falsification condition, no
hypothesis. Check: write the losing condition first.

### R71, baseline and ablation (P22)

A baseline is the simplest method that could work. an
ablation removes one part and measures the drop.
Concrete: the C12 majority baseline (0.5) against the
stump (1.0). Rule: the baseline runs before the fancy
method. Check: every claim names its baseline.

### R72, seeds and uncertainty (P22)

A seed fixes the random draws. uncertainty says how
much the number wobbles across seeds. Concrete: the
C09 ratio is 4.2887 at seed 11 and 3.9594 at seed 12.
Rule: report the seed with the number. Check: rerun
with a new seed and watch the wobble.

### R73, predicted versus measured (P22)

Predicted comes from theory. measured comes from code.
Concrete: predicted ratio 4.0, measured 4.2887.
Rule: label each number at the point of use. Check:
the gap between them is the interesting part.

### R74, Jensen and ELBO recap (P08/P18)

Jensen: for concave phi, phi(E[X]) >= E[phi(X)]. The
ELBO applies it to log sum_z p(x,z) with weights q.
Concrete: the C07 numbers give -0.7340 >= -0.8779.
Rule: the inequality direction follows the concavity
of log. Check: the gap equals KL(q||posterior).

### R75, sampling versus density (P08)

Sampling draws new points. density scores given points.
Concrete: ten ancestral draws vs the Phi formula for
the CDF. Rule: match the tool to the question. Check:
samples answer "make", densities answer "how likely".

### R76, support and absolute continuity (P18)

A density ratio p/q needs q > 0 wherever p > 0, or the
ratio explodes. Concrete: q = [1, 0] against a
posterior with mass on z = 1 breaks the ELBO sum.
Rule: check support before trusting a variational
claim. Check: no zero of q sits on positive mass.

### R77, negative results (P22)

A clean "it did not work" with the protocol attached.
Concrete: the f04 rise from B = 5 to B = 25 on six
validation points is a negative result for "more
trees always help". Rule: negative results need the
same protocol rigor as positive ones. Check: the
regime qualifier travels with the result.

### R78, reproducibility protocol (P22)

Seed, versions, shapes, dtypes, trial counts: the five
items that let a stranger rerun the number. Concrete:
"numpy 1.26.4, float64, seed 7" heads every toy in
this course. Rule: a number without its protocol is
decoration. Check: rerun and compare.

## Diagnostic routing (extended completion run)

Learners entering U10 should self-check R70-R78: a miss
on R70 or R73 sends the learner back before the U10
lesson. A miss on R74 sends the learner to U04-C08
first. A miss on R76 sends the learner to reread C07
before C08.
