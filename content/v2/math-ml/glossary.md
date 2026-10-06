# glossary.md, math-ml term index (RUN 5: U01-U06, U04a-c, U09 leaves)

Date: 2026-10-06. Each term is defined at first use in the lesson. This file is the quick index.

- set: a collection of distinct items. Order does not matter.
- function: a rule that maps each input to exactly one output.
- domain: the set of allowed inputs of a function.
- image: the set of outputs a function actually produces.
- notation: the agreed symbols used to write mathematics.
- unit: the kind of quantity a number measures, for example meters.
- dimension check: a test that both sides of an equation carry the same units.
- sum: the total of a list of numbers.
- product: the result of multiplying a list of numbers.
- index: a position label, for example the 2 in x_2.
- exponent: a count of repeated multiplication, for example the 2 in x^2.
- logarithm: the inverse of exponentiation. Log_b(x) answers "b to what
  power gives x".
- vector: an ordered list of numbers treated as one object.
- norm: the length of a vector.
- dot product: the sum of item-wise products of two vectors.
- axis: one direction of a table or array. Axis 0 runs down rows,
  axis 1 runs across columns.
- shape: the tuple that counts items along each axis.
- assertion: a runtime check that states what must be true at that point.
- floating point: the finite-precision number format computers use.
- precision: how many digits a number format can hold.
- seed: the starting value that makes a random generator repeatable.
- reproducibility: the property that the same code and seed give the same
  result on every run.

## U02 terms

- subspace: a flat set through the origin, closed under addition and scaling.
- span: the set of all linear combinations of a list of vectors.
- dimension (of a subspace): the count of independent directions in it.
- rank: the count of independent rows or columns of a matrix. The count of
  nonzero singular values.
- nullspace: the set of vectors a matrix sends to zero.
- column space: the set of all outputs of a matrix.
- rank-nullity: rank plus nullity equals the input dimension.
- projection: the shadow of a vector on a line or subspace, dropped straight down.
- residual: the leftover after projection. Orthogonal to the projection target.
- linear map: a function on vectors that preserves addition and scaling.
- determinant: the signed area (volume) scale factor of a square matrix.
- singular matrix: a square matrix with determinant zero. No inverse.
- inverse: the matrix that undoes another: A^{-1} A = I.
- pseudoinverse: the best possible undo for any matrix. Inverts kept directions,
  returns zero on killed ones.
- eigenvector: a nonzero vector that keeps its line through a matrix.
- eigenvalue: the stretch factor of an eigenvector.
- SVD: the factorization M = U Sigma V^T. Sigma holds the singular values.
- singular value: a true stretch factor of a matrix. Always non-negative.
- PSD (positive semidefinite): a symmetric matrix with x^T A x >= 0 for all x.
  equivalently all eigenvalues >= 0.
- quadratic form: the scalar x^T A x. Measures curvature along x.
- covariance: the average product of deviations from the mean.
- correlation: covariance divided by the two standard deviations. In [-1, 1].
- condition number (kappa): sigma_max / sigma_min. The worst-case error amplifier
  of a linear solve.
- low-rank approximation: keeping only the top singular values and vectors.

## U04a block terms

- sample space: the set of all possible outcomes.
- random variable: a function mapping each outcome to a number.
- event: a set of outcomes.
- PMF: the probability table of a discrete random variable. Rows sum to 1.
- PDF: the density curve of a continuous random variable. Area under it is 1.
- CDF: the running total P(X <= x). Climbs from 0 to 1.
- joint distribution: the full probability table of two or more variables.
- marginal distribution: the distribution of one variable after summing out others.
- conditional distribution: the distribution of one variable given another's value.
- expectation: the probability-weighted average. The long-run center.
- variance: the average squared distance from the expectation.
- standard deviation: the square root of variance. In original units.
- sample: the draws actually observed.
- distribution (true): the unknown rule generating the data.
- IID: independent and identically distributed. Each draw from the same P,
  knowing one draw tells nothing about another.
- likelihood: P(data | parameter), read as a function of the parameter.
- log-likelihood: the log of the likelihood. Turns products into sums.
- MLE: the parameter value maximizing the likelihood.
- density estimation: guessing the full distribution from a sample.
- histogram: binned counts of a sample. A density estimate.
- estimation error: the gap between estimate and truth. Typically ~ 1/sqrt(n).

## U03 terms

- partial derivative: the slope of a function when every input but
  one is frozen.
- chain rule: the total slope is the product of the link slopes.
- gradient: the vector of partial derivatives. It points steepest uphill.
- Jacobian: the table of all first derivatives of a vector function.
- Hessian: the table of second derivatives. Names curvature.
- Taylor approximation: the local polynomial portrait of a smooth
  function.
- convex: one bowl. The chord between any two points sits above the
  graph.
- least squares: the fit that minimizes the mean squared error.
- gradient descent: repeated steps against the slope.
- SGD: gradient descent with one random sample per step.
- Newton method: jumps to the bottom of the local parabola using
  curvature.
- learning rate: the step size of an optimizer.
- constraint: a rule that forbids some answers.
- Lagrangian: the objective plus the fine for fence violation.
- Lagrange multiplier: the fine per unit of violation.
- KKT conditions: the four checks that certify a fenced optimum.
- dual function: the best the Lagrangian can do over x, as a
  function of the multiplier.
- gradient check: a finite-difference second opinion on a coded
  gradient.

## U04b terms

- entropy: the average surprise of one draw from a distribution.
- KL divergence: the extra surprise of using the wrong distribution.
  not symmetric.
- cross-entropy: entropy plus KL. The total surprise under the wrong
  model.
- ML estimate: the parameter that makes the observed data most
  likely.
- score: the derivative of the log-likelihood. Zero at the MLE.
- multivariate Gaussian: the bell in many dimensions, named by a
  mean vector and a covariance matrix.
- covariance matrix: the table of pairwise spreads. Its eigenvalues
  name the stretch directions.
- mixture distribution: a weighted menu of component distributions.

## U05 terms

- population risk: the expected loss over the true distribution.
  The deployment price.
- empirical risk: the average loss on the sample. The training
  scoreboard.
- ERM: empirical risk minimization. Pick the rule in the class
  with the smallest empirical risk.
- hypothesis class: the set of rules the learner may pick from.
- loss: the price list handed to the optimizer.
- metric: the number the business grades. Need not equal the loss.
- bias (of an estimator): how far the average estimate sits from
  truth.
- variance (of an estimator): how much the estimate wobbles across
  datasets.
- penalty: a tax on weight size added to the training objective.
- ridge: least squares plus a quadratic penalty.
- MAP estimate: the parameter that maximizes posterior belief.
- train/test split: fit on one part, score once on the other.
- cross-validation: rotate the held-out part and average the
  honest scores.
- capacity: how many functions the model class can express.
- overfitting: low training error with high test error from
  excess capacity.
- learning curve: train and validation error plotted against n.
- leakage: future information smuggled into training.
- distribution shift: the deploy distribution differs from the
  train distribution.
- covariate shift: the x distribution moves.
- concept shift: the x-to-y relation moves.

## U04c terms

- Bayes classifier: the rule that predicts the most probable
  class at each x. Optimal for 0-1 loss.
- Bayes error: the smallest achievable population risk. The floor.
- Neyman-Pearson rule: maximize detection under a false-alarm
  budget.
- minmax classifier: minimize the worst of the two error rates.
- ROC curve: the tradeoff curve of TPR against FPR over
  thresholds.
- AUC: the area under the ROC curve. The probability a random
  positive outscores a random negative.
- latent variable: an unobserved quantity the model sums over.
- incomplete likelihood: the likelihood with latent variables
  summed out. Log outside the sum.
- EM algorithm: alternate soft imputation (E) and weighted fitting
  (M). Monotone in the incomplete likelihood.
- responsibility: the posterior probability that a point belongs
  to a component.
- Jensen inequality: log of an average is at least the average of
  the logs. Builds the EM bound.
- identifiability: different parameters name different
  distributions. Fails under label switching.
- label switching: permuting mixture labels leaves the likelihood
  unchanged.
- degeneracy (mixture): a component collapses onto one point and
  the likelihood explodes.
- Parzen window: kernel density estimation. Every point votes
  with a smooth puff.
- bandwidth: the reach of the Parzen puff. The bias-variance knob.
- nearest neighbor classifier: predict the label of the closest
  training point. k-NN votes among k.

## U06 terms

- normal equations: X^T X w = X^T y. The stationarity condition of
  the residual sum of squares.
- RSS: residual sum of squares. The minimized quantity in OLS.
- sigmoid: 1/(1+e^-z). Maps scores to (0, 1).
- logit: log(p/(1-p)). The inverse of the sigmoid. Log-odds.
- softmax: e^{z_k}/sum e^{z_j}. Normalizes logits to a
  probability vector.
- cross-entropy loss: -log p_true. The surprise at the true class.
- GLM: generalized linear model. Linear score through a link
  function, fit by likelihood.
- canonical link: the link paired with an exponential family:
  identity for Gaussian, logit for Bernoulli.
- kernel: a function k(x, z) that equals a dot product in some
  feature space.
- feature map: phi. The explicit map into the kernel space.
- Gram matrix: K_ij = k(x_i, x_j). Must be PSD.
- margin (geometric): 2/||w||. The width of the empty street.
- support vector: a training point on the margin edge. Its alpha
  is positive.
- hinge loss: max(0, 1 - y(w.x + b)). Zero past the margin.
- slack: xi_i. The margin violation of point i in the soft-margin
  objective.
- SVM dual: maximize over alphas >= 0 with sum alpha_i y_i = 0.
  w = sum alpha_i y_i x_i.
- subgradient: a slope that stays below a convex function at a
  kink. The hinge's optimizer.

## U09 leaf terms

- k-means objective: within-cluster sum of squares. Lloyd
  iterations lower it monotonically.
- Lloyd iteration: assign to nearest center, then move centers to
  group means.
- standardization: zero mean, unit variance per column. The
  default honest ruler for distances.
- PCA: eigendecomposition of the covariance. Keep the top
  directions by variance.
- variance explained: eigenvalue share of the total. 99.72
  percent for PC1 on the lesson toy.
- reconstruction error: squared distance from a point to its
  low-rank reconstruction. Equals dropped eigenvalues over d.
- factor model: x = W z + mu + eps with per-dimension noise.
  PCA is the equal-noise special case.
- inertia: mean squared distance to the nearest center. The
  k-means score, train or held-out.
- elbow: the k where held-out inertia stops falling fast. A
  heuristic, not a proof.

## U07 terms

- decision region: input set that reaches one leaf. Always a
  rectangle in axis-aligned trees.
- impurity: the error rate of labeling by the node class
  distribution. Gini, entropy, misclassification.
- gain: parent impurity minus weighted child impurity. The
  split score.
- cost-complexity pruning: R_alpha(T) = R(T) + alpha |T|.
  Grow full, then collapse weak splits.
- bootstrap replicate: n draws with replacement from n
  points.
- out-of-bag: points a replicate missed. Honest error
  estimate.
- bagging: average of trees on bootstrap replicates. Cuts
  variance, not bias.
- random forest: bagging plus random feature subsets per
  split. Decorrelates trees.
- AdaBoost: reweight points toward misses. vote stumps by
  alpha = 0.5 ln((1-eps)/eps).
- gradient boosting: fit residuals (negative gradients),
  add with learning rate nu.
- residual: r_i = y_i - F(x_i). The next round's target.
- precision: TP/(TP+FP). recall: TP/(TP+FN). F1: their
  harmonic mean.
- balanced accuracy: (TPR + TNR)/2. Chance level is 0.5.
- permutation importance: score drop when one column is
  shuffled. Audits the tree, not the world.
- baseline: the dumb reference (majority, random,
  one-rule) every model must beat.

## U08 terms

- MLP: stacked affine maps with per-item nonlinearities.
- forward pass: x -> z -> a -> out with shapes checked per
  line.
- backprop: one backward chain-rule walk for all gradients.
- adjoint: the dL/d(variable) carried backward. do, dz, da.
- finite-difference check: central differences against the
  analytic gradient. The trust contract.
- convolution: sliding dot products. Valid mode gives
  n - k + 1 outputs.
- weight sharing: one kernel reused at all positions.
  Translation invariance as architecture.
- receptive field: the input span one output sees.
- RNN: h_t = f(W_h h_{t-1} + W_x x_t). Same weights each
  step.
- vanishing gradient: the w^T product dies below 1.
  Exploding gradient: it blows up above 1.
- LSTM gate: sigmoid switch on erase (forget), write
  (input), read (output).
- cell state: the additive memory c. Constant error
  carousel when f = 1, i = 0.
- attention: softmax(QK^T/sqrt(d)) V. Soft lookup over
  keys.
- transformer block: attention + feedforward, each with
  residual add and normalization.
- layer norm: mean/var over features, per item. batch
  norm: over the batch, per feature.
- Adam: per-coordinate adaptive steps from moment
  estimates m, v with bias correction.

## U10 terms

- ancestral sampling: draw z ~ p(z), then x ~ p(x|z).
- latent variable: unobserved z that generates x. Labels
  may be unidentifiable.
- ELBO: E_q[log p(x|z)] - KL(q||p). Lower bound on
  log p(x).
- KL gap: log p(x) - ELBO = KL(q||posterior) >= 0.
- reparameterization: z = mu + sigma eps. Gradients flow
  through mu, sigma.
- KL collapse: q retreats to the prior. the latent dies.
- GAN: D maximizes V, G minimizes V. Likelihood-free
  sampler training.
- mode collapse: G outputs one point. V stops moving.
- Jensen-Shannon divergence: symmetric, max log 2. The
  GAN optimum's divergence.
- empirical CDF: staircase from samples. True CDF: smooth.
- beta-VAE: recon + beta KL. beta -> infinity kills the
  latent.
- predicted vs measured: theory number vs code number,
  labeled at point of use.
- falsifiable hypothesis: a claim with a stated losing
  condition.
