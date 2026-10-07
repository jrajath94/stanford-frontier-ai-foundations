# Keys: Lesson 09, Regularization and model selection

## Breadth recall

E01: J_lambda(theta) = J(theta) + lambda R(theta),
lambda >= 0.

E02: theta_MAP = argmax_theta prod_i
p(y^{(i)}|x^{(i)}, theta) p(theta).

E03: A Gaussian prior theta ~ N(0, tau^2 I) gives
log p(theta) = -||theta||^2/(2 tau^2), which is the
ridge penalty with lambda = 1/tau^2.

E04: The l1 ball has corners on the coordinate
axes. Loss contours meet it at corners, which are
sparse. The l2 ball is smooth, so contact is
almost never exactly sparse.

E05: Split data into k folds. For each fold,
train on k-1 folds and validate on the held-out
fold. Average the k validation scores.

E06: Validation leakage is any use of
validation/test information during training or
model selection: preprocessing fit on all data,
feature selection on all data, tuning on the test
set.

E07: Stop training when validation error starts
rising. The step count acts as the complexity knob:
few steps keep weights near initialization (small
norm), many steps fit noise. It is implicit l2
regularization.

E08: Learning curves plot train and validation
error vs n. Both high and close: bias (more data
will not help). A gap between them: variance (more
data helps).

## Deep oral ladders

L01: (1) J + lambda * (1/2)||theta||^2. (2)
Degree-8, n = 12: unregularized test MSE ~2.4,
ridge at lambda = 1 gives ~0.3. (3) As E08.
(4) Closed form or GD. Sweep lambda on a log
grid. (5) Ridge shares weight across correlated
features. LASSO picks one. (6) lambda must be
positive. X^T X + lambda I is PD for lambda > 0,
so a singularity complaint means lambda <= 0 or
a bug. (7) Ridge shrinks but never zeroes. It
does not do feature selection. (8) Grid of
lambdas, 5-fold CV inside training data, refit
winner on all training data, single test report.

L02: (1) Training error falls with complexity, so
it always picks the most complex model. (2)
Degrees 0-10: training error minimal at 10,
validation minimal at 2. (3) Split, train on
train, score on validation, pick the minimum.
(4) Standard k-fold loop with preprocessing
inside folds. (5) Holdout: cheap, noisy. k-fold:
expensive, stable. AIC: analytic, assumes the
model class. (6) Leakage (preprocessing or
tuning saw the test data) or selection bias (the
reported number is the best of many). (7) k-fold
costs k full trainings. Use holdout for large
models. (8) Train/validation/test: select on
validation, report once on test.

## Analytical exercises

E09: theta <- theta - eta grad J_lambda =
theta - eta (grad J + lambda theta) = (1 - eta
lambda) theta - eta grad J.

E10: log posterior = sum_i log p(y^{(i)}|x^{(i)},
theta) - ||theta||^2/(2 tau^2) + const.
Maximizing equals minimizing negative log
likelihood + (1/(2 tau^2))||theta||^2, the ridge
objective with lambda = 1/tau^2.

## Failure diagnosis

E11: Leakage. Standardization used the full
dataset, so validation folds saw test statistics
through the mean and variance. Validation error
is optimistic. Fix: fit the scaler inside each
fold on the training folds only.

## Counterfactual comparison

E12: Team B. Team A's number is contaminated: the
test set guided the choice, so it is a second
validation set and the reported error is
optimistic by an unknown amount. Team B's nested
CV keeps an untouched test set.

## Research question

E13: Falsifiable claim: on overparameterized
least squares from small initialization, GD lands
closer (in l2 distance) to the minimum-norm
interpolator than Adam with default settings.

## Implementation task

E14: Verified by the U-shaped test-error curve in
log lambda with the minimum at an interior point.
