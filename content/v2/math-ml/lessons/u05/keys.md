# Keys, lesson 05 risk, regularization, and generalization

Date: 2026-10-06. Test-mode: solve closed-book first. All numbers
computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## E01

R(h) = E[l(y, h(x))]: the expected loss over the true distribution.
R_hat(h) = (1/n) sum_i l(y_i, h(x_i)): the average loss on the
sample. Words: population risk is the forever scoreboard. Empirical
risk is the scoreboard from the data at hand.

## E02

Rule "predict 1 when x >= 1": predictions [0, 1, 1, 1]. Truth
[0, 0, 1, 1]. Only x = 1 disagrees. R_hat = 1/4 = 0.25.

## E03

The memorizer is right on the 4 training points and guesses at
random elsewhere. On x uniform [0, 4] the training points have
measure 0, so the population risk is 0.5 (random guess on a
balanced problem). The bound is honest because it uses no
information the memorizer does not have: outside its 4 points it
knows nothing.

## E04

J(c) = (1/3)[c^2 + (1-c)^2 + (1-c)^2]. dJ/dc = (1/3)[2c - 2(1-c) -
2(1-c)] = 0 gives 6c - 4 = 0, c = 2/3. J(2/3) = (1/3)(4/9 + 1/9 +
1/9) = 6/27 = 2/9 = 0.2222.

## E05

Squared: the mean. Absolute: the median. 0-1 with majority rule:
the most frequent class.

## E06

Mismatch: squared loss optimizes a regression target while the
deployment grades class verdicts. A prediction of 0.49 for label 1
is "close" in squared loss but wrong in accuracy. Fix: train a
proper classification loss (logistic, hinge) and report accuracy on
held-out data.

## E07

(1) IID draws from a fixed distribution. (2) A fixed estimator, a
fixed function of the data. (3) n -> infinity with everything else
frozen.

## E08

MSE = p(1-p) = 0.21. Ratio to IID (0.0525): 4.0. The four copies
carry one flip of information.

## E09

MLE: bias 0, var 0.0525, MSE 0.0525. Laplace: bias -0.0667, var
0.02333, MSE 0.02778. Check: (-0.0667)^2 + 0.02333 = 0.00444 +
0.02333 = 0.02778. The split adds up.

## E10

Degree 2. Bias squared ~0.0005, variance ~0.0256, total ~0.0261
(the script prints 0.0219 at the grid used. The minimum sits at
degree 2 either way).

## E11

0-1 loss. The split needs squared error: E[(t - theta)^2] expands
into the three terms. For 0-1 loss there is no such expansion, and
"bias" of a classifier needs a separate definition.

## E12

Sxx = 0 + 1 + 4 + 9 = 14. Sxy = 0 + 1 + 6 + 6 = 13. w = 13/14 =
0.928571 (n lambda = 0). w = 13/16 = 0.8125 (n lambda = 2).

## E13

Objective at OLS w: 3.653061. At ridge w: 3.4375. 3.4375 < 3.653061,
so the ridge w wins under the penalized objective. The penalty moved
the optimum and the numbers confirm it.

## E14

The penalty sums w_1^2 + w_2^2 in raw units. A weight of 1000 on
the millimeter feature is "small" in real terms but huge to the
penalty. One lambda then over-taxes the millimeter feature. Fix:
standardize features first, so the penalty taxes comparable scales.

## E15

log p(w|data) = -RSS/(2 sigma^2) - w^2/(2 tau^2) + const. Multiply
by -2 sigma^2: RSS + (sigma^2/tau^2) w^2 + const. Minimizing this
is ridge with n lambda = sigma^2/tau^2. Derivative:
2 Sxx w - 2 Sxy + 2 (sigma^2/tau^2) w = 0 gives
w = Sxy/(Sxx + sigma^2/tau^2).

## E16

sigma^2/tau^2 = 1/0.5 = 2. w_MAP = 13/16 = 0.8125. Ridge with
n lambda = 2: w = 13/16 = 0.8125. Equal to all printed digits.

## E17

Train: w = 13/14. Residuals: 0, 0.0714, 1.1429, -0.7857. Squares:
0, 0.0051, 1.3061, 0.6173. Mean: 0.4821. Test: residuals 1.0857,
-0.7429. Squares 1.1788, 0.5518. Mean: 0.8653.

## E18

No. The test set picked the winner, so the winning score is the
maximum over candidates, which is optimistic. It needs its own
held-out check: that is nested validation.

## E19

Hold out x = 2, y = 3. Fit on [0, 1, 3], [0, 1, 2]: Sxx = 0 + 1 +
9 = 10, Sxy = 0 + 1 + 6 = 7, w = 0.7. Prediction at x = 2: 1.4.
Squared error: (3 - 1.4)^2 = 2.56. Matches the table.

## E20

Pick the lambda with the smallest LOOCV mean: 1.1806 (n lambda = 2)
beats 1.8515 (n lambda = 0). The rule is "minimize the honest
score", not "minimize the train score".

## E21

Selection bias: the best of 100 CV means is optimistic, because you
maximized over noise. Fix: nested CV, or a final held-out set that
played no role in the search.

## E22

Degree 3 with intercept has 4 parameters. There are 4 points. A
degree-3 polynomial through 4 points in general position is exact,
so the residual is ~0 (3.07e-30, float noise).

## E23

2.8757, 0.5313, 28.5613. Best: degree 2.

## E24

Not necessarily. With n = 400 the U bottom moves right: 4
parameters cannot memorize 400 noisy points, so degree 3 would sit
near the bottom instead of off the cliff. Capacity is relative to
n.

## E25

n = 3: 0.3343 - 0.0259 = 0.3084. n = 40: 0.2714 - 0.1611 = 0.1104.

## E26

No gap with high validation error means the model family cannot
express the truth: bias, not variance. The fix is capacity or
features, not data.

## E27

(1) Imputation statistics computed on all the data. (2) Feature
selection run before the split. (3) Early stopping on the test
set. (4) Group overlap: same patient, user, or time window in both
parts.

## E28

The grand mean carries test information into every training point.
Leak channel: target-adjacent statistic from the future. Fix:
compute the mean on the training part only, then apply it to both.

## E29

Residuals: -0.5286, -0.9857, -0.6429. Squares: 0.2794, 0.9716,
0.4133. Mean: 0.5548.

## E30

Covariate shift (x moves), label shift (y moves), concept shift
(the relation moves). Reweighting addresses covariate shift when
the support overlaps. It cannot fix concept shift.

## Ladders

L01. (1) R is the expected loss over the truth. R_hat is the sample
average. (2) 0.25, one error in four. (3) If the loss is chosen
after seeing data, ERM can pick a loss that flatters the sample.
the guarantee needs the loss fixed first. (4) The 4-line snippet in
C01. (5) The last 7 days are not IID: weekly cycles, trends, and
feedback break the fixed-distribution clause. Use time-aware
splits.
L02. (1) A loss maps (truth, prediction) to a price. (2) The C02
table: squared buys the mean 2/3 at 0.2222. (3) d/dc E[(Y-c)^2] =
-2E[Y-c] = 0 gives c = E[Y]. (4) Check what the 0.95 means: with
squared loss on 0/1 labels, a constant 0.95 predictor can score
well without classifying anything. Recompute accuracy from hard
predictions. (5) False alarms have a price: use a cost-weighted
loss or tune the decision threshold on the true cost, not accuracy.
L03. (1) IID, fixed estimator, n -> infinity. (2) 0.0525 versus
0.21, ratio 4. (3) Dependence keeps the variance at O(1) instead of
O(1/n): the effective sample size is the number of independent
pieces. (4) Compare retrained models on logged data versus live
A/B: if live behavior diverges from the log replay, a feedback
loop moves the distribution. (5) False in general: more data
from a shifted or dependent stream need not help. More IID data
from the fixed truth helps.
L04. (1) MSE = bias^2 + variance + noise. (2) Laplace: -0.0667,
0.02333, 0.02778 beats MLE 0.0525. (3) Expand E[(t - theta)^2]
around E[t]: cross term vanishes. (4) Bias falls with degree,
variance rises, total is U-shaped with minimum at degree 2.
(5) k = 1: near-zero bias, high variance. k = 50: higher bias,
low variance. Same tradeoff, different knob.
L05. (1) J(w) = RSS + n lambda w^2. (2) 0.9286 versus 0.8125.
(3) dJ/dw = -2 Sxy + 2(Sxx + n lambda) w = 0. (4) The penalty may
sit on the wrong scale (unstandardized features), or the overfit
may be in the family, not the weights: no penalty adds a missing
feature. (5) Standardize, then a lambda path with CV: with 200
samples the penalty is the main defense. Start lambda large, relax
by CV.
L06. (1) Maximize log likelihood + log prior. (2) 0.8125 both
ways. (3) The log prior is -w^2/(2 tau^2): a quadratic penalty.
(4) The prior is a modeling choice with consequences: it states a
belief about the world, and a wrong tight prior bends answers.
That is more than a hack: it is an assumption you can argue with.
(5) Translate the knowledge into a scale: tau is the plausible
size of w. Set tau^2 from domain knowledge, then check sensitivity
by refitting at tau/3 and 3 tau.
L07. (1) Split before fitting, lock the model, score once.
(2) 0.4821 train, 0.8653 test. (3) Scoring an unlocked model lets
fitting continue on the test set: the number is gamed. (4) Innocent:
the test draw was easy, or train had outliers. Guilty: the test
set leaked into training. (5) Split by time: train on the past,
validate on the future. Never shuffle time series.
L08. (1) Split into k parts, rotate the held-out part, average.
(2) The fold table: CV picks n lambda = 2. (3) The penalty term
scales with the fold size. Without rescaling, lambda would mean
different strengths on different folds. (4) The CV score was gamed
by the search (selection bias), or the deployment distribution
shifted after the CV. (5) Use few folds (3-5), subsample for the
search, and reserve one final time-based holdout. Budget the fits.
L09. (1) The range of functions the class can express. (2) f02:
train falls, test is U-shaped, best at degree 2. (3) A richer
class contains the poorer one, so the training minimum can only
improve. (4) The U bottom lies beyond the tried degrees: try larger
degrees or richer families until the test curve turns up.
(5) Double descent shows test error falling again past the
interpolation point in some regimes: the classical U is not the
whole story, and the second descent needs its own theory.
L10. (1) 0.3084 at n = 3, 0.1104 at n = 40. (2) Train 0.0986,
shift test 0.5548. (3) With few points the model fits noise
cheaply. With many points the noise cannot all be fit, so the
average residual rises toward the noise floor. (4) The family is
wrong (bias plateau), or the validation set itself shifted.
(5) Confirm the shift on fresh data, reweight if the support
overlaps, otherwise recollect from the new distribution and
retrain. Then monitor the inputs, not just the loss.
