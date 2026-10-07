# keys.md, U01 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Features: age, systolic, cholesterol. Target: heart attack in 10
years. Hypothesis: rule from features to prediction. Parameters:
the dials inside the rule.

## E02

X shape (50, 7). y shape (50,). theta shape (7,) plus intercept,
or (8,) with x_0 folded in.

## E03

Empirical risk is the average loss of the hypothesis on the
training set.

## E04

Using the post-surgery report as a feature when predicting a
heart attack that leads to surgery.

## E05

The hypothesis was chosen to minimize that same score, so the
score rewards the choice. Fresh data does not.

## E06

Squared loss: (y_hat - y)^2. Log loss for true label 1 and
predicted probability p: -log(p).

## E07

Empirical risk: 30/200 = 0.15. Test risk estimate: 46/200 =
0.23. Gap: 0.08.

## E13

Train RMSE: sqrt(0.09) = 0.30. Test RMSE: sqrt(0.25) = 0.50.
MSE gap: 0.25 - 0.09 = 0.16. The gap alone does not prove
overfitting: it can also come from distribution shift
between the splits or from noise in the test estimate. The
train/test protocol (E12) must rule those out first.

## E08

Diagnosis: memorization (overfitting). Fixes: (1) shrink the
hypothesis class or add a penalty (U09). (2) Get more training
data or add a validation gate with early stopping.

## E09

Team B trains: log loss is smooth, gradient descent moves. Team
A cannot train: 0/1 loss has zero gradient almost everywhere.
Team B also reports the business metric: 0/1 accuracy on held
out data. Training loss and reporting metric serve different
roles.

## E10

Setup: generate data y = f(x) + noise with known noise rate r.
Train a memorizer (1-NN or a huge table) for m in {50, 200,
1000, 5000}. Measure empirical risk (0 by construction) and
test risk on 10,000 fresh points. Claim: test risk stays near r
for all m, so the gap stays near r. Falsified if the gap shrinks
with m.

## E11

```python
import numpy as np
X = np.random.default_rng(0).normal(size=(1000, 20))
y = np.random.default_rng(1).normal(size=(1000,))
theta = np.zeros(20)
pred = X @ theta
assert X.shape == (1000, 20)
assert y.shape == (1000,)
assert pred.shape == (1000,)
assert (X.T @ X).shape == (20, 20)
assert (X.T @ y).shape == (20,)
```

## E12

The IID assumption breaks: the deployment distribution differs
from the training distribution (covariate shift). Evaluation
must use data from the new city, not the old test set.

## L01 key

(1) Population risk is the expected loss over unseen data.
(2) Coin toy: always-heads scores 0.7 on 10 flips with 7 heads,
truth is 0.5.
(3) Empirical risk is the sample mean of i.i.d. draws of the
loss, whose expectation is the population risk.
(4) See E11 for the audit pattern.
(5) Cross-validation averages several held-out estimates, so it
has lower variance than one split. It costs k fits.
(6) Causes: test set leaked into training, or the test set is
too small and luck favored it.
(7) Hospital data fails IID across time (protocols change),
across sites (populations differ), and within patients
(repeat visits correlate).
(8) Gate: compute gap = test metric - train metric on a locked
test set. Block release if the gap exceeds the pre-registered
threshold. Log the decision.
