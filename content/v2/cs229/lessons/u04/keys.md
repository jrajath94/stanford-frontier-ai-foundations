# keys.md, U04 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate.

## E01

p(x, mu, Sigma) = 1/((2 pi)^{d/2} |Sigma|^{1/2})
exp(-(1/2)(x - mu)^T Sigma^{-1} (x - mu)).

## E02

phi = mean of y. mu_c = mean of x over class c. Sigma =
(1/n) sum_i (x^{(i)} - mu_{y^{(i)}})(x^{(i)} -
mu_{y^{(i)}})^T.

## E03

The log odds is log p(x|y=1) - log p(x|y=0) + log
odds(phi). With shared Sigma the -(1/2) x^T Sigma^{-1} x
terms cancel, leaving terms linear in x.

## E04

p(x_1, ..., x_d | y) = prod_j p(x_j | y).

## E05

phi_{j|y} = (1 + count_{j,y}) / (2 + n_y).

## E06

Bernoulli: the email is a per-word presence checklist.
Multinomial: the email is a sequence of word draws.

## E07

phi = 0.5. mu0 = 1, mu1 = 6. Centered values: class 0: -1, 1.
class 1: -1, 1. Pooled Sigma = (1/4)(1 + 1 + 1 + 1) = 1.

## E08

log p(y=1|x)/p(y=0|x) = -(1/2)(x-mu1)^T S^{-1}(x-mu1) +
(1/2)(x-mu0)^T S^{-1}(x-mu0) + log(phi/(1-phi)). Expand:
the x^T S^{-1} x terms cancel. What remains:
x^T S^{-1}(mu1 - mu0) + const. Linear in x.

## E09

Diagnosis: miscalibration from double-counted evidence
(naive Bayes overconfidence). The accuracy is fine. The
probabilities are not. Fix: calibrate post-hoc (Platt
scaling or isotonic regression on validation), or switch
to a discriminative model.

## E10

At n = 1e6, team B (logistic) wins: with Poisson
features the Gaussian assumption is wrong, and in the
large-data limit the weaker assumptions win (notes page
42). At n = 100, team A may win on variance: GDA has
fewer effective parameters to estimate. The honest
answer: check both with validation.

## E11

Setup: two Gaussians with mean separation s in
{0.5, 1, 2, 4}. For n in {50, 100, 500, 2000, 10000},
fit GDA and logistic, record test error. Claim: the
crossover n where logistic beats GDA grows as s
shrinks. Falsified if logistic never wins or wins at
the same n for all s.

## E12

```python
import numpy as np
docs = [["buy", "now"], ["buy", "buy"], ["hello"]]
labels = [1, 1, 0]
vocab = ["buy", "now", "hello"]
V = len(vocab)
counts = {0: np.ones(V), 1: np.ones(V)}
for d, y in zip(docs, labels):
    for w in d:
        counts[y][vocab.index(w)] += 1
probs = {c: counts[c] / counts[c].sum() for c in (0, 1)}
for c in (0, 1):
    assert abs(probs[c].sum() - 1.0) < 1e-12
```

## L01 key

(1) y ~ Bernoulli(phi), x|y=c ~ N(mu_c, Sigma).
(2) See SL-02 computed example.
(3) Separate the log likelihood. Derivatives give class
means.
(4) Fit on the toy. Verify the boundary is the line
y = 1 and posteriors sum to 1.
(5) Gaussian data: GDA wins small-n, logistic catches up.
Poisson data: logistic wins at all n.
(6) Cause: fewer independent points than dimensions, or
collinear features. Fix: ridge on Sigma, or drop
features.
(7) Real classes rarely share covariance. Check with a
covariance comparison, else use QDA or logistic.
(8) n = 200: run a Gaussianity check (e.g., compare
class covariances). Pick GDA if it passes. Otherwise use logistic
otherwise. Validate both.

## L02 key

(1) p(x|y) = prod_j p(x_j|y).
(2) See SL-04 computed example.
(3) The log likelihood separates per (j, y). The derivative
gives the fraction.
(4) Smoothed code above. Assert no zero entries.
(5) Short documents: Bernoulli competitive. Long:
multinomial wins. Bernoulli drowns in absence terms.
(6) Cause: an unseen feature zeroes a product. Fix:
Laplace smoothing.
(7) Correlated features get counted twice. Probabilities
overstate confidence. Calibrate or decorrelate.
(8) Pipeline: tokenize, Laplace-smoothed multinomial NB,
validation-tuned threshold, reliability diagram on
held-out data.
