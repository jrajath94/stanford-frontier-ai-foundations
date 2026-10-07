# keys-01.md, diagnostic 01

Date: 2026-10-06. Closed-book answers. Keep separate.

## D01

Features: age, blood pressure, cholesterol. Target: heart attack in
10 years.

## D02

Overfitting: the model memorized the training set and does not
generalize.

## D03

Underfitting (high bias). Fix: use a model family that can bend,
for example add squared features.

## D04

To generalize is to score well on data the model has not seen.

## D05

Student B. Student A contaminated the test set by tuning on it, so
the score measures tuning effort, not future performance.

## D06

95 percent accuracy. It is a poor metric: a useless model scores
95 percent because the class balance is 95 to 5. Use precision and
recall, or a balanced metric.

## D07

`x` has shape (4,). The design matrix has shape (200, 4).

## D08

The 0/1 loss is flat almost everywhere, so its gradient is zero
wherever it exists. Gradient methods get no direction to move.
