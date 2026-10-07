# Lab 03: bias-variance, ridge, and model selection

Date: 2026-10-06. Units: cs229-U08, cs229-U09.
Work the tasks, then check keys-lab-03.md. Run all code.

Setup: numpy only. Seeds as stated per task. No other installs.

## Task 1: the bias-variance decomposition, measured

Data: truth y = x^2 on [0, 1], noise sd 0.3, n = 15.
For each degree in [1, 2, 3, 4, 5]: draw 40 datasets
(seed 1000 + t for trial t), fit np.polyfit on each,
predict on 60 fixed test points. Report per degree:
bias^2, variance, and total test MSE. State which
degree wins and which term dominates at degree 5.

## Task 2: the ridge path

Data: y = sin(2 pi x) + noise (sd 0.3), x in [0, 1],
12 points, seed 42. Fit degree-8 polynomial with ridge
lambda in [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0].
Report test MSE on 300 fresh points for each lambda.
State the shape of the curve and the best lambda.

## Task 3: training error lies, validation decides

Data: y = sin(2 pi x) + noise (sd 0.3), 30 points,
seed 42. Split: first 20 train, last 10 validation.
For degrees 0..7: fit on train, report train MSE and
validation MSE. State the selected degree and explain why
training error alone picks the wrong one.

## Deliverable

A short log: the three result blocks with numbers. No
essay. The numbers must match keys-lab-03.md within the
stated tolerance.
