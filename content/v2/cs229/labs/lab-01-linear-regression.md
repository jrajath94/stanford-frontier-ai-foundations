# Lab 01: linear regression mechanics

Date: 2026-10-06. Unit: cs229-U02.
Work the tasks, then check keys-lab-01.md. Run all code.

Setup: numpy only. Seed 42 everywhere. No other installs.

## Task 1: batch gradient descent

Data: X has x0 = 1 folded in, 30 examples, x in [0, 5],
y = 2 + 1.5 x + noise (sd 0.4). Implement batch gradient
descent with alpha = 0.1 for 2000 steps. Record the final
theta and the final cost. It must match lstsq within 1e-3.

## Task 2: closed form and the orthogonality check

Solve the same data with lstsq. Compare with the Task 1
theta. Compute norm(X.T @ (y - X @ theta)). Report the
value. State what it proves.

## Task 3: rank deficiency drill

Duplicate the x column. Try the explicit inverse
inv(X.T @ X) @ X.T @ y. Record what happens. Then solve
with pinv. Report both thetas and the singular values of X.

## Task 4: bandwidth experiment

Data generation (pin this exactly. The keys below reproduce
only from this procedure):
- x_train = np.linspace(0, 6, 30) (even grid, not uniform).
- rng = np.random.default_rng(42).
- y_train = sin(x_train) + rng.normal(0, 0.5, 30) (the 30
  noise draws come first from the stream).
- xt = rng.uniform(0, 6, 200) (the 200 test points come
  next from the same stream).
- Test targets are NOISELESS: yt = sin(xt).
Implement LWLR exactly as the U02 lesson's lwlr_predict:
linear fit with intercept, weights
w = exp(-(x - xq)^2 / (2 tau^2)). For tau in
[0.2, 0.5, 1.0, 2.0, 5.0], compute the test MSE on the
200 points. Plot test MSE vs tau. State the shape and
the best tau.

## Deliverable

A short log: the four result blocks with numbers. No
essay. The numbers must match keys-lab-01.md within the
stated tolerance.
