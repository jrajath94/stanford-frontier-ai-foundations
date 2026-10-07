# notation_and_shapes.md, cs229

Date: 2026-10-06. Shared symbols used in every unit. Local lessons
may add symbols. They must define them before first use.

## Scalars, vectors, matrices

- `m`: number of training examples. Scalar, positive integer.
- `n`: number of features. Scalar, positive integer.
- `x`: input vector, shape `(n,)`.
- `x^{(i)}`: i-th training input. Superscript is an index, not a power.
- `y`: target. Scalar for regression, `0/1` or `1..K` for
  classification.
- `y^{(i)}`: i-th training target.
- `X`: design matrix, shape `(m, n)`, rows are examples.
- `theta` / `\theta`: parameter vector, shape `(n,)`.
- `theta^T x`: dot product, a scalar.

## Learning objects

- `h(x)` or `h_theta(x)`: hypothesis, maps input to prediction.
- `J(theta)`: cost (training objective), scalar.
- `L(y_hat, y)`: loss on one example, scalar.
- `R(h)`: risk (expected loss), scalar.
- `R_hat(h)`: empirical risk on the training set, scalar.

## Probability

- `p(x)`: density or mass at `x`. `p_theta(x)`: the parameterized form.
- `E[X]`: expectation. `Var(X)`: variance. `Cov(X)`: covariance
  matrix, shape `(n, n)`.
- `q(z|x)`: approximate posterior. `p(x|z)`: likelihood term.

## Deep learning

- `z`: pre-activation, shape `(units,)`.
- `a`: activation, shape `(units,)`.
- `W`: weight matrix, shape `(out, in)`. `b`: bias, shape `(out,)`.

## Shape convention

Vectors are column vectors. `X` has shape `(m, n)`: `m` rows
(examples), `n` columns (features). All lessons follow this.

## Units

No physical units. Features carry their own units (feet squared,
dollars, counts). Targets carry their own. Losses are unitless
scores. Gradients carry units of loss per parameter unit.
