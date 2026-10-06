# Lab 03, calculus and optimization computations

Unit: math-ml-U03. Date: 2026-10-06. numpy 1.26.4, float64.
Keys in labs/keys-lab-03.md. Test-mode: solve closed-book, then check.

## Task 1, least-squares gradient by hand then in code

Points (1, 2), (2, 3), (3, 5), (4, 4). Fit y = w x.
(a) By hand: compute w* = sum x_i y_i / sum x_i^2.
(b) In code: compute the gradient at w = 1 and at w = w*. Verify
the gradient at w* is within 1e-12 of zero.
(c) Take one GD step from w = 1 with eta = 0.1. State the new w and
whether the cost fell.

## Task 2, learning-rate regimes

J(w) = (w - 3)^2, w0 = 0, 12 steps.
(a) Run eta = 0.05, 0.5, 1.5. Record the final J for each.
(b) Compute the multiplier |1 - 2 eta| for each eta and predict the
regime before looking at the numbers.
(c) Write one sentence: what would you tell a teammate whose
training loss exploded at step 500 after a calm start?

## Task 3, Newton versus gradient descent

h(x) = x^4 - 3x^2, x0 = 1.0.
(a) Run 5 Newton steps. Record the iterates and the final value.
(b) Verify the final value equals sqrt(1.5) within 1e-9.
(c) Break it: start Newton at x0 = 0.1. Record what happens and
explain using the Hessian sign at the start point.
