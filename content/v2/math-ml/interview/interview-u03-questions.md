# Interview bank, U03 calculus and optimization

Date: 2026-10-06. Questions only. Keys in interview/keys-u03.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Define a partial derivative in one sentence, then give df/dx of
x^2 + 3xy + y^2 at (2, 1).
B2. State the chain rule for y = (2x+1)^3 and name the two link
slopes at x = 2.
B3. What are the shapes of the gradient, the Jacobian, and the
Hessian for f: R^3 -> R?
B4. What does a positive-definite Hessian everywhere tell you about
a function, and what is the name of that property?
B5. Write the gradient-descent update and the one number that
controls its regime.
B6. State all four KKT conditions for min x^2 subject to x >= 1.

## Deep ladder D1, learning rates (5 follow-ups)

D1.1. Define the stability boundary for gradient descent on a
quadratic bowl.
D1.2. Toy: J(w) = (w-3)^2, w0 = 0, 12 steps. State the final J for
eta = 0.05, 0.5, 1.5 from the lesson.
D1.3. Derive or justify: why does each step multiply the distance to
the bottom by |1 - 2 eta|?
D1.4. Implement/debug: a teammate's loss explodes at step 500 after
a calm start. The eta never changed. Name the mechanism and the two
diagnostics you run first.
D1.5. Changed constraint: the bowl is now J(w) = 5(w-3)^2. State the
new boundary and predict what eta = 0.5 does.

## Deep ladder D2, Newton (5 follow-ups)

D2.1. Define the Newton step in one sentence.
D2.2. Toy: one Newton step on (w-3)^2 from w = 0. Do it by hand.
D2.3. Derive or justify: why do the correct digits roughly double
per step near the answer?
D2.4. Implement/debug: Newton on h(x) = x^4 - 3x^2 from x0 = 0.1
converges to x = 0. The code is correct. Explain the outcome using
the Hessian sign.
D2.5. Research critique: "Newton is always better than gradient
descent because it uses more information." Attack the claim: name
two regimes where it loses and the replacement in each.

## Analytical/quantitative (2)

Q1. Least squares: points (1, 2), (2, 3), (3, 5), (4, 4), model
y = w x. Without a computer: find w*, the gradient at w = 1, and the
result of one GD step with eta = 0.1.
Q2. The Hessian of f(x, y) = x^2 + 3xy + y^2 is [[2, 3], [3, 2]]
with eigenvalues -1 and 5. A teammate calls the surface "a bowl
because the diagonal is positive." Diagnose the error and name the
true shape.

## Implementation/debug (1)

T1. This code intends a central-difference gradient check of
J(w) = sin(w) + w^2 at w = 1:

```python
import numpy as np
def J(w): return np.sin(w) + w**2
h = 1e-7
fd = (J(1+h) - J(1)) / h
print(abs((np.cos(1) + 2) - fd))
```

It prints 6.03703385060328e-08: the forward difference is first-order,
while the lesson's central difference reached 4.2e-10 absolute
(1.6e-10 relative). Name the bug, fix it, and state the expected
absolute error after the fix.

## Changed-constraint scenarios (2)

S1. Your parameters are 1e6 and the Hessian is dense. Newton is
impossible. Name the practical replacement for the step direction
and the assumption it needs.
S2. Your constraint set is x >= 1 AND x <= 2, and the objective is
x^2. Solve it, name both multipliers, and state which fence binds.

## Research critique (1)

R1. "We replaced SGD with full-batch GD because GD's steps are
exact. Training got slower per epoch but each step is better, so
final quality must improve." Critique: is the reasoning sound? Name
the missing quantity and the experiment that decides.
