# Interview bank, U06 linear models, kernels, and margins

Date: 2026-10-06. Questions only. Keys in interview/keys-u06.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Write the normal equations and name the assumption that makes
X^T X invertible.
B2. State in one sentence each what changes when Gaussian noise
becomes Bernoulli noise in a linear-score model.
B3. Softmax on logits [2, 1, 0] gives which probability to class
0, and what is the cross-entropy loss?
B4. A kernel k(x, z) = (1 + x z)^2: write the explicit feature map
with correct coefficients and verify the dot product at x = 2,
z = 3.
B5. Define the geometric margin and the hinge loss in one sentence
each. What does C price?
B6. The SVM dual has one linear constraint besides alpha >= 0.
State it and say what breaks if it is dropped.

## Deep ladder D1, least squares to logistic (5 follow-ups)

D1.1. Define the OLS estimator in one sentence.
D1.2. Toy: the lesson's C01 numbers. Give w_hat, RSS, and the
residual sum, and say which equation forces the sum to zero.
D1.3. Derive or justify: why does Gaussian noise imply the
squared loss?
D1.4. Implement/debug: a teammate fits logistic regression on
separable data and reports ||w|| = 10^6 after 2000 steps,
still falling loss. Diagnose.
D1.5. Changed constraint: Laplace noise replaces Gaussian.
Predict the new loss and the new "normal equations".

## Deep ladder D2, kernels and the dual (5 follow-ups)

D2.1. Define a kernel in one sentence and state the PSD
requirement.
D2.2. Toy: the lesson's C07 Gram. Give the three eigenvalues and
the PSD verdict.
D2.3. Derive or justify: from the hard-margin Lagrangian, show
w = sum alpha_i y_i x_i and state where the kernel enters.
D2.4. Implement/debug: see T1 below.
D2.5. Research critique: "With the kernel trick, model capacity
is free." Attack the claim with two measured numbers from the
lesson.

## Analytical/quantitative (2)

Q1. Points x = [0, 1, 2, 3], y = [0, 1, 3, 2], model y = b + w x.
Without a computer: compute X^T X, its determinant, the
inverse, w_hat, the four residuals, and RSS. Then compute
sigma2_hat and the predictive variance at x = 5.
Q2. Two points: x+ = (1,1), y = +1. X- = (-1,-1), y = -1.
Without a computer: write the dual objective as a function of
one variable a, solve for a*, recover w, and verify primal =
dual. Then add the constraint you almost forgot and say what
value the careless version gives.

## Implementation/debug (1)

T1. A teammate solves K alpha = b with an RBF Gram (gamma =
1e-6) on 3 points. The solver returns residual 4.12e-11 and
they declare victory. You run the same code with gamma = 0.5.
The premise was executed 2026-10-06:

- gamma = 1e-6: cond(K) = 3.227e6, alpha = [-750000.0,
  500001.0, 250001.5], residual 4.12e-11.
- gamma = 0.5: cond(K) = 4.202, alpha = [-0.732, 2.2044,
  2.9181], residual 0.0.
- gamma = 1e-6 with jitter 1e-6: ||alpha|| = 471405.1 versus
  935415.3 without jitter.

Diagnose: is the solver broken? What is actually wrong, what
does the residual hide, and what is the real fix (not the
jitter)?

## Scenarios (2)

S1. You inherit a churn classifier: logistic regression on 40
raw features, one of them "account age in seconds" (~1e9
scale). Training loss oscillates and the weights are enormous.
Walk through your diagnosis: what you measure first, what the
numbers mean, and the two-line fix.
S2. A kernel SVM gets train accuracy 1.0 and validation 0.55 on
a 2000-point task. Your teammate wants a bigger C. Argue from
the Gram: what do you check first, what would confirm
memorization, and what do you change instead?

## Research critique (1)

R1. "The hinge loss is obsolete: logistic loss is smooth,
convex, and gives probabilities, so it dominates the hinge
everywhere." Take a position. Use the lesson's C12 numbers
(0.7167 vs 0.7833 under 15 percent label noise), state exactly
what the smoothness buys and what it costs, and name the
experiment that would change your mind.
