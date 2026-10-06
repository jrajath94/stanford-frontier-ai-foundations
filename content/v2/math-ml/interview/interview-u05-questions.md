# Interview bank, U05 risk, regularization, and generalization

Date: 2026-10-06. Questions only. Keys in interview/keys-u05.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Write the population risk and the empirical risk for 0-1 loss,
and say which one training minimizes.
B2. Three losses on y = [0, 1, 1]: name the constant each one buys
(squared, absolute, 0-1 majority rule).
B3. State the bias-variance decomposition of MSE and the assumption
it needs.
B4. Ridge with n lambda = 2 moves w from 0.9286 to 0.8125 on the
lesson toy. Explain the move in one sentence.
B5. Why does in-sample MSE always prefer lambda = 0, and what do
you use instead to pick lambda?
B6. Name three leak channels subtler than copying the label into
the features.

## Deep ladder D1, capacity (5 follow-ups)

D1.1. Define model capacity in one sentence.
D1.2. Toy: read figure f02. Give the train and test MSE at degrees
1, 2, 3, and name the best degree.
D1.3. Derive or justify: why does train error fall monotonically
as degree rises?
D1.4. Implement/debug: a teammate reports test error falling at
every degree tried, up to degree 9. What do you tell them to try
next, and why?
D1.5. Changed constraint: n grows from 4 to 400 with the same
noise. Predict how the U bottom moves and why.

## Deep ladder D2, cross-validation (5 follow-ups)

D2.1. Define k-fold CV in one sentence.
D2.2. Toy: the lesson's LOOCV table. Which lambda wins, by what
rule, and what are the two CV means?
D2.3. Derive or justify: why does the fold penalty need rescaling
with the fold size?
D2.4. Implement/debug: CV picks lambda = 0.5, but the deployed
model overfits. Name two distinct causes.
D2.5. Research critique: "Nested CV is wasteful. One CV loop is
enough for both selection and reporting." Attack the claim.

## Analytical/quantitative (2)

Q1. Points x = [0, 1, 2, 3], y = [0, 1, 3, 2], model y = w x.
Without a computer: compute Sxx, Sxy, the OLS w, and the ridge w
for n lambda = 2. Then compute the penalized objective at both w
values and state which wins.
Q2. A coin has p = 0.7. From n = 4 flips you use the Laplace
estimator (k+1)/(n+2). Compute its bias, variance, and MSE, and
compare with the MLE's MSE. Show the split adds up.

## Implementation/debug (1)

T1. A teammate's pipeline does this before the train/test split:

```python
rng = np.random.default_rng(7)
X = rng.normal(0, 1, (40, 2))
y = (X[:, 0] > 0).astype(int)
rng.shuffle(X)          # shuffle rows of X
# ... then split X[:30] / X[30:] with y[:30] / y[30:]
```

They report test accuracy 0.2 on a problem that should be
trivially separable, and they blame the model. The premise was
executed 2026-10-06: with the shuffle, test accuracy is 0.2.
without it, 1.0. The shuffled rows agree with the true labels only
35% of the time. Diagnose the bug, explain the three numbers, and
give the fix.

## Scenarios (2)

S1. You inherit a churn model with test AUC 0.97. The top feature
is "days since last login", computed from a table that includes
the prediction date. Production AUC is 0.61. Walk through your
diagnosis: what you check first, what confirms the leak, and what
you change in the pipeline.
S2. Your validation curve plateaus at 0.27 while the noise floor
is ~0.25, and the train-val gap closed at n = 2000. The team wants
10x more labels. Argue from the curve: what do you recommend
instead, and what single experiment settles it?

## Research critique (1)

R1. "Double descent refutes the bias-variance tradeoff, so
capacity control is obsolete." Take a position. Use the lesson's
f01/f02 evidence, state exactly what double descent challenges
and what it leaves intact, and name the experiment that would
change your mind.
