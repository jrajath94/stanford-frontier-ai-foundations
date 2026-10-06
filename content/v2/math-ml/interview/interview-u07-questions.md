# Interview bank, U07 trees and ensembles

Date: 2026-10-06. Questions only. Keys in interview/keys-u07.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Define a decision region in one sentence and state the
shape every region of an axis-aligned tree must have.
B2. Write the Gini, entropy, and misclassification formulas
for class probabilities p_k. Which one would you never use
to grow a tree, and why?
B3. A node holds counts [4, 1]. Give all three impurity
scores from the lesson.
B4. State the variance formula for the average of B
estimators with pairwise correlation rho. What happens at
rho = 1?
B5. AdaBoost: write the alpha formula and the weight update.
What does alpha = 0 mean?
B6. Define precision, recall, and F1. A model has accuracy
0.95 on 95/5 data with recall 0. What is its F1?

## Deep ladder D1, splits and growth (5 follow-ups)

D1.1. Define impurity gain in one sentence.
D1.2. Toy: the lesson's C03 numbers. Give the best
threshold, its gain, and the weighted child Gini.
D1.3. Derive or justify: why does the scan cost O(n log n)
per feature?
D1.4. Implement/debug: see T1 below.
D1.5. Changed constraint: the features are categorical with
1000 levels. What breaks in the midpoint scan, and what is
the standard fix?

## Deep ladder D2, ensembles (5 follow-ups)

D2.1. Define bagging in one sentence and state what it
cannot fix.
D2.2. Toy: the C06 numbers. Give the ensemble variance at
rho = 0.3, B = 10 and at rho = 0.6, B = 100.
D2.3. Derive or justify: the variance rearrangement from
Var(mean) to the rho form.
D2.4. Compare: bagging averages independent-ish fits.
boosting fits residuals sequentially. When does each win?
D2.5. Research critique: "With B = 10,000 trees, the
ensemble variance goes to zero." Attack with the formula
and one measured number.

## Analytical/quantitative (2)

Q1. Node counts [6, 2]. Without a computer: compute Gini,
entropy (nats, use ln 3 = 1.0986, ln 2 = 0.6931), and
misclassification. A split gives children [6, 0] and
[0, 2]. Compute the Gini gain.
Q2. AdaBoost round: eps = 0.2, four weights equal.
Without a computer: compute alpha, the unnormalized new
weights for a correct and a missed point, and the
normalized vector as exact fractions.

## Implementation/debug (1)

T1. A teammate's tree code computes
`score = (p ** 2).sum()` at each candidate threshold
and picks the threshold with the minimum score. On the
lesson's C03 toy their code picks t = 0.2. (a) Reproduce:
compute the weighted sum-of-squares at each candidate
(0.2, 0.35, 0.5, 0.7, 0.85) and confirm the minimum is
0.6 at t = 0.2/0.85. (b) Diagnose the bug in one
sentence. (c) State the one-line fix and the correct
pick.

## Scenarios (2)

S1. Fraud detection, 1 positive in 10^4 transactions.
Your gradient-boosted tree reports accuracy 0.9999 and
the team wants to ship. Walk through your evaluation:
which metrics, which baseline, which threshold decision,
and what you tell the team. Then name the failure mode
if the positives drift next quarter.
S2. A random forest's OOB error is 0.05 but the held-out
test error is 0.25. List three mechanisms that explain
the gap, ordered by how you would check them, and say
what each check costs.

## Research critique (1)

R1. "Feature importance from our forest proves that
sensor X2 is irrelevant, so we removed it." Attack the
claim with two numbers from the lesson's C11 toy, then
state what evidence would actually justify removing a
sensor.
