# Keys, lesson 07 trees and ensembles

Date: 2026-10-06. All numbers computed 2026-10-06 unless marked
as hand arithmetic (verifiable by hand).

## E01

Rectangles: x1 <= 0.5 votes 0. x1 > 0.5 and x2 <= 0.5 votes 1.
x1 > 0.5 and x2 > 0.5 votes 1.

## E02

(0.9, 0.1): x1 > 0.5, x2 <= 0.5 -> 1. (0.1, 0.9): x1 <= 0.5
-> 0.

## E03

p = [0.9, 0.1]. Gini = 1 - (0.81 + 0.01) = 0.18. Entropy =
-(0.9 ln 0.9 + 0.1 ln 0.1) = 0.3251. Misclassification = 0.1.

## E04

Gini = 0.5, entropy = ln 2 = 0.6931, misclassification = 0.5.

## E05

t = 0.7: left [0,0,0,1] counts [3,1], Gini = 1 - (9/16 +
1/16) = 0.375. right [1,1] pure, Gini 0. Weighted: (4/6)
(0.375) = 0.25. Confirmed.

## E06

Parent Gini 0.5, weighted child Gini 0.0, gain 0.5 = parent
Gini. Both children pure: the split explains all the
impurity.

## E07

x1 <= 0.5: left labels [0,1], right [1,0]. both Gini 0.5.
weighted 0.5. gain 0. Same for entropy (both 0.6931, gain
0). Yes, both zero.

## E08

Stump at t = 0.475 predicts 0 left, 1 right. Train labels:
[0,0,0,1,0,1,1,1]. Point x = 0.7 (label 0) falls right and
is called 1: the single miss. Error 1/8 = 0.125.

## E09

Alpha 0.2: depth-1 cost 0.125 + 0.4 = 0.525. depth-2 cost
0.0 + 0.8 = 0.8. Picks depth-1.

## E10

At n = 1000 there are 1000^1000 possible replicates. the
count grows superexponentially. Enumeration is hopeless.
sampling is the method.

## E11

(3/4)^4 = 81/256 = 0.3164.

## E12

Var((1/B) sum h_i) = (1/B^2)(sum_i Var(h_i) + sum_{i!=j}
Cov(h_i,h_j)) = (1/B^2)(B sigma^2 + B(B-1) rho sigma^2)
= sigma^2/B + (B-1) rho sigma^2/B = rho sigma^2 +
(1 - rho) sigma^2/B.

## E13

0.9 + 0.1/1000 = 0.9001. Almost no help: the shared
wobble dominates at any affordable B.

## E14

Rep 2 = [0,1,2,3] misses nothing, so point 2 loses its
only OOB rep: no OOB prediction for point 2. OOB error
then rests on points 0 and 1 only.

## E15

eps 0.4: alpha = 0.5 ln(1.5) = 0.2027. eps 0.1: alpha =
0.5 ln 9 = 1.0986. The accurate round (eps 0.1) votes
louder.

## E16

[1/6, 1/6, 1/6, 1/2]. Sum: 3/6 + 1/2 = 1. Exact.

## E17

Round 2 fits a stump on weights [1/6,1/6,1/6,1/2]. the
x = 3 point dominates, so the stump contorts to fix it,
likely at the price of a clean point. Defense: cap
rounds, or switch to a loss that resists outliers.

## E18

Residuals [-1, 0, 1], sum 0. The mean start guarantees a
zero-sum residual vector.

## E19

nu = 0.5: F1 = [2 - 0.5, 2 + 0, 2 + 0.25] = [1.5, 2.0,
2.25]. New residuals: [-0.5, 0, 0.75].

## E20

d/dF (1/2)(y - F)^2 = -(y - F) = F - y. Negative
gradient: y - F = r. Shown.

## E21

Predict positive always: accuracy 5/100 = 0.05, worse
than all-negative 0.95. Accuracy rewards the majority
class either way.

## E22

Precision 0.4444, recall 0.8, F1 0.5714. TNR = 90/95 =
0.9474. Balanced accuracy = (0.8 + 0.9474)/2 = 0.8737.

## E23

ROC plots TPR against FPR = FP/(FP + TN). With 95
negatives, even 5 false positives give FPR 0.0526 while
precision is 4/9. The huge TN count dilutes the FPR
axis. the PR curve has no such dilution.

## E24

x2 == x1 exactly, so the split on x2 at 0.5 gives the
same gain 0.5. The tree picks x1 by order, not merit.

## E25

x1 pure noise, x2 = y exactly, but list x1 first and
cap depth at 1 with a tie-break on order: gain
importance x1 0.5, x2 0.0. Any equivalent construction
with the order forced earns full marks.

## E26

0.7^2 + 0.3^2 = 0.49 + 0.09 = 0.58.

## E27

For: 0.83 clears 0.5 by a wide margin on the same data.
Against: n = 6 is tiny. the gap could be luck. report a
confidence interval or a larger test set before
claiming a win.

## E28

Here the stump is the right baseline because the data
is linearly separable by one threshold. beating it is
the minimum bar. It is the wrong baseline when the
true boundary needs depth (XOR): then a stump scores
0.5 and every model beats it trivially.

## E29

Reference implementation:

```
def predict(p):
    x1, x2 = p
    if x1 <= 0.5:
        return 0
    return 1 if x2 <= 0.5 else 1
pts = [(0.2,0.2,0),(0.3,0.7,0),(0.2,0.8,0),(0.7,0.3,1),(0.8,0.8,1),(0.7,0.7,0)]
assert [predict(p[:2]) for p in pts] == [0,0,0,1,1,0]
```

## E30

Accept any hypothesis of the form: claim, baseline,
metric, and a falsification condition. Example: "Two-
step lookahead beats greedy at depth 3 on parity
tasks. baseline greedy depth-3. metric held-out
accuracy. falsified if the gap is within noise over
20 seeds."

## Ladders

L01. (1) A decision region is the set of inputs that
reach one leaf. (2) The three C01 rectangles. (3) Each
split adds one interval constraint per involved
feature. the leaf region is their product. (4)
Accept a recursive predict. (5) With x1 + x2 <= t the
regions become polygons, not rectangles. the
axis-aligned story breaks.

L02. (1) Gini is the error rate of labeling by the
node distribution. (2) 0.32 / 0.5004 / 0.2. (3) Draw
a label from p: the chance it misses a second
independent draw is 1 - sum p_k^2. (4) Both pick
t = 0.5 on the C03 toy. (5) Pay for entropy when
probabilities (not just ranks) feed a later
calibration step. the log scale matches.

L03. (1) Gain = parent impurity minus weighted child
impurity. (2) 0.5 at t = 0.5. (3) It commits to the
best single split without looking deeper. (4) XOR:
all single-split gains are 0. (5) Two-step lookahead
costs O((d n)^2) per node in the naive form. accept
any honest cost analysis.

L04. (1) R_alpha(T) = R(T) + alpha |T|. (2) 0.05 keeps
depth-2 (0.2 < 0.225). 0.1 prunes to depth-1. (3) A
weak first split can open the door to a strong second (XOR).
early stop kills the pair. (4) Accept a working
collapse loop. (5) Train error keeps falling with
leaves, so the penalty never binds honestly. real
splits get cut at high alpha.

L05. (1) n draws with replacement. (2) The three
seed-7 sets. (3) P(miss) = (1 - 1/n)^n -> 1/e.
(4) Accept the OOB table code. (5) The O(n) memory
per replicate and the B sequential fits. also the
OOB bookkeeping at 10^6 rows needs care.

L06. (1) rho sigma^2 + (1 - rho) sigma^2/B. (2)
0.37, 0.604, 0.208. (3) See E12. (4) Accept a
pairwise disagreement estimate. (5) It fails when
tree variances differ a lot (mixed depths) or
correlations are not uniform. then the formula is a
rough guide.

L07. (1) alpha = 0.5 ln((1-eps)/eps). (2) The weight
update numbers. (3) Minimize sum exp(-y(F + alpha h))
over alpha. (4) Accept two rounds with renormalized
weights. (5) See interview keys-u07 T1.

L08. (1) r = y - F. (2) 2.0 -> 0.5. (3) See E20.
(4) Accept the loop with nu = 0.3. (5) AdaBoost
minimizes exponential loss on reweighted points.
gradient boosting minimizes any differentiable loss
by fitting its negative gradient.

L09. (1) Precision = TP/(TP+FP). recall =
TP/(TP+FN). (2) 0.4444, 0.8, 0.5714. (3) The harmonic
mean punishes a low value harder: 2/(1/P + 1/R).
arithmetic mean would let high recall hide low
precision. (4) Accept a scored PR computation. (5)
Report precision at the operating recall (or expected
cost per decision). the risk team prices false
positives, not accuracy.

L10. (1) Gain importance sums impurity gains per
feature. (2) 0.5 vs 0.0. (3) The first split takes
all the credit. order breaks the tie. (4) Accept the
permutation loop. (5) "x2 is useless": attack with
permute-x2 accuracy 1.0 (the tree never reads it)
and the correlation 1.0 (credit is arbitrary).
