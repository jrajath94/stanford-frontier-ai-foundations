# Keys, interview U07 trees and ensembles

Date: 2026-10-06. Each answer: minimum sufficient
explanation, strong answer, red flags, rubric, remediation.

## B1

Minimum: a decision region is the input set that reaches
one leaf. every region is an axis-aligned rectangle (a
product of intervals).
Strong: adds that the rectangle comes from intersecting
one interval constraint per split on the path.
Red flags: "regions are circles". "any shape".
Rubric: definition 1, rectangle 1. Remediation: C01.

## B2

Minimum: Gini 1 - sum p_k^2. entropy -sum p_k log p_k.
misclassification 1 - max p_k. Never grow with
misclassification: it is flat, ties stall the greedy
search.
Strong: adds strictly concave vs piecewise flat, with
the tie mechanism.
Red flags: "misclassification is fine, it is the true
objective". Rubric: formulas 2, selection 1.
Remediation: C02.

## B3

Minimum: Gini 0.32, entropy 0.5004 nats,
misclassification 0.2.
Strong: shows p = [0.8, 0.2] and each arithmetic step.
Red flags: entropy in bits without saying so (0.7219
bits is also correct if labeled).
Rubric: 1 per score. Remediation: C02, E03.

## B4

Minimum: rho sigma^2 + (1 - rho) sigma^2/B. At rho = 1
the variance stays sigma^2: no help at any B.
Strong: derives it or states the shared-wobble
interpretation.
Red flags: "variance goes to zero as B grows" without
the rho qualifier.
Rubric: formula 2, rho=1 case 1. Remediation: C06.

## B5

Minimum: alpha = 0.5 ln((1-eps)/eps). w_i <- w_i
exp(-alpha y_i h(x_i)), normalize. alpha = 0 means
eps = 0.5: the stump is chance-level, contributes
nothing.
Strong: derives alpha from the exponential loss.
Red flags: "alpha = 0 means perfect stump".
Rubric: formulas 2, interpretation 1. Remediation:
C08, lab Task 2b.

## B6

Minimum: precision TP/(TP+FP), recall TP/(TP+FN), F1
their harmonic mean. Recall 0 gives F1 = 0.
Strong: notes accuracy 0.95 is the majority-class
trap.
Red flags: reporting accuracy as the headline.
Rubric: definitions 2, F1 = 0 with reason 1.
Remediation: C10.

## D1

D1.1. Gain = parent impurity minus weighted child
impurity.
D1.2. Best t = 0.5, gain 0.5, weighted child Gini
0.0.
D1.3. Sort O(n log n) once. sweep O(n) with running
counts. The sort dominates.
D1.4. See T1.
D1.5. Midpoints are meaningless for unordered
levels. Fix: order levels by mean target (or try all
2^{k-1} - 1 partitions for small k), or one-hot/hash.
Strong: names the ordering trick and its cost.
Rubric: 2 per follow-up. Remediation: C03.

## D2

D2.1. Bagging fits B trees on bootstrap replicates
and averages. It cannot fix bias.
D2.2. 0.37 and 0.604.
D2.3. See keys E12 (U07 lesson).
D2.4. Bagging wins on high-variance low-bias
learners (deep trees). boosting wins when the weak
learner has signal but high bias, and on clean data.
Strong: cites the XOR-stump failure for bagging and
the outlier-chasing failure for boosting.
D2.5. Attack: the formula's first term rho sigma^2
survives any B. measured 0.604 at rho = 0.6,
B = 100. "Goes to zero" needs rho = 0.
Rubric: 2 per follow-up. Remediation: C05-C09.

## Q1

p = [0.75, 0.25]. Gini = 1 - (0.5625 + 0.0625) =
0.375. Entropy = -(0.75 ln 0.75 + 0.25 ln 0.25) =
0.75(0.2877) + 0.25(1.3863) = 0.2158 + 0.3466 =
0.5623. Misclassification = 0.25. Children pure:
weighted Gini 0. Gain 0.375.

## Q2

alpha = 0.5 ln(0.8/0.2) = 0.5 ln 4 = 0.6931.
Correct: 0.25 e^-0.6931 = 0.125. missed: 0.25
e^0.6931 = 0.5. Sum = 3(0.125) + 0.5 = 0.875.
Normalized: [1/7, 1/7, 1/7, 4/7]. Check: 3/7 +
4/7 = 1.

## T1

(a) Weighted sum-of-squares: t=0.2: 0.6, 0.35:
0.75, 0.5: 1.0, 0.7: 0.75, 0.85: 0.6. Minimum
0.6 at 0.2/0.85. (Executed 2026-10-06.)
(b) Diagnosis: sum p^2 measures purity, not
impurity. its minimum picks the most impure
split.
(c) Fix: score = 1 - (p**2).sum(). Correct pick:
t = 0.5.
Strong: also notes maximizing sum p^2 would have
been equivalent to the Gini minimum.
Red flags: "the code needs more data".
Rubric: reproduce 2, diagnose 2, fix 1.
Remediation: C02-C03.

## S1

Minimum: report precision/recall/F1 or PR-AUC, not
accuracy. baseline = always-negative (0.9999) and a
one-rule model. pick the threshold by expected cost
per decision on validation. tell the team the
accuracy headline is the baseline, not a result.
Drift: positives change shape -> precision at the
fixed threshold collapses. monitor the positive
rate and PR, not accuracy.
Strong: names a concrete cost matrix and a
rollback trigger.
Red flags: shipping on accuracy. no baseline.
Rubric: metrics 2, baseline 1, threshold 1,
drift 1. Remediation: C10, C12.

## S2

Minimum: (1) duplicates/near-copies leaking across
the OOB boundary -> check duplicate rate, cost one
pass. (2) distribution shift between OOB-era data
and test -> check input summaries, cost one pass.
(3) OOB computed on too few points or with a bug
(e.g. in-bag trees voting) -> recompute OOB from
stored replicates, cost one refit-free pass.
Strong: orders by cost and names the exact check.
Red flags: "forests always generalize".
Rubric: 2 per mechanism with check. Remediation:
C07.

## R1

Minimum: attack with permute-x2 accuracy 1.0 (the
tree never reads x2) and correlation(x1,x2) = 1.0
(the credit split is arbitrary). Gain importance
describes the tree, not the world.
Strong: adds that removing x2 removed the only
redundant spare. justification needs drop-column
retraining or a controlled ablation showing no
metric change.
Red flags: "importance is causation".
Rubric: two numbers 2, mechanism 2, valid
evidence 1. Remediation: C11.
