# Keys, interview bank U05

Date: 2026-10-06. Scoring rubrics included. Minimum sufficient
answers first.

## Breadth

B1. R(h) = E[l(y, h(x))]. R_hat(h) = (1/n) sum_i l(y_i, h(x_i)).
Training minimizes R_hat. Strong: adds that R is the deployment
price and R_hat is the gamed number. Red flag: claiming training
minimizes R.
B2. Squared: the mean 2/3. Absolute: the median 1. 0-1 majority:
the most frequent class 1. Strong: gives the risks 0.2222, 0.3333,
0.3333. Red flag: "they all give the mean".
B3. MSE = bias^2 + variance + irreducible noise. Needs squared
error and expectation over datasets. Strong: notes it fails for
0-1 loss. Red flag: applying the split to accuracy.
B4. The penalty taxes weight size, so the optimizer trades raw fit
for smaller w. Strong: cites the objective values 3.6531 versus
3.4375. Red flag: "regularization reduces overfitting" with no
mechanism.
B5. The penalty pulls w off the least-squares fit, so raw
residuals must rise. In-sample MSE measures fit, not
generalization. Use CV or a held-out set. Strong: names the
decision rule (minimize the honest score). Red flag: picking
lambda by train error.
B6. Any three: full-data imputation statistics, feature selection
before the split, early stopping on the test set, group overlap
(same patient/user in both parts), time-traveling joins, grand-mean
standardization. Strong: explains the timeline violation in each.
Red flag: only "do not copy the label".

Rubric per breadth item: 2 points for the minimum sufficient
answer, +1 for the strong-answer element, 0 if a red flag appears.
12 points total for the section. Remediation: misses on B1-B3 go
back to lesson C01-C04. Misses on B4-B6 go to C05, C08, C11.

## Deep ladder D1

D1.1. (2 pts) The range of functions the class can express.
D1.2. (2 pts) Train: 0.03075, 0.01513, ~0. Test: 2.8757, 0.5313,
28.5613. Best: degree 2.
D1.3. (3 pts) A richer class contains the poorer one, so the
training minimum over the larger class cannot be worse.
D1.4. (3 pts) Try larger degrees or richer families: the U bottom
lies beyond degree 9. Falling test error means capacity is still
the binding constraint.
D1.5. (3 pts) The bottom moves right: with 400 points, 4
parameters cannot memorize the noise, so degree 3 sits near the
bottom instead of off the cliff. Capacity is relative to n.
13 points. Remediation: D1.2 miss -> re-read f02. D1.3 miss ->
lesson C09.

## Deep ladder D2

D2.1. (2 pts) Split into k parts, rotate the held-out part, average
the k honest scores.
D2.2. (2 pts) n lambda = 2 wins by smallest CV mean: 1.1806 beats
1.8515.
D2.3. (3 pts) The penalty term scales with the number of points in
the fit. Without rescaling, the same lambda would mean different
strengths on different fold sizes.
D2.4. (3 pts) Two distinct: the CV score was gamed by the search
(selection bias: best of many lambdas is optimistic), or the
deployment distribution shifted after the CV.
D2.5. (3 pts) Attack: the winning CV mean is the maximum over the
search, hence optimistic. Reporting it as the final number replays
the train-score fallacy one level up. Nested CV (or a final
holdout) is the price of an honest report. "Wasteful" confuses
compute cost with correctness.
13 points. Remediation: D2.4 miss -> lesson C08 failure case and
C12.

## Quantitative

Q1. Sxx = 14, Sxy = 13. OLS w = 13/14 = 0.928571. Ridge w = 13/16
= 0.8125. Objective J(w) = RSS + 2 w^2: at OLS 3.653061, at ridge
3.4375. Ridge wins. 5 points: 1 per number, 1 for the verdict.
Q2. Bias = 3.8/6 - 0.7 = -0.0667. Var = 4*0.21/36 = 0.02333. MSE =
0.00444 + 0.02333 = 0.02778. MLE MSE = 0.0525. Laplace wins.
5 points.

## Debug T1

Diagnosis: rng.shuffle(X) permutes the rows of X but not y, so
each feature row is paired with the wrong label. The model trains
and tests on misaligned pairs: the true relation (label = sign of
x_0) is destroyed. The three numbers: 1.0 without the shuffle is
the separable truth. 0.2 with the shuffle is below chance because
the misalignment is systematic on this seed (only 35% of rows kept
their true label, and the median threshold then anti-correlates).
0.35 is the measured label agreement after shuffling. Fix: shuffle
indices once and apply to both, or split first and never shuffle
features alone. Minimum sufficient: "the shuffle misaligned X and
y". Strong: explains why 0.2 < 0.5 on this seed and gives the
index-based fix. Red flag: blaming the model or the split ratio.
5 points. Remediation: lesson C11 (process bugs beat model bugs).

## Scenarios

S1. Strong answer: check the feature's timestamp logic first (does
"days since last login" use information from after the prediction
date?). Confirm by rebuilding the feature with a strict cutoff and
watching test AUC collapse toward 0.61. Change: point-in-time
correct feature pipeline, backtest on a time split. Red flag:
"the model overfit. Add regularization". 5 points.
S2. Strong answer: the closed gap says data is not the constraint.
the plateau near the noise floor says the family is adequate but
the features may be exhausted. Recommend feature/model work, not
labels. Settling experiment: fit a richer family on current data.
if validation does not move, the plateau is the limit. Red flag:
"more data never hurts, buy the labels". 5 points.

## Research critique R1

Strong position: double descent challenges the classical U's
universality (test error can fall again past interpolation in some
overparameterized regimes) but leaves the mechanism intact: f01's
bias falls and variance rises with flexibility on this toy, and
f02's U holds in the classical regime. What would change the mind:
a controlled experiment on the lesson's own polynomial toy showing
a second descent with matched budgets and seeds, not a citation.
Red flag: "double descent proves bigger is always better". 5 points.

Total: 12 + 13 + 13 + 10 + 5 + 10 + 5 = 68 points. Pass bar: 48.
Ladder follow-ups must be answered in order. A D-item miss sends
the learner to the matching lesson section before the next item.
