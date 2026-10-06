# Answer keys, U04a first source block

Date: 2026-10-06. Kept separate from the lesson per the assessment rule.
Test-mode: read the block lesson first, answer closed-book, then check
here. All numbers computed 2026-10-06, numpy 1.26.4, float64.

## E01

Sample space of two coin flips: {HH, HT, TH, TT}, 4 outcomes. Each has
probability 1/4 under the fair independent model.

## E02

Yes: S(HH) = 2, S(HT) = S(TH) = 1, S(TT) = 0. Each outcome maps to
exactly one number, so S is a random variable on this space.

## E03

PMF rows: 0.1 five times plus 0.5 = 1.0. Valid PMF: rows are
nonnegative and sum to 1.0.

## E04

Failure: rows sum to 1.2, violating the total-1 axiom. Not a PMF.
Fix: divide every row by 1.2, or find the missing/extra mass. Never
treat unnormalized scores as probabilities.

## E05

Marginal of Y from the SB03 table: P(Y = 0) = 0.3 + 0.1 = 0.4,
P(Y = 1) = 0.2 + 0.4 = 0.6. Column sums. Check: 0.4 + 0.6 = 1.0.

## E06

P(X = 1 | Y = 0) = P(1, 0) / P(Y = 0) = 0.1 / 0.4 = 0.25. Zoom into
the Y = 0 column and renormalize.

## E07

Failure: conditioning on P(X = 0) = 0 divides by zero. The conditional
distribution is undefined: there is no X = 0 row to renormalize.
Guard: check the marginal is positive before dividing.

## E08

E[X] for Bernoulli(0.3) = 0 * 0.7 + 1 * 0.3 = 0.3. The mean equals the
success probability.

## E09

Loaded die E = 4.5, Var = 3.25 measured. Check: weights sum to 1.0.
E = 1*.1+2*.1+3*.1+4*.1+5*.1+6*.5 = 4.5. Variance 3.25 > 2.9167 of
the fair die: the load spreads outcomes wider.

## E10

Failure: the Cauchy distribution has no finite expectation. The
sample mean of Cauchy draws never settles. It wanders with jumps.
"Average" is meaningless here. Use the median instead.

## E11

The 8 draws have 4 ones (mean 0.5). Adding one more draw of 1: new
mean 5/9 = 0.5555555555555556 measured. One draw moves the estimate
by 0.0556: small n means fragile estimates.

## E12

Failure: a biased archive (only severe cases) is a sample from the
wrong P. No estimator fixes a wrong sampling distribution. Fix at
collection time: sample from the deployment population, or reweight
with known selection probabilities.

## E13

IID joint for H, T, H at p = 0.6: 0.6 * 0.4 * 0.6 = 0.144. The
product form needs both IID halves: same p each flip, independent
flips.

## E14

Failure: scanner drift breaks "identically distributed". Early and
late images obey different P, so one pooled estimate mixes two
truths. Fix: model the drift (time-varying P) or split the sample.

## E15

MLE for 10 flips with 7 heads: p_hat = 7/10 = 0.7. Derivative check:
7/0.7 - 3/0.3 = 10 - 10 = 0.

## E16

Log-likelihood at 0.7: 7*log(0.7) + 3*log(0.3) = -6.108643020548935
measured.

## E17

Failure: with n = 1 the MLE is 0 or 1, total certainty from one draw.
The likelihood at the boundary is degenerate. This is why MAP with a
prior (Lec 25) exists: the prior keeps tiny-n estimates honest.

## E18

Histogram of [0.1, 0.4, 0.9, 1.2] with 2 bins on [0, 1.5]: edges
[0, 0.75, 1.5], counts [2, 2] measured.

## E19

Predicted typical error: sqrt(0.3*0.7/8) = 0.1620185174601965
measured. Observed error: |0.5 - 0.3| = 0.2. Same order: the rate
predicts the miss size, not the exact miss.

## E20

Research design (open, graded on falsifiability): rerun the
Bernoulli(0.3) toy with n = 800, seed 7. Hypothesis: |mean - 0.3| <
0.05. Baseline: n = 8 gave error 0.2. Metric: measured absolute error
of the sample mean. The 1/sqrt(n) law predicts error ~0.016. Failure
criterion: hypothesis rejected if the measured error exceeds 0.05.
rate claim rejected if the error does not shrink by roughly
sqrt(800/8) = 10x versus the n = 8 run.

## Ladder answer sketches L01-L04

Each ladder is graded define -> toy -> derive -> implement ->
complexity -> compare -> debug -> critique -> design. Strong answers
quote one measured number from this block (for example marginals
[0.5, 0.5], log-likelihood -2.249340578475233, error 0.2 vs predicted
0.1620). Common red flag: confusing the sample with the
distribution. Rubric: 2 points per rung, 18 per ladder. Remediation
for a failed rung is the matching SB section.
