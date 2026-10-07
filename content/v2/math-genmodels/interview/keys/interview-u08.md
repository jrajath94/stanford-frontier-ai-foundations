# Interview keys: U08

Unit: math-genmodels-U08. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice. Strong answers, red flags, rubrics,
remediation.

## Q1

Strong: it scores unseen points, so it measures generalization,
not fit. Tuning on it makes it a second training set: the score
inflates and the honesty is gone. Red flag: "more data is
always better." Rubric: the generalization argument plus the
locking rule. Remediation: U08-C01.

## Q2

Strong: whenever the bound is loose: bound -1.70 against exact
-1.7371 leaves the VAE truth in [-1.70, infinity). The bound
wins the reported number and the true ranking stays unknown.
Red flag: "the higher number wins." Rubric: the interval
argument. Remediation: U08-C02.

## Q3

Strong: fraction of true modes the model can generate. The
likelihood averages log-density over points: one good mode
can outweigh a missing mode in the sum. Red flag: "high
likelihood means full coverage." Rubric: the definition plus
the averaging mechanism. Remediation: U08-C03.

## Q4

Strong: sharp, plausible samples that cover only half the
data's variety. Red flag: "the model is good." Rubric: the
one sentence with both halves. Remediation: U08-C04.

## Q5

Strong: truth 0.5 N(-2,0.25)+0.5 N(2,0.25) versus N(0,4.25):
same mean and variance, FID 0, one hump versus two. Or the
3-point discrete law at -2.5612, 0, +2.5612. Red flag: "FID
0 means identical." Rubric: the construction with moments
verified. Remediation: U08-C05.

## Q6

Strong: nearest training distance per sample. a histogram
piled at zero signals a copy. Do not trust eyes: the memorizer's
samples look perfect. Red flag: visual inspection.
Rubric: the test named. Remediation: U08-C06.

## L1 ladder

1. Sum of log densities on unseen points.
2. A: -10.1073, B: -70.949.
3. x = -2.1: B gives -33.85 nats there, the largest single
   gap. B left no mass on the left mode.
4. Exact -1.7371. bound -1.8621 with gap 0.125.
5. Exact versus exact, bound versus bound, gaps labeled.
   never rank a bound against an exact number silently.

## L2 ladder

1. Precision: samples on the manifold. Recall: modes
   covered.
2. A: 0.4708/1.0, B: 0.9555/0.5.
3. Matched moments, FID 0, wrong shape: moments are blind
   past order two.
4. NN 0.0066 (copying) versus 0.4666 (honest).
5. Likelihood misses sharpness, precision misses coverage,
   FID misses shape, eyes miss copying. The suite has no
   single blind spot.

## A1

Strong: 1-D Frechet: (m1-m2)^2 + (s1-s2)^2. Matched: 0.
Discrete: mass 1/3 at -sqrt(6.375), 0, +sqrt(6.375): mean 0,
variance 4.25. Red flag: "only the Gaussian works."
Rubric: the formula plus the construction.

## A2

Strong: SE = sqrt(4.25/5000) = 0.0292. The -0.0317 is 1.09
SEs out: consistent, not bias. Red flag: "the estimator is
biased." Rubric: the SE and the ratio.

## D1

Strong: Inception never saw X-rays: the features are
irrelevant, so FID is blind to the artifacts. Fix: domain
features plus human ratings. Test: two sample sets humans
rank differently must separate on the metric. Red flag:
tuning to the Inception FID. Rubric: the mismatch, the fix,
the test.

## T1

Strong: replace the NN test with a held-out reference split
for precision/recall and output-only membership probes.
Lose: direct copy detection. State the residual risk. Red
flag: "the proxy is equivalent." Rubric: the redesign plus
the loss.

## T2

Strong: held-out likelihood where densities exist (exact).
it scores probability on data. Hides: sharpness and mode
dropping. State both beside the number. Red flag: FID
without caveats. Rubric: choice, defense, hiding.

## R1

Strong: attack 1: FID-0 counterexample, moments only.
Attack 2: "best" needs samples, coverage, cost, task: one
projection cannot decide. Fair: fixed features,
precision/recall, likelihood where exact, matched budgets,
seed uncertainty. Red flag: FID as verdict. Rubric: both
attacks plus the suite.
