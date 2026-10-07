# Interview bank: U08 (questions)

Unit: math-genmodels-U08. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u08.md.

## Breadth (6)

Q1. Why is held-out likelihood the honest score? What breaks if
you tune on it?
Q2. When can a VAE bound beat a flow's exact likelihood and
still lose the true comparison?
Q3. Define mode coverage. Why does likelihood average it away?
Q4. A model has precision 1.0 and recall 0.5. Describe its
samples in one sentence.
Q5. Construct two laws with FID 0 that look nothing alike.
Q6. How do you test for memorization without trusting your
eyes?

## Deep ladders (2 x 5)

L1 (scoring):
1. Define held-out likelihood.
2. Compute A versus B on the toy.
3. Name the point that decides it and why.
4. Separate bound from exact with the toy numbers.
5. State the comparison rule for mixed families.

L2 (samples):
1. Define precision and recall for samples.
2. Compute all four toy numbers.
3. Show the FID-0 counterexample.
4. Run the NN memorization test.
5. Explain why the suite beats any single metric.

## Analytical exercises (2)

A1. Derive the 1-D FID formula and construct the discrete
FID-0 counterexample.
A2. Compute the standard error of the synthetic recovery and
judge the -0.0317.

## Implementation and debug (1)

D1. FID is good but humans reject the samples on X-ray data.
Diagnose, fix, and give the test.

## Changed-constraint scenarios (2)

T1. The training set cannot leave the hospital. Redesign the
evaluation and name the loss.
T2. The leaderboard takes one number. Pick it, defend it, and
name what it hides.

## Research critique (1)

R1. "Best FID, therefore best generative model." Attack with
two arguments and propose the fair experiment.
