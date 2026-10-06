# Interview bank, U04a probability and estimation (Lec 02-10)

Date: 2026-10-06. Questions only. Keys in interview/keys-u04a.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Define a random variable in one sentence, then give the PMF of a
fair die.
B2. What is the difference between a joint distribution and a
conditional distribution, in plain words?
B3. State the expectation and variance of a fair six-sided die.
B4. What does the IID assumption license, and what breaks without
it?
B5. Define the likelihood of data under a parameter, in one
sentence.
B6. What is the 1/sqrt(n) rule about, and what does it say for
n = 100?

## Deep ladder D1, likelihood and MLE (5 follow-ups)

D1.1. Define the likelihood L(theta) and the log-likelihood l(theta)
for IID data.
D1.2. Toy: flips H, T, H, H. Compute theta_hat and verify the score
is zero there.
D1.3. Derive or justify: why do we maximize the log instead of the
likelihood itself?
D1.4. Implement/debug: a teammate's MLE for a coin returns 0.0 after
four tails. The downstream pipeline then assigns zero probability to
a test set containing heads and everything collapses. Diagnose and
name the fix.
D1.5. Changed constraint: the flips are not IID but come in
correlated pairs. Which MLE step breaks first, and what is the
replacement?

## Deep ladder D2, expectation and the sample (5 follow-ups)

D2.1. Define expectation for a discrete variable.
D2.2. Toy: a sample of 8 draws from a Bernoulli(0.3) has mean 0.5.
Is the sample "wrong"? Explain using the 1/sqrt(n) rule.
D2.3. Derive or justify: why is the sample mean an unbiased
estimator of the population mean?
D2.4. Implement/debug: a colleague estimates variance with divide-
by-n and reports it as "the unbiased variance." Name the error and
the size of the bias for n = 4.
D2.5. Research critique: "With n = 1e9, the empirical distribution
equals the true distribution, so density estimation is solved."
Attack the claim.

## Analytical/quantitative (2)

Q1. Joint table: P(X=0, Y=0) = 0.4, P(0, 1) = 0.1, P(1, 0) = 0.2,
P(1, 1) = 0.3. Compute both marginals, P(Y=1 | X=0), and state
whether X and Y are independent.
Q2. Four data points [2.1, 2.5, 1.9, 2.3]. Compute the MLE Gaussian
parameters and state the score-check value from the lesson.

## Implementation/debug (1)

T1. This code intends to verify that PMF rows sum to 1:

```python
import numpy as np
pmf = np.array([2/7, 3/7, 2/7])
print(pmf.sum() == 1.0)
```

It prints False: the sum is 0.9999999999999999. Name the cause, fix
it, and state the general rule for float equality in checks.

## Changed-constraint scenarios (2)

S1. Your data are chest X-ray images, not coin flips (Lec 06
framing). Which U04a facts survive unchanged, and which modeling
step must change first?
S2. Your sample has n = 8 but the distribution has 1000 outcomes.
Name the estimation failure and the practical replacement for the
raw empirical distribution.

## Research critique (1)

R1. "Our histogram density with 4 bins on 8 points matches the true
density's shape, so the density estimate is validated." Critique:
what does the 1/sqrt(n) rule say about this claim, and what
experiment would actually validate the estimator?
