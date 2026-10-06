# Interview keys, U04a probability and estimation (Lec 02-10)

Date: 2026-10-06. Computed values, numpy 1.26.4, float64.

## Breadth

B1. A random variable maps outcomes to numbers. Fair die PMF:
P(X = k) = 1/6 for k = 1..6.
B2. The joint distribution describes two variables together. The
conditional zooms in on one variable after fixing the other and
renormalizes.
B3. Expectation 3.5. Variance 35/12 = 2.9167.
B4. IID licenses treating the sample as repeats of one distribution:
it justifies the product-form likelihood and generalization from
sample to population. Without it, the product form breaks and the
sample may mislead.
B5. The likelihood is the probability (density) of the observed data
as a function of the parameter.
B6. The 1/sqrt(n) rule: estimation error shrinks like 1/sqrt(n).
For n = 100 the typical error scale is 0.1.

## Deep ladders

D1.1. L(theta) = product p(x_i. Theta). l(theta) = sum log p(x_i.
theta).
D1.2. theta_hat = 3/4 = 0.75. Score: 3/0.75 - 1/0.25 = 0.
D1.3. The log turns the product into a sum (numerical stability,
no underflow) and is monotone, so the argmax is unchanged. Sums
differentiate term by term.
D1.4. Diagnose: zero heads in four flips gives MLE 0, a
support-mismatch: the model calls heads impossible. Fix: Laplace
add-one smoothing, or a Beta prior (MAP), so no observed-class
probability is exactly zero.
D1.5. The product-form likelihood breaks first: correlated pairs do
not factor. Replacement: a joint model for the pair, or a
composite/pairwise likelihood.

D2.1. E[X] = sum x p(x): the probability-weighted center.
D2.2. Not wrong: the 1/sqrt(n) rule says the standard error of the
mean at n = 8 is about sqrt(0.3*0.7/8) = 0.162, so 0.5 is about 1.2
standard errors from 0.3. Unlucky, not broken.
D2.3. E[sample mean] = (1/n) sum E[X_i] = mu by linearity of
expectation. IID gives each term mean mu.
D2.4. Error: divide-by-n is the MLE, which is biased down by a
factor (n-1)/n. The unbiased version divides by n - 1. For n = 4
the bias factor is 3/4: the MLE reports 75 percent of the unbiased
value on average.
D2.5. Attack: 1e9 samples do not cover a large or continuous space.
the empirical distribution is still spiky and the 1/sqrt(n) error,
while small per bin, is not zero. Tails and rare events remain
unseen. Density estimation is not solved. Smoothing and model
choice still decide.

## Analytical/quantitative

Q1. P(X=0) = 0.5, P(X=1) = 0.5. P(Y=0) = 0.6, P(Y=1) = 0.4.
P(Y=1 | X=0) = 0.1/0.5 = 0.2. Independent? No: 0.1 != 0.5*0.4 =
0.2.
Q2. mu_hat = 2.2, sigma2_hat = 0.05. Score check:
-1.7763568394002505e-14, numerical zero.

## Implementation/debug

T1. Cause: binary floating point cannot represent 2/7 and 3/7
exactly, so the sum is 0.9999999999999999, not 1.0. Fix: compare
with a tolerance, abs(pmf.sum() - 1.0) < 1e-12, or use
np.isclose. General rule: never test float equality with ==. Test
closeness with a tolerance.

## Changed-constraint scenarios

S1. Survives: random variables, PMF/PDF/CDF, expectation, IID as a
modeling choice, the likelihood principle. Must change first: the
model family. Pixels are high-dimensional and dependent, so the coin
and die models give way to densities over images (the Lec 06 chest
X-ray framing).
S2. Failure: the empirical distribution puts mass 1/8 on 8 outcomes
and zero on 992: extreme sparsity, and any test point outside the 8
gets probability zero. Replacement: smoothing, a parametric model,
or a low-dimensional embedding before estimation.

## Research critique

R1. The 1/sqrt(n) rule says: with n = 8 the bin counts have
standard error about sqrt(8)/4 ~ 0.7 per bin on counts of ~2: the
"shape" is mostly noise. The claim mistakes a noisy picture for
validation. Real validation: repeated sampling (bootstrap or fresh
samples), comparing the histogram against held-out data, or
measuring a proper scoring rule on a test set.

## Scoring rubric (per ladder)

- Define (1 pt): one crisp sentence.
- Toy (1 pt): correct numbers.
- Derive/justify (2 pts): the key step named.
- Implement/debug (2 pts): the mechanism or bug named exactly.
- Compare/break/transfer (2 pts): boundary or failure with a case.
- Red flags: confusing likelihood with probability of the
  parameter, "n large fixes everything", float == in checks.
- Remediation: miss on D1 -> reread U04a SB07 and lesson-04b SB12.
  Miss on D2 -> reread U04a SB04-SB05 and SB08. Miss on T1 ->
  reread U01-C11 precision.
