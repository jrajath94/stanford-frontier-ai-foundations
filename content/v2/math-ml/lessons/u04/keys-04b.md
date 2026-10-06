# Lesson 04b keys, entropy, KL, and maximum likelihood

Unit: math-ml-U04. Date: 2026-10-06. Do not read before attempting.
All values computed 2026-10-06, numpy 1.26.4, float64.

## Exercises

E01. H = -(0.75 ln 0.75 + 0.25 ln 0.25) = 0.5623351446188083 nats.
In bits: 0.5623351446188083 / 0.6931471805599453 = 0.811278124459133.
E02. H = 0.5 ln 2 + 0.25 ln 4 + 0.25 ln 4 = 1.039720770839918 nats.
Surprise per unit probability is -log p: B and C give 1.3863 each,
A gives 0.6931. B and C contribute the most surprise per unit.
E03. Error: the formula is differential entropy, which is a relative
quantity and can be negative. A sharp distribution on a short
interval has negative differential entropy. No bug. The discrete
intuition does not transfer.
E04. D_KL = 0.7 ln(0.7/0.5) + 0.3 ln(0.3/0.5) = 0.7(0.3365) +
0.3(-0.5108) = 0.08228287850505181 nats.
E05. D_KL(p||p) = sum p log(p/p) = sum p log 1 = 0. The code returns
0.0 for q = p.
E06. D_KL = 0.7 ln(0.7/1.0) + 0.3 ln(0.3/0.0). The second term
diverges to +infinity. KL is infinite: the support-mismatch failure.
E07. d/dt [p1 ln(p1/t) + p2 ln(p2/(1-t))] = -p1/t + p2/(1-t) = 0
gives t/(1-t) = p1/p2, so t* = p1 = 0.7. The scan measured the
minimum at (0.7, 0.0).
E08. The minimum toll is D_KL(p||[0.5,0.5]) = 0.0823, not 0. The
"converged" report would hide that the family cannot reach the
truth: the residual toll is model bias, not noise.
E09. theta_hat = 1/4 = 0.25. Score: 1/0.25 - 3/0.75 = 4 - 4 = 0.
E10. mu_hat = 2.0. sigma2_hat = ((1-2)^2 + (3-2)^2)/2 = 1.0.
E11. New mu_hat = (2.2*4 + 20)/5 = 5.76. Failure mode: one outlier
drags the mean to a point where four of the five data points do not
live. The MLE has no resistance to contamination.
E12. The n-1 variance is 0.0667. Selection boundary: use the MLE
0.05 when the number feeds the likelihood or a downstream fit. Use
0.0667 when reporting an unbiased variance estimate.
E13. p_hat = [0.4, 0.2, 0.0]. Downstream failure: any test event
with outcome C gets likelihood zero, so the whole product is zero.
E14. Add-one: [5, 3, 1]/13 = [0.3846, 0.2308, 0.0769]. Cost: the
fake counts bias every probability and spend mass on events never
seen. With large n the cost vanishes.
E15. mu_hat = [0.0833, 0.1]. Sigma_hat = [[0.0847, 0.005],
[0.005, 0.1]]. Trace 0.1847. Eigenvalue sum 0.0832 + 0.1015 =
0.1847. Match.
E16. Sigma_hat has rank at most 1 (two centered points span one
direction). The MLE covariance is singular: the density formula
divides by zero determinant. Fix: shrinkage toward a diagonal, or a
diagonal-only model.
E17. p(0) = 0.5 * 0.3989422804014327 + 0.5 * 1.4867e-6 = 0.1995.
It exceeds p(2.5) = 0.01753 because x = 0 sits at the peak of the
left bell while x = 2.5 sits in the valley between the bells.
E18. Error: mu_hat = 2.5 is the valley, where the measured density
is 0.01753, an order of magnitude below the humps at 0.1995. High
likelihood relative to other single bells does not make one bell
the right family. The mixture fits the shape and the single bell
does not.

## Deep ladders, strong answers

L01. Entropy is the average surprise of one draw. Toy: 0.6931 nats
versus 0.5623 nats. The weight p(x) is the frequency of each
surprise: the average is over draws. The log-2 maximum says the
uniform distribution is the foggiest. Gini when splits need speed.
entropy when logs feed downstream math. Break: differential entropy
goes negative for sharp densities. Transfer: [0.51, 0.49] has
higher entropy. The [0.99, 0.01] classifier is more confident, and
entropy is the number that says so.

L02. KL is the extra surprise of navigating with the wrong map,
direction truth-to-model. Toy: 0.0823 versus 0.0872. It is not
symmetric because the averaging distribution differs: each direction
weights the log-ratio by its own truth. Check: H + KL = cross-
entropy, measured to float noise. JS when symmetry is needed.
Break: zero support gives infinite toll. Transfer: the token's
1e-9 probability makes the log-likelihood term explode to -20.7
nats. The fix is to smooth or to put a floor on the output distribution.

L03. The toll is D_KL(data || model). Toy: scan finds t = 0.7. The
coin finds 0.75. The derivative zero is the stationary point of the
toll, and with the truth in the family the stationary point is the
truth. Check: score zero at each estimate. Moment matching when the
likelihood is intractable. Break: misspecified family leaves a
residual toll. Zero heads gives a degenerate zero. Transfer: with
n = 10 the risks are variance (the estimate wobbles) and
misspecification (the family may miss the truth). Report both.

L04. The center is the mean, the width is the mean squared spread.
Toy: N(2.2, 0.05). Score equations: sum (d_i - mu)/sigma^2 = 0 and
-n/(2 sigma^2) + sum (d_i - mu)^2/(2 sigma^4) = 0. Check: score
-1.78e-14. n for likelihood work, n-1 for unbiased reporting.
Break: the 20.0 outlier. Transfer: heavy tails break the Gaussian
score. Replace with a Laplace (median) or Student-t model.

L05. The pair (mu, Sigma) names the center and the stretch. Toy:
the six-point fit and the 0.01753 valley. The covariance MLE is the
average outer product because the score equation for Sigma sets it
equal to the empirical second moment. Check: trace 0.1847 both
ways. Full covariance for small d with real correlations. Diagonal
for large d. Mixture for few true components. Kernels for unknown
count. Break: n < d singular. One bell in the valley. Transfer:
with 100 samples in R^768, fit a diagonal or low-rank-plus-diagonal
covariance: the full matrix has 295k free numbers and 100 samples
cannot fill them.
