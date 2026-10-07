# Keys: Lesson 17, appendices, historical supplements, research synthesis

## Breadth recall

E01: A.1.1 linear combinations of
independent Gaussians stay Gaussian.
A.1.2 conditioning a joint Gaussian.
A.1.3 KL between same-covariance
Gaussians is half the Mahalanobis
distance of the means. A.1.4 chain
rule: joint KL splits into marginal
plus expected conditional KL.

E02: FA: x = mu + Lambda z +
epsilon, low-rank signal plus
diagonal noise. Its math survives in
the Kalman filter's Gaussian
conditioning (U16).

E03: Likelihood (forward algorithm),
most likely path (Viterbi), and
parameter learning (Baum-Welch EM).

E04: A full predictive distribution:
mean plus calibrated variance that
grows away from the data.

E05: One factor at a time, matched
budgets, predict first, report
negative results.

E06: Report mean and spread over all
seeds (never the best seed), a
difference inside the error bars is
not a win, failed runs count.

## Deep oral ladders

L01: (1) A.1.1: sum a_s epsilon_s ~
N(0, (sum a_s^2) I). A.1.3: KL =
(1/2)(m_1 - m_2)^T Sigma^{-1}(m_1 -
m_2). (2) (1/2)(1^2) = 0.5. (3) Log
densities differ only in the
quadratic term in x, constants
cancel, E_P[(X-m_2)^T Sigma^{-1}
(X-m_2)] - d = (m_1-m_2)^T
Sigma^{-1}(m_1-m_2). (4) Script:
0.4998 vs 0.5. (5) Closed form:
exact, needs Gaussian assumptions.
Numerical: general, expensive in
high dimensions. (6) Different
covariances (missing trace/log-det
terms), or too few samples (Monte
Carlo noise). (7) A factorization
that is valid but puts all the KL
in one uninformative term hides
where the models differ. (8)
Sequence KL = sum over t of
E[KL(pi_theta(.|s_t) ||
pi_ref(.|s_t))]: audit per-token
KL to find where drift concentrates.

L02: (1) Ablation: one-factor
removal test. Baseline: the simple
method the new one must beat.
Error bar: the spread of the
estimate. (2) Mean 8.10, sample std
0.18, se 0.058. (3) se = s /
sqrt(n): the sample mean's standard
deviation under IID sampling.
(4) Script: resample 10 with
replacement 20000 times, 2.5 and
97.5 percentiles [7.99, 8.21].
(5) Matched: same data, compute,
tuning per arm, the difference is
attributable. Unmatched: the winner
may just have the bigger budget.
(6) Overlapping error bars: no
evidence of a difference. Diagnose:
more seeds, or a paired comparison
(same seeds per arm) to cut
variance. (7) The best seed is
selection bias: it reports the max
of a noisy draw, not the method's
performance. (8) Arms: linear phi,
quadratic phi, tabular grid. Same
sampled states and simulator calls.
Metric: closed-loop return, 20
seeds. Prediction written first:
quadratic wins, tabular close but
costly. Report all seeds including
divergences.

## Analytical exercises

E07: log p(x) - log q(x) = -(1/2)
[(x-m_1)^T S^{-1}(x-m_1) -
(x-m_2)^T S^{-1}(x-m_2)], constants
cancel. E_P of the bracket: write
x - m_2 = (x - m_1) + (m_1 - m_2).
E_P[(x-m_1)^T S^{-1}(x-m_1)] = d =
tr(I). Cross term zero since
E_P[x - m_1] = 0. Remaining: (m_1 -
m_2)^T S^{-1}(m_1 - m_2). Half
gives (A.9).

E08: Mean 8.10, sample std 0.18,
se = 0.18/sqrt(10) = 0.058. The
t-multiplier for n = 10 (9 df) at
95% is 2.26, not 2, because the
Gaussian 2.0 is the large-n limit,
with 9 degrees of freedom the
heavier t tails need 2.26.

## Failure diagnosis

E09: Rule 1 violated (one factor at
a time). Five changes confound the
result: the experiment shows the
bundle helps, not which component
does. Redo as five single-factor
ablations with matched budgets, or
downgrade the claim to "the bundle
helps."

## Counterfactual comparison

E10: Team A: the shift pages
nobody. No monitoring means nobody
knows. No rollback means the fix is
a new deploy under pressure. No
owner means the incident has no
driver. First 24 hours: confusion,
then a rushed patch. Team B: the
skew alert fires, the canary metric
drops, the tested rollback runs in
minutes, the owner triages. First
24 hours: a controlled revert and a
postmortem with data. The gate is
the difference.

## Research question

E11: Claim: on data rougher than
the RBF kernel's lengthscale, the
GP's 95% posterior intervals cover
the truth less than 70% of the
time. Metric: empirical coverage of
the posterior intervals on held-out
points. Baseline: a GP with a
Mattern kernel fit by marginal
likelihood, which must cover at the
nominal rate.

## Implementation task

E12: Accept if: ridge recovers the
true theta within 0.05 on the
synthetic linear problem, one
k-means iteration does not increase
the objective, value iteration hits
error ratio 0.9, seed statistics
give mean 8.10, se 0.058 within
0.005, KL Monte Carlo is within
0.01 of 0.5.
