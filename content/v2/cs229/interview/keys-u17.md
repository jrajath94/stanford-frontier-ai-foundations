# Keys: interview U17

## Breadth

Q01: A.1.1: weighted sums of
independent Gaussians are Gaussian
with summed squared weights. A.1.2:
Gaussian conditioning formula.
A.1.3: same-covariance KL is half
the Mahalanobis mean distance.
A.1.4: KL chain rule.

Q02: x = mu + Lambda z + epsilon,
z ~ N(0,I), epsilon ~ N(0,Psi
diagonal). Survives in the Kalman
filter's Gaussian conditioning.

Q03: It writes a sequence KL as a
sum of per-step conditional KLs,
which is what makes per-token KL
penalties in RLVR well-defined.

Q04: One factor at a time, matched
budgets, predict first, report
negative results.

Q05: Report all seeds with spread,
overlapping bars are not a win,
failed runs count.

Q06: Train/serve skew, latency and
cost, feedback loops, reward
validity at scale, rollback,
ownership.

## Deep ladders

L01: F1: KL = (1/2)(m_1 -
m_2)^T Sigma^{-1}(m_1 - m_2). F2:
0.5. F3: Constants cancel, cross
term has zero mean under P, half
the quadratic remains. F4: When
covariances differ (missing trace
and log-det terms) or Sigma is
singular. F5: KL(q(x_1:T) ||
p(x_1:T)) = KL(q(x_T)||p(x_T)) +
sum_t E[KL(q(x_{t-1}|x_{t:T}) ||
p(x_{t-1}|x_{t:T}))], then read
each term as a token-level penalty.

L02: F1: Same data, same compute,
same tuning effort per arm, only
the ablated factor differs. F2:
Confounded: the bundle's effect is
measured, not the component's.
Attribution is impossible. F3:
0.058. F4: No evidence of a
difference. Options: more seeds, or
a paired design (same seeds per
arm) to remove shared noise. F5:
Offline baseline beaten on matched
budgets, canary on live traffic,
monitored metrics with alerts, a
tested rollback, a named owner.

## Analytical

A01: Scalar case: epsilon_s ~ N(0,1)
independent, S = sum a_s epsilon_s.
MGF: M_S(t) = prod_s exp(a_s^2 t^2
/ 2) = exp(t^2 sum a_s^2 / 2), the
MGF of N(0, sum a_s^2). MGFs
determine the distribution.

A02: The bound (R/gamma)^2 depends
only on the radius R and the margin
gamma, not on n. In the online
setting n is unbounded (the stream
never ends), so a bound growing
with n would be vacuous. The
mistake bound says errors stop
after a finite count no matter how
long the stream runs.

## Implementation and debug

D01: A.1.3 gives 0.5, but the Monte
Carlo lands near 0.44: the true KL
is (1/2)(1 + 0.5 - 2 + log 4)
= 0.443. The violated condition is
"same covariance": A.1.3 needs
Sigma_P = Sigma_Q, and here they
differ.

## Changed-constraint scenarios

S01: A.1.4 (chain rule) survives:
it is about factorizations, not
Gaussians. A.1.1-A.1.3 do not.
The Kalman update is replaced by a
particle filter or another
nonparametric belief update: no
closed form without Gaussianity.

S02: Run the ablation of the single
component the paper's story depends
on most (the one the title claims).
It supports exactly one claim:
"removing X changes the metric by
delta under these conditions." It
cannot support "X is why the method
works."

## Research critique

R01: Two numbers with no spread,
no sample size, and no baseline
budget statement prove nothing.
Under the rules: report all seeds
with error bars, check overlap,
name the tuning budget. 0.83 vs
0.79 with overlapping intervals is
noise. Strong answer asks for the
missing numbers before engaging
with the claim.
