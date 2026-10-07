# Keys: Lesson 08, Generalization and sample complexity

## Breadth recall

E01: For regression with truth h*, noise variance
sigma^2, learner h_S, and h_avg = E_S[h_S]:
MSE = sigma^2 + (h* - h_avg)^2 + E_S[(h_avg -
h_S)^2] (irreducible noise + bias^2 + variance).

E02: Irreducible noise is the variance of the label
given the input, sigma^2. No model predicts it. It
lower-bounds MSE.

E03: Curve: test error falls, rises to a peak near
params = n (interpolation threshold), falls again in
the overparameterized regime.

E04: For iid Bernoulli(phi) with mean phi_hat:
P(|phi - phi_hat| > gamma) <= 2 exp(-2 gamma^2 n).

E05: P(A_1 union ... union A_k) <= sum P(A_i).

E06: Uniform convergence: the empirical risk is
close to the true risk simultaneously for all h in
H, not just one fixed h.

## Deep oral ladders

L01: (1) Noise: sigma^2. Bias^2: (h* - h_avg)^2.
variance: E[(h_avg - h_S)^2]. (2) Degrees 1, 2, 5
on quadratic truth: linear underfits, quadratic is
right, degree-5 overfits. (3) Claim 8.1.1 applied
twice as in SL-01. (4) Plot bias^2 falling and
variance rising with degree. (5) The bound view
gives worst-case guarantees over H. The
decomposition gives average-case over datasets.
(6) Bias: the family cannot represent the truth.
(7) It describes the procedure, not the dataset at
hand. For one dataset use validation. (8) Grid
over degrees, validation error, pick the minimum.
report test once.

L02: (1) As E04. (2) n >= (1/(2*0.0025)) *
log(200/0.05) = 200 * 8.294 = 1659. (3) As in
SL-03: Hoeffding per h, union bound over k, solve
for n. ERM within 2 gamma of the best in H.
(4) Function bound_n(k, gamma, delta) returning the
formula. (5) Finite-H is honest but useless for
continuous parameters. The 64-bit argument extends
it with a representation hack. (6) n too small or
gamma too ambitious. Increase n or accept larger
gamma. (7) It depends on the computer, not the
problem. Vacuous for large d. (8) n >= (1/(2 *
0.0009)) * log(100/0.01) = 555.6 * 9.21 = 5118.

## Analytical exercises

E07: n >= (1/(2 * 0.01)) * log(40/0.1) = 50 *
log(400) = 50 * 5.99 = 300.

E08: E[(A+B)^2] = E[A^2] + 2 E[A]E[B] + E[B^2] by
independence. E[A] = 0 kills the cross term.

## Failure diagnosis

E09: With k = 1e6 and n = 100 the uniform bound is
vacuous: log(2e6/0.05) ~ 17.5, gamma >=
sqrt(17.5/200) ~ 0.30 at delta = 0.05. Searching a
million models on 100 points and reporting the best
test error is selection bias. The number means
nothing.

## Counterfactual comparison

E10: Team B. The decomposition explains average
behavior over datasets. Cross-validation measures
this dataset. On real data with unknown truth,
measured error beats derived error.

## Research question

E11: Falsifiable claim: the test-error peak occurs
at params = n, and moves right when n doubles.

## Implementation task

E12: Verified by plotting. Bias^2 falls with
degree, variance rises. The sum is U-shaped in the
classical regime.
