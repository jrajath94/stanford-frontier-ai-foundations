# Answer keys: U02

Unit: math-genmodels-U02. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: p(x) = sum_z p(z) p(x|z). P(R) = 0.6 x 0.8 + 0.4 x 0.3 =
0.60. Red flag: reporting a joint mass as the marginal. Rubric: the
formula plus 0.60.

## B2

Strong: the posterior is p(z|x), the law over hidden causes after
data. P(A|R) = 0.48/0.60 = 0.8. P(A|G) = 0.12/0.40 = 0.3. Red flag:
answering with the likelihood p(R|A) = 0.8 and calling it the
posterior. Rubric: both numbers with the normalizer shown.

## B3

Strong: p(z|x) = p(x|z) p(z) / p(x). After one red: (0.8, 0.2). Use
it as the prior for the second red: joints (0.64, 0.06), marginal
0.70, posterior (0.914, 0.086). Red flag: multiplying likelihoods
without renormalizing. Rubric: the intermediate posterior and the
final pair.

## B4

Strong: log is concave, so E[log X] <= log E[X]. On the toy:
E_q[log v] = -1.1499, log E_q[v] = log 0.372 = -0.9889. The
inequality holds. Red flag: flipping the direction. Rubric: the
curvature named and both numbers.

## B5

Strong: joint form E_q[log p(x,z) - log q(z)] = -0.5390. Working
form E_q[log p(x|z)] - KL(q||p(z)) = -0.5174 - 0.0216 = -0.5390.
Red flag: dropping the entropy term. Rubric: both forms computed to
agreement.

## B6

Strong: the gap is KL(q(z)||p(z|x)) = 0.0282. ELBO + gap =
-0.5390 + 0.0282 = -0.5108 = log p(R). Red flag: calling the gap a
free parameter. Rubric: the KL computation and the sum check.

## L1 ladder

1. Derive: log p(x) = log sum_z q(z) p(x,z)/q(z) = log E_q[ratio]
   >= E_q[log ratio] by Jensen, log concave. That expectation is the
   ELBO.
2. The gap is KL(q(z)||p(z|x)), the distance from q to the true
   posterior.
3. On the toy: 0.0282 nats, from 0.7 x log(0.7/0.8) + 0.3 x
   log(0.3/0.2).
4. q = posterior makes the KL zero, so ELBO = log p(x) = -0.5108
   exactly. The bound is tight.
5. Grid q_A over 0.05 to 0.95 in steps of 0.05. At each point
   compute ELBO and KL(q||posterior). The sum stays -0.5108 within
   1e-12 everywhere. Constant sum confirms the identity.
Red flags: deriving the ELBO without naming where Jensen is used.
Rubric: each rung correct, the experiment falsifiable.

## L2 ladder

1. Define: write z = g(theta, eps) with eps from a fixed law. Then
   d/dtheta E_q[f] = E_eps[df/dg x dg/dtheta]. The randomness no
   longer depends on theta.
2. Compute: E = 1 + 4 = 5. dE/dmu = 2 x 1 = 2.0. dE/dsigma =
   2 x 2 = 4.0.
3. The score-function estimator multiplies f(z) by d log q/dtheta,
   which has large variance. Reparameterization differentiates a
   deterministic map, so the sampling law stays fixed and the
   variance drops. Measured: std 0.2546 versus 0.1171.
4. State the law: std scales as 1/sqrt(K). Measured 0.3956, 0.2041,
   0.1001 at K = 100, 400, 1600. Ratios near 2 per quadrupling.
5. Decide: target std 0.05 from 0.1001 at K = 1600 needs K near
   6400 by the 1/sqrt law.
Red flags: claiming reparameterization removes bias. It reduces
variance. Rubric: the numbers and the K decision with the law.

## A1

Strong: KL(q||p(z|x)) = E_q[log q - log p(z|x)]. Substitute p(z|x)
= p(x,z)/p(x). The log p(x) term is constant in z: E_q[log q - log
p(x,z)] + log p(x) = -ELBO + log p(x). Rearranged: log p(x) = ELBO
+ KL. Each term: ELBO is the tractable expectation, KL is the
unobservable gap. Red flag: expanding the KL in the wrong
direction. Rubric: the substitution and the rearrangement.

## A2

Strong: KL(N0||N1) = 0.5 x (tr(S1^-1 S0) + dmu^T S1^-1 dmu - k +
log(det S1/det S0)). With S1 = I, mu = 0, S0 = [[1, 0.9],[0.9, 1]]:
tr = 2, det S0 = 0.19, KL = 0.5 x (2 - 2 + log(1/0.19)) = 0.8304.
Red flag: forgetting the -k term. Rubric: the formula and the
verified number.

## D1

Strong: .detach() on the whole z expression severs the autograd
graph. mu and sig get zero gradient on every step, so the parameter
update is 0.0 while the loss still moves under other terms. Fix:
z = mu + sig * eps with no detach. Verification: compare the
gradient against the closed form 2 x mu = 2.0, and assert it is
nonzero after one step. Red flag: "fixing" by raising the learning
rate. Rubric: root cause plus an independent check.

## T1

Strong: read "prior 0.1" with the old masses scaled to 0.9: prior
(0.54, 0.36, 0.10). Marginal: 0.54 x 0.8 + 0.36 x 0.3 + 0.10 x 0.5
= 0.432 + 0.108 + 0.050 = 0.590. Posterior: (0.732, 0.183, 0.085).
The old q padded with 0 gives log q_C = -inf, so the ELBO is -inf.
What breaks: the 2-state q family cannot cover a 3-state latent.
The family must grow a third state. Red flag: renormalizing the
posterior without rescaling the prior. Rubric: the rescaling stated
and the -inf diagnosed.

## T2

Strong: 1 KB holds 256 float32 parameters. The encoder has 13h + 4
params, so h <= 19. New amortized count: 251. New crossover: 4n >
251, so n >= 63. The smaller encoder may underfit, so the
amortization gap can grow: verify on held-out points with a few
per-point refinement steps. Red flag: keeping h = 32 and hoping.
Rubric: the memory arithmetic and the new crossover.

## R1

Strong: counterexample 1: fix the model, improve q from the prior
to (0.7, 0.3). ELBO rises from -0.6155 to -0.5390. The model did
not change. Only the bound tightened. Counterexample 2: the
label-swapped model of C12 has the identical marginal and hence
identical likelihood, while its parameters differ completely.
Higher ELBO can also come from a richer q family with the same
p(x). Settling experiment: on the toy, grid over q with the model
fixed and show ELBO varies while the exact log marginal stays
-0.5108. For model comparison use the exact marginal or held-out
likelihood with matched q families. Red flag: proposing a bigger
encoder as the fix. Rubric: two valid counterexamples plus an
experiment with a decision rule.
