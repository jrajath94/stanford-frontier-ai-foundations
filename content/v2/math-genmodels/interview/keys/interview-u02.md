# Interview keys: U02

Unit: math-genmodels-U02. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers, red flags, and rubrics. Original practice material.

## Q1

Strong: p(x) = sum_z p(z) p(x|z). Continuous z: the sum becomes an
integral, p(x) = integral p(z) p(x|z) dz. Red flag: writing the
joint as the answer. Rubric: both forms with the sum-to-integral
change named.

## Q2

Strong: the posterior p(z|x) is the belief over hidden causes after
data. The likelihood p(x|z) scores data under a fixed cause. They
point in opposite causal directions and differ by the prior and the
evidence. Red flag: using the two interchangeably. Rubric: the
direction contrast stated.

## Q3

Strong: ELBO = E_q[log p(x|z)] - KL(q(z)||p(z)). The first term
rewards reconstructions of the data. The second pulls q toward the
prior and charges for latent information. Red flag: calling the KL
term a prior. It is a divergence, not a law. Rubric: both terms
with their roles.

## Q4

Strong: the gap is KL(q(z)||p(z|x)) = log p(x) - ELBO. It is zero
exactly when q equals the true posterior. Red flag: claiming a
large ELBO means a small gap. Rubric: the identity and the zero
condition.

## Q5

Strong: VI minimizes reverse KL, KL(q||p). That is mode-seeking:
q sits on posterior modes and may drop minor ones. Forward KL is
mode-covering and is what maximum likelihood minimizes. Red flag:
reversing the two behaviors. Rubric: direction named with the
behavior.

## Q6

Strong: it buys low-variance gradients through sampling by writing
z = g(theta, eps) with fixed noise. It fails for discrete latents:
the map is not differentiable. Red flag: claiming it removes bias.
Rubric: the mechanism plus the discrete failure.

## L1

1. Toy: z in {A,B}, prior (0.6,0.4), P(R|A)=0.8, P(R|B)=0.3.
2. P(R) = 0.60. P(A|R) = 0.48/0.60 = 0.8.
3. log p(x) = log E_q[p(x,z)/q(z)] >= E_q[log p(x,z)/q(z)] by
   concavity of log.
4. Numpy: elbo = (q * (log_joint - log q)).sum(). Check abs(elbo +
   kl - logp) < 1e-12.
5. Exact: O(K) per point, fine for K = 2, dead at K = 10^6 or
   continuous z. At n = 10,000 amortize: one encoder, O(1) per new
   point. Marginalize exactly only when K is small and exactness
   matters.
Red flags: skipping the renormalization in step 2. Rubric: all five
rungs with the verification.

## L2

1. z = g(theta, eps), eps ~ fixed law. d/dtheta E_q[f] =
   E_eps[df/dg dg/dtheta].
2. E = 1 + 4 = 5. dE/dmu = 2.
3. Score function: E[f(z) d log q/dtheta]. The d log q factor has
   high variance. Reparam fixes the sampling law, so only the
   deterministic map moves.
4. std ~ 1/sqrt(K). From 0.1001 at K = 1600, target 0.05 needs K
   near 6400.
5. Root cause class: severed autograd graph (detach on the sampling
   path). Test: compare against closed form 2 x mu and assert
   nonzero.
Red flags: confusing variance with bias. Rubric: the equation, the
number, the law, the debug.

## A1

Strong: KL = E_q[log q - log p(z|x)] = E_q[log q - log p(x,z)] +
log p(x) = -ELBO + log p(x). Rearranged. Support condition: q(z) >
0 implies p(x,z) > 0, else the log ratio is infinite. Red flag:
dropping the support condition. Rubric: the substitution chain and
the condition.

## A2

Strong: Monte Carlo the expectation: sample z ~ q, average log
p(x,z) - log q(z). Unbiased for the ELBO. Variance O(1/K) in the
sample count. Cost O(K) per step, independent of the 10^6 states.
Red flag: proposing to sum all states "once per epoch". Rubric:
unbiased named, cost stated.

## D1

Strong: a Bernoulli family has one degree of freedom, and the true
posterior (0.8, 0.2) is in it, so the ceiling is not the family.
The reported ceiling -0.62 sits below the ELBO at q = prior
(-0.6155), which means the optimizer never moved t off its init:
the bug is a broken gradient or a learning rate of zero on t.
Check: print t over steps. If t is frozen, fix the optimizer wiring.
If t moves but the ELBO stalls, check the log computations for a
detached graph. Red flag: blaming the Bernoulli family. Rubric:
the init-freeze diagnosis with the check.

## T1

Strong: mean-field on a 50-D correlated posterior leaves a large
irreducible KL gap (the 2-D toy already costs 0.8304 nats) and
reports wrong uncertainties. Cheapest upgrade: low-rank plus
diagonal covariance, O(d x r) params, captures the top correlation
directions. Red flag: adding layers to the encoder instead of
fixing the family. Rubric: the gap named and the upgrade costed.

## T2

Strong: per-point VI at 2 s/point cannot meet 1 s/point. Switch to
amortized inference: one encoder forward pass per point, O(1).
Sacrifice: the amortization gap. New points unlike training get
worse posteriors. Mitigation: monitor the encoder ELBO on live
points and fall back to per-point refinement when it drops.
Red flag: proposing a faster optimizer as the fix. Rubric: the
pipeline change and the named sacrifice.

## R1

Strong: reason 1: a higher ELBO can come from a better q with an
unchanged or worse p(x). The bound tightened, the model did not.
Reason 2: different q families make ELBOs incomparable. The richer
family wins the bound comparison for free. Settling experiment:
fix one q family, or better, compare exact or held-out likelihoods
with matched inference budgets, plus sample-quality checks the
likelihood cannot see. Red flag: proposing more training as the
settle. Rubric: two valid reasons plus an experiment with a
decision rule.
