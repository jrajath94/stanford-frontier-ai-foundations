# Lab U02: variational inference on the two-bag toy

Unit: math-genmodels-U02. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u02.md. Do not open the keys
until the code runs.

## E1: marginalization audit

Write marginal(prior, like) returning the marginal. Test it on the
toy: P(R) must be 0.60, P(G) must be 0.40. Then write a broken
variant that forgets to weight by the prior and report what it
returns. Explain in one sentence why the broken number is wrong.

## E2: posterior grid

Grid q_A over 0.05 to 0.95 in steps of 0.05. At each grid point
compute the ELBO at x = R and KL(q||posterior). Plot ELBO versus
q_A. Verify ELBO + KL equals log p(R) = -0.5108 at every grid point
within 1e-9. Report the q_A that maximizes the ELBO.

## E3: ELBO identity check

Implement the ELBO both ways: the joint form E_q[log p(x,z) - log
q(z)] and the reconstruction-minus-KL form. Evaluate both at q =
(0.7, 0.3) and at q = posterior. Confirm the two forms agree and
that the posterior choice closes the gap to zero.

## E4: reparam versus score-function shootout

On the z^2 toy (mu = 1, sigma = 2): implement both gradient
estimators for dE/dmu. Run 50 repeats of 1000 samples with seed 7.
Report mean and std of each. Then repeat at K = 100, 400, 1600 for
the reparam estimator and verify the 1/sqrt(K) law from the std
ratios.

## E5: amortization crossover

Compute per-point versus amortized parameter counts for the C10
encoder at n in (10, 100, 1000, 10000). Find the crossover n where
amortized wins on count. Then impose the 1 KB memory cap from T1:
recompute the max hidden size and the new crossover. Report both.

## E6: mean-field gap

Build the correlated target Sigma = [[1, 0.9],[0.9, 1]]. Compute
the KL gap for the mean-field family N(0, I). Then fit the best
full-covariance Gaussian (it matches Sigma) and confirm the gap
drops to 0. Write the one-sentence lesson for a lab report.
