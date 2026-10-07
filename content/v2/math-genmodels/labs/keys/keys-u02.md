# Lab keys: U02

Unit: math-genmodels-U02. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

marginal returns 0.60 for R and 0.40 for G. The broken variant that
sums the likelihoods without the prior returns 1.10 for R, which is
not a probability. Lesson: the prior weights the paths. Unweighted
paths do not form a law.

## E2

The ELBO curve peaks at q_A = 0.80, the true posterior. ELBO + KL
equals -0.5108 at every grid point within 1e-9. The peak value is
-0.5108. Lesson: the best q in the family is the posterior, and the
identity holds pointwise, not just on average.

## E3

Joint form at q = (0.7, 0.3): -0.5390. Recon-minus-KL form:
-0.5174 - 0.0216 = -0.5390. At q = posterior: both give -0.5108
and the KL gap is 0. Lesson: two algebra paths, one number. The
posterior choice is the tight bound.

## E4

Reparam: mean about 1.98, std about 0.12. Score-function: mean about
1.92, std about 0.25. Both center near the true 2.0. Reparam std at
K = 100, 400, 1600: about 0.40, 0.20, 0.10. Ratios near 2 per
quadrupling confirm 1/sqrt(K). Seed 7 gives 0.3956, 0.2041, 0.1001.
Lesson: unbiased is not enough. Variance decides the usable batch
size.

## E5

Per-point params: 40, 400, 4000, 40000. Amortized: 420 always.
Crossover: amortized wins on count for n > 105. Under the 1 KB cap:
max hidden size 19, amortized count 251, new crossover n >= 63.
Lesson: sharing wins at scale, but the cap can force a weaker
encoder and a larger amortization gap.

## E6

Mean-field gap: 0.8304 nats. Full-covariance fit: 0. The lab-report
sentence: "A diagonal variational family leaves an irreducible
0.8304 nat gap on a correlated posterior. Only a richer family can
close it, and no optimizer can close it for the poor family."
