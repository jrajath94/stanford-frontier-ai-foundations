# Lab keys: U01

Unit: math-genmodels-U01. Date: 2026-10-06. Baseline: October 6, 2026.

## E1

audit_law passes the toy law only. Sum 1.4 fails the sum check.
Negative entry fails the non-negativity check. Raw counts fail the
sum check. Lesson: three distinct failure modes, one checker.

## E2

support_of returns ['R','G','B']. The 50 seed-2 samples pass. The
injected "Y" raises. Lesson: check the sampler output, not the
sampler code.

## E3

Expected counts: (60000, 40000, 20000). A correct run lands within a
few hundred of each. Chi-square near 2 on 2 df. Values above 9.21
indicate a sampler bug. Seed 3 gives statistic about 1.1 (run to
confirm on your machine. The bound is what matters).

## E4

Expected pattern: MAP wins at n=3 and n=12, the gap closes by n=48,
MLE wins or ties at n=192. Typical crossover between n=48 and
n=192. Exact crossover moves with seed. Report yours with the seed.

## E5

H(p_hat) = 1.0114. H(uniform3) = 1.0986. KL(p_hat||uniform3) =
0.0872. KL(uniform3||p_hat) = 0.0959. Identity holds to 1e-12.
With a zero-count fourth outcome: KL(p4||uniform4) = 0.143 nats
(finite). KL(uniform4||p4) = infinity (log of 1/0). The direction
that integrates over q explodes when p has the zero.

## E6

The three curves coincide exactly because the components are
identical: w*g + (1-w)*g = g for every w. Lab-report sentence:
"When mixture components coincide, the weight parameter is not
identified: many parameter values give the same density, so fit the
density, not the label."
