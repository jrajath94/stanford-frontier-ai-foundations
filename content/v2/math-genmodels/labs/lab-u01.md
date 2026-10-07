# Lab U01: probability objects in code

Unit: math-genmodels-U01. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u01.md. Do not open the keys
until the code runs.

## E1: normalization audit

Write audit_law(p) that returns True only if every entry is >= 0 and
the sum is 1 within 1e-9. Test it on: the toy law, a law that sums to
1.4, a law with a negative entry, and raw counts. Report which fail
and why.

## E2: support checker

Write support_of(p, outcomes) returning the support list. Write
check_samples(p, outcomes, samples) that raises on any out-of-support
sample. Generate 50 samples from the toy law with seed 2 and run the
checker. Then inject one "Y" and confirm the raise.

## E3: sampler histogram test

Implement the inverse-CDF sampler from C06. Draw 120000 samples with
seed 3. Bin them. Run a chi-square test against p_hat by hand (no
scipy): statistic = sum (obs - exp)^2 / exp. Report the statistic and
the degrees of freedom. Expected: near 2, well below the 1% critical
value 9.21.

## E4: MLE versus MAP shootout

True law (0.5, 0.3, 0.2). For n in (3, 12, 48, 192): draw 300
datasets with seed 4, compute MLE and MAP (uniform Dirichlet prior,
alpha = 2) per dataset, record mean squared error to truth. Plot MSE
versus n for both. Report the crossover n where MLE starts winning.

## E5: entropy and KL by hand

On paper, then in code: H(p_hat), H(uniform3), KL(p_hat||uniform3),
KL(uniform3||p_hat). Confirm H(p,q) = H(p) + KL(p||q) numerically.
Then add a fourth outcome with zero count and recompute both KL
directions against uniform4. Explain which direction explodes and why.

## E6: identifiability demo

Fit is overkill here. Instead: define mixture density
0.5*N(0,1) + 0.5*N(0,1) and evaluate it on a grid for weight vectors
(0.5,0.5), (0.2,0.8), (0.9,0.1). Show the three curves coincide to
1e-12. Write the one-sentence lesson for a lab report.
