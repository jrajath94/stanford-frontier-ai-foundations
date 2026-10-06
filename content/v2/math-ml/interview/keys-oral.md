# Keys and rubrics, oral defenses

Date: 2026-10-06. Each defense: 15 minutes, 20 points. Pass at
14. The examiner scores live. No retroactive credit.

## General rubric (applies to all)

- Correct claim with live recomputation: full points.
- Correct claim quoted from memory with no derivation: half.
- Correct claim with a wrong derivation: quarter.
- Confident wrong claim: zero for the item, flagged in the
  mastery ledger as a confident mistake.
- "I do not know" with a correct statement of what would
  settle it: half.

## O-U01 (20)

Plot honest (axes labeled, units, zero baseline or defended):
4. Assertion named: 3. Breakage ranking (conclusion first, then
mean, then plot): 4. Seed-change survival (qualitative claims
survive, exact numbers do not): 4. Unlicensed claims named
(causation, extrapolation): 5.

## O-U02 (20)

Keep/kill demonstrated (column space vs nullspace): 4.
Rank-nullity stated: 3. Pre-image refused (infinite or none):
4. Rank-1 error = 0 (sigma_2 = 0): 4. kappa bound: relative
error <= 1e6 * eps ~ 2e-10 per unit roundoff, so 6 digits are
not safe: 5.

## O-U03 (20)

Critical points: x = 0 (saddle), x = +/-sqrt(1.5) (minima):
5. GD stalls at the saddle (zero gradient). Fix with a nudge
or momentum: 4. Newton at x = 1: f'' = 12 - 6 = 6 > 0,
converges locally: 4. KKT: x* = 1, multiplier >= 0,
complementarity holds: 4. FD error 1e-3: truncation O(h^2)
predicts ~1e-6 for smooth f. 1e-3 suggests a bug or a kink:
3.

## O-U04 (20)

Three estimates: MLE 0.75, Laplace 0.6667, MAP mode with
Beta(2,2): (3+2-1)/(4+2+2-2) = 4/6 = 0.6667, posterior mean
5/8 = 0.625: 6. Likelihood written, max at 0.75: 4. Bias
comparison: MLE unbiased, Laplace and MAP biased toward the
prior: 3. KL direction D_KL(p_data || p_model) for fitting:
3. Bandwidth doubling smooths: estimate falls toward the
global mean density: 4.

## O-U05 (20)

Two risks written: 4. Three leaks with tests (target in
features, time travel, preprocessing on all data): 6. Decline
labels: gap closed, error near noise floor, more labels buy
~nothing. Run the ablation (richer model) instead: 5. Ridge as
penalty and as Gaussian prior, both giving 0.8125: 5.

## O-U06 (20)

w_hat [0.3, 0.8] by hand: 5. Residual-sum proof: the intercept
column gives sum residuals = 0 as the first normal equation:
4. Logistic step: grad -0.25, w = 0.25: 4. Refuse to ship:
residual 4e-11 with cond 3.2e6 means cancellation garbage. 
fix gamma: 4. Duality: primal 0.25 = dual 0.25 via sum
alpha_i y_i = 0: 3.

## O-U09 (20)

Clusters found, J = 0.1333, local optimality argued (no single
reassignment lowers J): 5. Bad start path predicted (145.17 ->
9.04 -> 0.1333): 4. Scaling damage predicted (assignments
flip). Fix: standardize: 4. k = 2 loses on held-out (second
direction is noise): 4. Degeneracy named, pointer to lesson
04c C11 (not re-derived): 3.
