# Answer keys: U01

Unit: math-genmodels-U01. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: probability assigns numbers in [0,1] to discrete outcomes and
sums to 1. Density assigns non-negative values per unit of x to a
continuum and integrates to 1. Red flag: "density is probability".
Rubric: both normalization rules stated.

## B2

Strong: support is {x : p(x) > 0}. Toy support: {R, G, B}. N(0,1)
support: the whole real line. Red flag: confusing support with the
outcome space. Rubric: set notation plus both examples.

## B3

Strong: Z = 6+4+2 = 12. Law: (0.5, 0.333, 0.167). Red flag: forgetting
to divide. Rubric: Z and the normalized vector.

## B4

Strong: p(color|L): column L = (2,3,1)/6 = (0.333, 0.5, 0.167).
Red flag: dividing by the row sum. Rubric: correct normalizer (6)
and all three numbers.

## B5

Strong: likelihood is p(data | parameters) with data fixed. Log
likelihood under uniform: 12 ln(1/3) = -13.183 nats. Red flag:
maximizing over data instead of parameters. Rubric: definition plus
the number.

## B6

Strong: p_emp puts mass 1/n on each observed sample. Mass on an
unseen outcome is 0. Red flag: claiming it generalizes. Rubric:
formula plus the zero.

## L1 ladder

1. Define: H(p) = -sum p log p, expected surprise in nats.
2. Compute: -(0.5 ln 0.5 + 2*0.25 ln 0.25) = 0.347 + 0.693 = 1.040 nats.
3. Uniform maximizes because -log is convex and symmetry forces the
   maximum at equality. Jensen gives H <= ln K.
4. H(p,q) = H(p) + KL(p||q). Minimizing cross-entropy over q with p
   fixed equals minimizing KL.
5. Code: R -> 0, G -> 10, B -> 11. New law on this rung:
   rungs 1-4 use (0.5, 0.25, 0.25). This rung uses the B3 toy
   law (0.5, 0.333, 0.167). Expected length =
   0.5*1 + 0.333*2 + 0.167*2 = 1.5 bits, above the toy law's
   H = 1.459 bits.
Red flags: log base confusion. Claiming the code beats entropy.
Rubric: each rung correct. The code length computed, not asserted.

## L2 ladder

1. Bias: E[theta_hat] - theta. Variance: E[(theta_hat - E)^2].
2. MLE from 1 red: (1, 0, 0). Certain about red, blind to the rest.
3. Failure: zero mass on unseen outcomes. KL to any full-support
   truth is infinite.
4. MAP with Dirichlet prior alpha=2: posterior counts are
   (1+1, 0+1, 0+1) = (2, 1, 1) over total 1+3 = 4, giving
   (0.5, 0.25, 0.25). The prior adds one fake count per outcome.
5. Use MLE when n is large enough that prior washout is complete.
   rule of thumb: n >> K * alpha.
Red flags: calling MAP unbiased. Rubric: the MAP arithmetic and the
decision rule.

## A1

Strong: KL = E_p[-log(q/p)]. -log is convex. Jensen:
E[-log(q/p)] >= -log E[q/p] = -log 1 = 0. Equality exactly when q/p
is constant a.s., i.e. p = q. Red flag: applying Jensen in the wrong
direction. Rubric: convex function named, equality condition stated.

## A2

Strong: maximize sum c_i log theta_i + lambda (1 - sum theta_i).
Derivative: c_i/theta_i - lambda = 0, so theta_i = c_i/lambda. Sum
constraint gives lambda = n. Hence theta_i = c_i/n. On the toy:
(6,4,2)/12. Red flag: dropping the constraint. Rubric: Lagrange
setup and the final substitution.

## D1

Strong: the bug is searchsorted over edges including 0.0: u in
[0, 0.5) maps to index 1, shifting every bin up by one, and u near 1
maps to index 4, out of range. Fix: searchsorted(edges[1:], u).
Histogram test: 120000 draws, chi-square against p_hat, p > 0.01.
Red flag: "fixing" by clipping the output. Rubric: root cause plus a
statistical test, not eyeballing.

## T1

Strong: MLE over 4 colors from the same 12 draws: (0.5, 0.333,
0.167, 0). Support claim: the fourth color is impossible, which is
absurd from zero observations. KL to uniform over 4: infinite,
because q has mass where p_hat has none... direction check:
KL(p_hat||uniform4) is finite (0.5 ln(0.5/0.25) + ...) = 0.143 nats.
KL(uniform4||p_hat) is infinite. What breaks: the zero. Remedy:
smoothing. Red flag: reporting finite KL in both directions.
Rubric: both directions computed, the break named.

## T2

Strong: keep running counts c and total n. On new draw x: c[x] += 1,
n += 1, p_emp = c/n on demand. O(K) memory, O(1) per draw. Red
flag: storing all draws. Rubric: update rule plus complexity.

## R1

Strong: counterexample 1: the empirical distribution gets the best
possible training likelihood and memorizes (C08). Counterexample 2:
a model with support strictly inside the data support can still win
on training points it covers while assigning zero mass to real
held-out points (C02, C11). Settling experiment: fixed train/test
split, score held-out log-likelihood, require support audit on test
points, and add a sample-quality check the likelihood cannot see
(U08). Red flag: proposing more training data as the fix. Rubric:
two valid counterexamples plus an experiment with a decision rule.
