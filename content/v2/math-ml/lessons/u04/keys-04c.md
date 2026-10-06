# Keys, third source block (04c)

Date: 2026-10-06. Test-mode: solve closed-book first. All numbers
computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## E01

Predict argmax_k p(k|x). Assumption: the true priors and class
densities are known. Without them the rule is a benchmark, not a
method.

## E02

Posterior ratio: predict 1 when 0.5 N(x, 2) > 0.5 N(x, 0).
The densities cross where (x-2)^2 = x^2, i.e. x = 1. Error =
0.5 P(N(0,1) > 1) + 0.5 P(N(2,1) < 1) = 1 - Phi(1) = 0.1587.

## E03

Error = 0.5 (1 - Phi(1.5)) + 0.5 Phi(1 - 1.5) = 0.5 * 0.0668 +
0.5 * 0.3085 = 0.1877. Excess over Bayes: 0.1877 - 0.1587 =
0.0290.

## E04

No. The optimal rule predicts 1 when
p(1|x)/p(0|x) > c_FP/c_FN = 1/10, i.e. at posterior odds above
0.1, not 1. What changes: the threshold moves toward predicting
the costly class more often. The form stays "compare posterior
odds to a cost ratio".

## E05

t = 0.5. It is the only threshold with FPR = 0 <= 0.1, and it
achieves TPR = 2/3 = 0.6667.

## E06

Points in FPR order: (0,0), (0,1/3), (0,2/3), (0.5,1), (1,1).
Trapezoids: the vertical segments at FPR = 0 contribute 0. From
(0, 2/3) to (0.5, 1): width 0.5, mean height (2/3+1)/2 = 5/6,
area = 0.4167. From (0.5, 1) to (1, 1): width 0.5, height 1,
area = 0.5. Total = 0.9167.

## E07

The threshold adapts to the training scores' noise: with 2
negatives, FPR = 0 means "no false alarm yet", not "false alarms
impossible". On new negatives the realized FPR jumps. Small
samples make every ROC point noisy.

## E08

Cost = c_FP * FPR + c_FN * FNR. At t = 0.3: 0.5 c_FP. At t = 0.5:
0.3333 c_FN. t = 0.3 wins when 0.5 c_FP < 0.3333 c_FN, i.e. when a
miss costs more than 1.5x a false alarm. Example: c_FP = 1,
c_FN = 10.

## E09

L = prod_i [0.5 N(x_i, mu_1) + 0.5 N(x_i, mu_2)]. The log is
sum_i log(sum_z ...). The sum inside the log blocks term-by-term
differentiation: d/dmu of log(a + b) does not split.

## E10

N(0.2, 1) = 0.2897, N(0.2, 4) ~ 0.0003. r = 0.5*0.2897 /
(0.5*0.2897 + 0.5*0.0003) = 0.9990. The point belongs to
component 1 with ~99.9% responsibility.

## E11

Jensen's inequality (concavity of log): it builds the lower bound
that the M-step maximizes. Each EM iteration climbs a bound
touching the likelihood at the current point, so the likelihood
cannot decrease.

## E12

EM guarantees monotone ascent to a local maximum, not the global
one. Start B's basin of attraction holds a poor local maximum at
-14.1109. Monotone is a promise about direction, not destination.

## E13

mu_1 = sum_i r_i x_i / sum_i r_i. The responsibility-weighted mean
of the points.

## E14

Multiple restarts from spread-out starts, keep the run with the
best final likelihood. Optionally start from k-means. Never trust
a single EM run on a lumpy surface.

## E15

The term for x_1 = 0.2 is 0.5 / sqrt(2 pi sigma_1^2) =
0.5 / sqrt(2 pi sigma_1^2), which -> infinity as sigma_1^2 -> 0.
The other points keep finite density under component 2. The total
likelihood is unbounded above: no finite maximizer exists.

## E16

MAP = (7 + 2 - 1)/(10 + 2 + 2 - 2) = 8/12 = 0.6667. MLE = 0.7. The
prior pulls the estimate 0.0333 toward 0.5.

## E17

loglik([0, 5]) = -9.8125, loglik([5, 0]) = -9.8125, equal to all
printed digits. This proves the parameter vector is not
identifiable: two different parameters give the same distribution,
so the data cannot separate them.

## E18

Terms: K(2) = 0.0540, K(0) = 0.3989, K(-2) = 0.0540, K(-5) ~ 0.
Sum = 0.5069. p_hat = 0.5069/(4 * 0.5) = 0.2535.

## Ladders

L01. (1) argmax_k p(k|x) for 0-1 loss, needs true densities.
(2) 0.1587 at boundary x = 1. (3) The conditional risk at x is
1 - p(k|x). Minimize pointwise, then integrate. (4) It cannot beat the
true Bayes error: the estimated densities differ from the truth,
so the measured "0.1587" used a lucky test draw or the estimates
happen to be good. Recompute on fresh data. The number will move.
(5) Predict 1 when p(1|x) c_FN > p(0|x) c_FP: compare posterior
odds to the cost ratio.
L02. (1) TPR = TP/P, FPR = FP/N, ROC plots TPR vs FPR over
thresholds, AUC is the area. (2) The SB20 table. Minmax and NP
both pick t = 0.5 here. (3) Among thresholds with FPR <= 0.1,
t = 0.5 is feasible and gives the highest TPR, 0.6667. (4) The
validation negatives did not represent deployment (shift), or
the threshold was tuned on the validation scores themselves.
(5) Fix a daily false-alarm budget, convert to an FPR cap, pick
the threshold on held-out scores, and re-check the FPR weekly.
L03. (1) E: responsibilities from current params. M: weighted MLE
updates. (2) mu [1,4] -> [0.0026, 5.0635], likelihood -13.0089 ->
-3.4450. (3) log sum q p/q >= sum q log(p/q). Here q = posterior, which makes
the bound tight. M maximizes the bound. Chain the inequalities.
(4) A sign or normalization bug in the E-step, or the M-step
maximizing the wrong objective: the monotone proof assumes the
exact posterior and the exact M maximizer. (5) Generalized EM
(partial M-steps), gradient ascent on the incomplete likelihood,
or variational bounds.
L04. (1) r_i = p(z_i = 1 | x_i), the soft assignment. (2) Start A
-> -3.3320, start B -> -14.1109, both monotone. (3) The surface
has many local maxima. Restarts sample basins and the best final
likelihood is the practical answer. (4) Degeneracy: a component
collapsed onto one point. Fix with a variance floor or a prior.
(5) Mini-batch EM or online EM, k-means++ start, few restarts on
subsamples then refine the best on full data. Budget the O(n K)
per sweep.
L05. (1) The map theta -> p(x | theta) is one-to-one. (2) The swap
test: -9.8125 both ways. (3) Posterior proportional to
theta^(k+alpha-1) (1-theta)^(n-k+beta-1). Its mode is at
(k+alpha-1)/(n+alpha+beta-2). (4) Label switching: the chain
visits permuted modes and the mean lands mid-way. Diagnose by
plotting the trace of mu_1: it jumps between 0 and 5.
(5) Impose an ordering constraint (mu_1 < mu_2) or report the
partition, not the labels. Predictions are identified. Labels are
not.
