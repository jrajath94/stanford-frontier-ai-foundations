# Third source block, Lec 15-16, Lec 20-27, Tutorials 3-9

Unit: math-ml-U04 (Bayes, identifiability, calibration) and
math-ml-U09 latent-model leaves (mixture model, EM, Jensen, latent
posterior, initialization, degeneracy). Date: 2026-10-06.

## Source mapping

This lesson maps the third source block of the playlist
(SRC-04, titles enumerated 2026-10-06). Sections SB17-SB26 cover:
Lec 15 Risk Minimization Framework. Lec 16 Bayes Classifier.
Tutorial 3 Risk Minimization Framework. Lec 20 Latent Variable
Models. Lec 21 MLE for Latent Variable Models. Lec 22 Expectation
Maximization Algorithm. Tutorial 4 Minmax Classifier. Tutorial 5
Neyman Pearson Classifier. Tutorial 6 Example of NP Classifier, ROC
Curve. Lec 23 Convergence of EM. Lec 24 EM for GMMs. Lec 25 MAP
Estimate. Lec 26 Parzen Window. Lec 27 Nearest Neighbor Classifier.
Tutorial 8 Computation of EM for GMMs. Tutorial 9 MAP Estimate.

Inspection boundary: titles only. No transcript, slide, or notebook
was inspected (source_gaps.md G2, G6). The teaching below is
original, built from the topic names plus standard mathematical
results. It does not claim to reproduce the instructor's spoken
content, examples, or numbers. Source attribution for the leaf
concepts is PENDING. Tutorials 7A/7B belong to the second block
(lesson-04b). Tutorial 10 belongs to a later block. All numbers are
computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## Scope and objectives

Scope: the risk minimization framework as a source topic, the Bayes
classifier and its optimality, a worked ERM example, minmax and
Neyman-Pearson decision rules with ROC, latent variable models and
the incomplete likelihood, the EM algorithm with its convergence
argument, EM for Gaussian mixtures with initialization effects, MAP
estimation with conjugate priors, identifiability and label
switching, Parzen window density estimation, and the nearest
neighbor classifier.

Objectives: after this block the learner can write the Bayes
decision rule and compute its error on a Gaussian toy, run two EM
iterations by hand on six points, state the Jensen step that makes
EM monotone, show that swapping mixture labels leaves the
likelihood unchanged, compute a Parzen estimate, and read a ROC
table to pick a Neyman-Pearson threshold.

Dependencies: lessons/u05/lesson-05 (risk, ERM). U04 likelihood,
MLE, MAP, expectation. U02 Gaussians via U04b. Prerequisites R35-R44.

## How to read this block

Each section names its playlist title first, then follows the lesson
chain: question, toy, rule, computed example, code, checks, costs,
alternatives, failure case, shells. Assessment items close the
block. Keys in lessons/u04/keys-04c.md.

---

## SB17, Lec 15: the risk minimization framework

Playlist title: "Lec 15 Risk Minimization Framework".

Motivating question: what does the framework minimize, over what set,
and against what truth?

Start from zero. The framework names three objects: a hypothesis
class H (the rules you allow), a loss l (the price list), and the
risk R(h) (the expected price). ERM picks the h in H with the
smallest R_hat on the sample. The framework does not promise the
pick is good. It promises the question is well-posed: minimize a
measured price over a named set.

Toy. Six points: x = [0, 1, 2, 3, 4, 5], y = [0, 0, 0, 0, 1, 1].
H = {threshold rules x >= t}. h1: t = 2.5. Predictions
[0, 0, 0, 1, 1, 1]. One error at x = 3. R_hat = 1/6 = 0.1667.
h2: t = 3.5. Predictions [0, 0, 0, 0, 1, 1]. Zero errors.
R_hat = 0.0. ERM picks h2. Computed 2026-10-06.

Mental model. The framework is a courtroom: H is who may speak, l
is the law, R_hat is the evidence, and ERM is the verdict. Change
the law (loss) and the verdict changes (U05 C02). Enlarge who may
speak (capacity) and the verdict fits the evidence too well (U05
C09). The framework organizes the tradeoffs. It does not resolve
them.

Variables. H, l, R(h), R_hat(h). Same symbols as U05 C01.
Assumptions: H is fixed before seeing data. l is fixed before
seeing data. The sample is IID.

Why it exists. Without the framework, "train a model" is a slogan.
With it, every choice is named and can be argued about: the class,
the loss, the data.

Figure: the two-row table (h1 0.1667, h2 0.0, ERM picks h2) is the
figure. One claim: ERM is a well-posed selection rule. Source:
original toy.

Code.

```python
x = [0, 1, 2, 3, 4, 5]
y = [0, 0, 0, 0, 1, 1]
for t in (2.5, 3.5):
    pred = [1 if v >= t else 0 for v in x]
    print(t, sum(p != q for p, q in zip(pred, y)) / 6)
# 2.5 0.16666666666666666
# 3.5 0.0
```

Check. h1 errs only at x = 3. 1/6 = 0.1667. h2 errs nowhere.
Expected output: 0.1667, 0.0.

Costs. ERM over a finite H costs |H| times n. Over a continuous
class it needs an optimizer (U03). Alternatives: structural risk
minimization adds a capacity penalty to R_hat (this is C05's tax
with a theory name). Selection boundary: ERM when H is small or
n is large. Penalized ERM when H is rich and n is small.

Failure case. H too rich: the memorizer from U05 C01 has R_hat = 0
and R = 0.5. The framework's verdict is correct and useless.
Counterexample: the toy's h2 is ERM-optimal and still only as good
as the sample is representative.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (the verdict
rule), 5 (hand 1/6 versus code), 7 (memorizer counterexample).

---

## SB18, Lec 16: the Bayes classifier

Playlist title: "Lec 16 Bayes Classifier". Leaf: math-ml-U04-C03
(Bayes).

Motivating question: among all possible rules, which one has the
smallest population risk?

Start from zero. Two classes, equal priors 0.5 and 0.5.
p(x|0) = N(0, 1), p(x|1) = N(2, 1). The Bayes rule predicts the
class with the larger posterior: predict 1 when
p(x|1) > p(x|0), which for these symmetric Gaussians is x > 1.
Its error is P(wrong) = 0.5 * P(x > 1 | class 0) +
0.5 * P(x < 1 | class 1) = 1 - Phi(1) = 0.1587. No rule beats it.
That is the optimality claim, and the number is its price.

Mental model. The Bayes classifier is the ceiling of the
courtroom: it uses the true distributions, which you never have.
Every real classifier is measured by its distance to 0.1587 here.
ERM tries to reach the ceiling from data. The gap between your
rule and the Bayes rule is the learnable part of your error.
Shell 2.

Variables. pi_k priors, p(x|k) class densities, posterior
p(k|x) proportional to pi_k p(x|k). Decision: argmax_k p(k|x).
Assumptions: the true densities and priors are known. This never
holds in practice. The classifier is a benchmark, not a product.

Why it exists. "Best possible" needs a definition before "good
enough" means anything. The Bayes error is the floor no data
amount crosses.

Computed example, same objects. Computed 2026-10-06:

| rule | error |
|---|---|
| Bayes (threshold x = 1) | 0.1587 |
| threshold x = 1.5 | 0.1877 |

The 1.5 threshold pays 0.0290 extra for being off-center. The
Bayes rule is the floor.

Derivation. For 0-1 loss, the conditional risk of predicting k at
x is 1 - p(k|x). Minimizing it pointwise picks argmax_k p(k|x).
Integrate over x: no other rule has smaller total risk. The key
step is pointwise: the global problem splits per x because the
loss adds over points.

Figure: the table is the figure. One claim: 0.1587 is the floor.
Source: original toy.

Code.

```python
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
print(1 - Phi(1.0))  # 0.15865525393145707
```

Check. Phi(1) = 0.8413. 1 - 0.8413 = 0.1587. Expected output:
0.15865525393145707.

Costs. The rule costs O(K) posterior evaluations per point, once
the densities are known. Alternatives: plug-in classifiers estimate
the densities (LDA, QDA, naive Bayes) and inherit their estimation
error. Selection boundary: use the Bayes form when you trust your
density estimates. Otherwise the form is a target, not a method.

Failure case. Wrong densities, wrong rule: estimate the means
badly and the "Bayes" threshold moves off 1, paying more than
0.1587. Counterexample: with 0-1 loss the rule is optimal. With a
different loss (asymmetric costs) argmax posterior is the wrong
rule. The optimality is loss-specific.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (pointwise rule),
5 (0.1587 check), 7 (wrong-loss counterexample).

Assessment: E01-E04, L01.

---

## SB19, Tutorial 3: worked ERM example

Playlist title: "Tutorial 3 Risk Minimization Framework".

This section works the framework numerically, in the style of a
tutorial: one dataset, two candidate rules, every number shown.

Data: x = [0, 1, 2, 3, 4, 5], y = [0, 0, 0, 0, 1, 1]. Loss: 0-1.
Candidate A: threshold t = 2.5. Candidate B: threshold t = 3.5.

Step 1: predictions of A: [0, 0, 0, 1, 1, 1]. Compare with y:
positions 0,1,2,4,5 agree. Position 3 (x = 3, y = 0, pred 1)
disagrees. Errors: 1. R_hat(A) = 1/6 = 0.1667.
Step 2: predictions of B: [0, 0, 0, 0, 1, 1]. All agree.
R_hat(B) = 0.0.
Step 3: ERM picks B.
Step 4: honesty note. B's 0.0 is R_hat, not R. On a new point
x = 3.2 with true y = 0 (the classes overlap near 3), B predicts
1 and is wrong. The tutorial's verdict is about the sample.

The worked style is the point: every ERM claim should survive this
four-step treatment. Shells 1-5.

---

## SB20, Tutorials 4-6: minmax, Neyman-Pearson, and ROC

Playlist titles: "Tutorial 4 Minmax Classifier", "Tutorial 5 Neyman
Pearson Classifier", "Tutorial 6 Example of NP Classifier, ROC
Curve". Leaf: math-ml-U04-C12 (calibration: threshold choice).

Motivating question: the classifier outputs scores. Where do you
put the threshold when false alarms and misses cost differently?

Start from zero. Scores s = [0.1, 0.35, 0.4, 0.6, 0.8], labels
y = [0, 0, 1, 1, 1]. Threshold t turns scores into verdicts.

| t | TPR | FPR | FNR | max(FPR, FNR) |
|---|---|---|---|---|
| 0.3 | 1.0000 | 0.5000 | 0.0000 | 0.5000 |
| 0.5 | 0.6667 | 0.0000 | 0.3333 | 0.3333 |
| 0.7 | 0.3333 | 0.0000 | 0.6667 | 0.6667 |

Three decision rules, one table:

- Minmax: minimize the worst of the two error rates. t = 0.5 wins
  with 0.3333. It guards against the worst case without a cost
  model.
- Neyman-Pearson: fix FPR <= alpha, maximize TPR. With
  alpha = 0.1, t = 0.5 is the only candidate with FPR = 0, and it
  gives TPR = 0.6667. It is the constrained optimum: no rule with
  FPR <= 0.1 detects more.
- ROC: the points (0,0), (0, 1/3), (0, 2/3), (0.5, 1), (1, 1) trace
  the tradeoff curve. AUC = 0.9167 by the trapezoid rule. AUC is
  the probability that a random positive outscores a random
  negative.

Mental model. The threshold is a dial between two errors. Minmax
centers the worst case. Neyman-Pearson obeys a false-alarm budget.
ROC draws the whole dial so you can pick with costs in hand.
Calibration (C12's home) is the honest choice of this dial on held-out
scores, not on the training scores that produced them. Shell 2.

Variables. TPR = TP/P, FPR = FP/N, FNR = 1 - TPR. AUC dimensionless
in [0, 1]. Assumptions: scores are fixed. Only the threshold moves.
The NP guarantee needs the threshold chosen on data independent of
the score training.

Why it exists. Real deployments have asymmetric costs: a missed
tumor versus a false alarm, a blocked legit user versus a fraud
loss. Accuracy hides the tradeoff. The ROC table shows it.

Computed example, same objects. Computed 2026-10-06. AUC 0.9167.

Code.

```python
s = [0.1, 0.35, 0.4, 0.6, 0.8]
y = [0, 0, 1, 1, 1]
for t in (0.3, 0.5, 0.7):
    pr = [1 if v >= t else 0 for v in s]
    tp = sum(1 for a, b in zip(pr, y) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pr, y) if a == 1 and b == 0)
    print(t, tp / 3, fp / 2)
# 0.3 1.0 0.5
# 0.5 0.6666666666666666 0.0
# 0.7 0.3333333333333333 0.0
```

Check. At t = 0.5, scores >= 0.5 are [0.6, 0.8], both positives:
TP = 2, FP = 0. TPR = 2/3, FPR = 0. Expected output matches.

Costs. One sort of scores: O(n log n). The table is O(n) per
threshold. Alternatives: pick the threshold by expected cost
c_FP * FPR + c_FN * FNR when costs are known. Selection boundary:
use NP when a false-alarm budget is the contract. Use minmax when
no cost model exists. Use expected cost when costs are priced.

Failure case. Tune the threshold on the training scores: the ROC
looks better than it is, and the deployed FPR breaks the budget.
Counterexample: the table's FPR = 0 at t = 0.5 is measured on 2
negatives. With 2 negatives, "FPR = 0" means "no false alarm yet",
not "false alarms impossible". Small-sample ROC points are noisy.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (the three rules),
5 (table versus code), 7 (train-threshold counterexample).

Assessment: E05-E08, L02.

---

## SB21, Lec 20-21: latent variable models

Playlist titles: "Lec 20 Latent Variable Models", "Lec 21 MLE for
Latent Variable Models". Leaf: math-ml-U09-C05 (mixture model).

Motivating question: what if each point came from one of several
hidden sources, and you never saw which?

Start from zero. Six points: [0.2, -0.3, 0.1, 5.1, 4.8, 5.3]. Two
clusters are visible. Model: with probability 0.5 the point comes
from N(mu_1), else from N(mu_2). Both have variance 1.
N(x, m) below is the unit-variance Gaussian density at x. The cluster label z is
latent: unobserved. The observed likelihood sums over the hidden
choice: p(x) = 0.5 N(x, mu_1) + 0.5 N(x, mu_2). At
mu = [0, 5] the incomplete log-likelihood is -9.8125. The log sits
outside the sum. That placement is the whole difficulty: no closed
form, no direct MLE.

Mental model. A latent variable is a missing column in your
spreadsheet. The complete data (x, z) would be easy: each point
has an owner, and each owner's parameters are sample means. The
incomplete data (x alone) forces a sum inside the log, and the sum
defeats differentiation. EM (SB22) works around it by guessing the
missing column softly. Shell 2.

Variables. z in {1, 2} latent label. pi = 0.5 mixing weight.
mu_1, mu_2 means. sigma^2 = 1 known here. Incomplete likelihood:
prod_i sum_z p(x_i, z). Complete likelihood: prod_i p(x_i, z_i).
Assumptions: the number of components (2) is fixed and known. The
component family (Gaussian) is fixed.

Why it exists. Real data is mixed: speakers in audio, topics in
text, cell types in biology. One Gaussian cannot describe two
clusters. The mixture can, at the price of the hidden label.

Computed example, same objects. Computed 2026-10-06. Incomplete
log-likelihood at mu = [0, 5]: -9.8125. The three left points each
contribute about log(0.5 * 0.39) and the three right points the
mirror.

Code.

```python
import numpy as np
from math import sqrt, pi, exp, log
xm = [0.2, -0.3, 0.1, 5.1, 4.8, 5.3]
def N(v, m): return exp(-0.5 * (v - m) ** 2) / sqrt(2 * pi)
ll = sum(log(0.5 * N(v, 0.0) + 0.5 * N(v, 5.0)) for v in xm)
print(ll)  # -9.81248395492605
```

Check. For v = 0.2: 0.5*N(0.2, 0) = 0.1955, 0.5*N(0.2, 5) ~ 0.
log(0.1955) = -1.632. Six such terms sum to -9.8125. Expected
output: -9.81248395492605.

Costs. One likelihood evaluation: O(n K) for K components.
Alternatives: hard assignment (k-means) replaces the soft sum with
the nearest center. It is faster and wrong in the overlap region.
Selection boundary: use the mixture likelihood when overlap
matters and you can afford O(n K). Use k-means for a fast first
partition.

Failure case. The likelihood has K! symmetric modes (SB24): the
labels are arbitrary. Worse, it is unbounded (SB23 degeneracy):
one component can collapse onto one point with variance -> 0 and
drive the likelihood to infinity. The MLE does not exist without
constraints. Counterexample: mu_1 = 0.2, sigma_1^2 -> 0 makes the
first point's density explode. Maximum likelihood without bounds
is ill-posed here.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (sum inside the
log), 5 (hand -9.8125 versus code), 7 (collapse counterexample).

Assessment: E09-E10, L03.

---

## SB22, Lec 22-23: the EM algorithm and its convergence

Playlist titles: "Lec 22 Expectation Maximization Algorithm",
"Lec 23 Convergence of EM". Leaves: math-ml-U09-C06 (EM),
math-ml-U09-C07 (Jensen).

Motivating question: the log sits outside the sum and blocks
differentiation. How do you climb anyway?

Start from zero. Same six points, start mu = [1, 4], sigma^2 = 1,
pi = 0.5. E-step: for each point compute the responsibility, the
soft guess of its owner: r_i = 0.5 N(x_i, 1) /
(0.5 N(x_i, 1) + 0.5 N(x_i, 4)). M-step: recompute each mean
as the responsibility-weighted average, and sigma^2 and pi the same
way. After one iteration: mu = [0.0026, 5.0635], sigma^2 = 0.0592,
pi = 0.4999. After two: mu = [0.0000, 5.0667], sigma^2 = 0.0444,
responsibilities [1, 1, 1, 0, 0, 0]. The clusters separated
themselves. The incomplete log-likelihood: -13.0089 at the start,
-3.4450 after one step, -3.3320 after twenty steps from this
start. It never decreases.

Mental model. EM is a two-dance: guess the missing column softly
(E), then fit as if the guess were data (M). Each M-step climbs a
lower bound that touches the true likelihood at the current point.
Climbing the bound cannot lower the truth. That is the whole
convergence argument, and Jensen's inequality is the step that
builds the bound. Shell 2.

Variables. r_i = p(z_i = 1 | x_i): responsibility of component 1
for point i. Q: the expected complete log-likelihood under the
current responsibilities. Assumptions: the E-step uses the exact
posterior (possible here because z is discrete and tiny). The M-step
has a closed form for Gaussians.

Why it exists. Latent models are everywhere and their likelihoods
are sums inside logs. EM turns the hard problem into easy problems
in alternation. It is the workhorse behind mixtures, HMMs, and
missing-data methods.

The Jensen step. log sum_z p(x, z) = log sum_z q(z) p(x, z)/q(z)
>= sum_z q(z) log(p(x, z)/q(z)), for any distribution q, because
log is concave. Choose q(z) = p(z | x) at the current parameters:
the bound touches the likelihood there. The M-step maximizes the
bound. So the likelihood at the new parameters is at least the
bound there, which is at least the bound at the old parameters,
which equals the old likelihood. Four links, monotone climb. Each
link is an inequality or an equality you can check.

Computed example, same objects. Computed 2026-10-06:

| step | mu_1 | mu_2 | sigma^2 | log-likelihood |
|---|---|---|---|---|
| init | 1.0000 | 4.0000 | 1.0000 | -13.0089 |
| iter 1 | 0.0026 | 5.0635 | 0.0592 | -3.4450 |
| iter 2 | 0.0000 | 5.0667 | 0.0444 | rising |
| iter 20 | 0.0000 | 5.0667 | small | -3.3320 |

Figure: the table is the figure. One claim: the likelihood never
decreases. Source: original toy, computed 2026-10-06.

Code (one EM step).

```python
import numpy as np
from math import sqrt, pi, exp, log
xm = np.array([0.2, -0.3, 0.1, 5.1, 4.8, 5.3])
mu = np.array([1.0, 4.0])
s2 = 1.0
pi = 0.5
def N(v, m, s2): return np.exp(-0.5 * (v - m) ** 2 / s2) / sqrt(2 * pi * s2)
g1 = pi * N(xm, mu[0], s2)
g2 = (1 - pi) * N(xm, mu[1], s2)
r = g1 / (g1 + g2)                       # E-step
mu = np.array([np.sum(r * xm) / r.sum(), np.sum((1 - r) * xm) / (1 - r).sum()])
print(mu)  # [0.0026 5.0635]
```

Check. The left three points sit near 0: their responsibilities
are ~1, so mu_1 becomes their mean 0.0026. The right three go to
mu_2 = 5.0635. Expected output: [0.0026 5.0635].

Costs. One iteration: O(n K). Total: iterations times that.
Alternatives: direct gradient ascent on the incomplete likelihood
needs the same O(n K) per step with more tuning. Selection
boundary: use EM when the M-step is closed form. Use gradients
when it is not.

Failure case. EM climbs to a local maximum, not the global one.
Start at mu = [2.5, 2.6] and it converges to mu = [2.12, 2.95]
with log-likelihood -14.11, far below -3.33. The monotone promise
holds (it rose from -24.86). The destination is still bad.
Counterexample: the bound touches locally. Monotone is not
global. Initialization decides (SB23).

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (the bound),
4 (the E/M code), 5 (monotone numbers), 7 (bad-start
counterexample).

Assessment: E11-E13, L03.

---

## SB23, Lec 24 and Tutorial 8: EM for GMMs, initialization, degeneracy

Playlist titles: "Lec 24 EM for GMMs", "Tutorial 8 Computation of
EM for GMMs". Leaves: math-ml-U09-C08 (latent posterior),
math-ml-U09-C10 (initialization), math-ml-U09-C11 (degeneracy).

Motivating question: the same algorithm from two starts reaches two
answers. Which one do you trust?

Start from zero. Same six points, 20 EM iterations. Start A
mu = [1, 4]: ends at mu = [0.0000, 5.0667], log-likelihood -3.3320.
Start B mu = [2.5, 2.6]: ends at mu = [2.1213, 2.9460],
log-likelihood -14.1109. Both climbs were monotone. Start B found
a poor local maximum where the components never separated. The
algorithm is deterministic given the start. The start is the
algorithm's fate.

Mental model. The likelihood surface of a mixture is lumpy: K!
symmetric hills (label permutations) plus spurious hills where
components share the data badly. EM walks uphill from where you
drop it. Multiple restarts with the best final likelihood is the
standard defense. The responsibilities are the latent posterior
p(z | x): after convergence from start A they are [1, 1, 1, 0, 0,
0], a hard, confident assignment. Shell 2.

Variables. Same as SB22. Restart count R. Best-of-R likelihood.
Assumptions: enough restarts to hit the good basin. No theorem
says how many is enough.

Why it exists. Every EM user meets the bad start. The defense is
procedural (restarts, k-means start, annealing), not mathematical.
Know the failure and you stop trusting single runs.

Degeneracy. The likelihood is unbounded: set mu_1 = x_1 = 0.2 and
let sigma_1^2 -> 0. The first point's density explodes as
1/sigma_1 while the other points keep finite density under
component 2. The likelihood goes to infinity. The "MLE" does not
exist. In the toy run, sigma^2 shrank 1.0 -> 0.0592 -> 0.0444
across two iterations: the direction of the collapse is visible
even though this run stopped at a sensible answer. Practical
defenses: a variance floor, a prior on the variances (MAP, SB24),
or tied/shared variances.

Computed example, same objects. Computed 2026-10-06:

| start | final mu | final log-likelihood |
|---|---|---|
| A [1, 4] | [0.0000, 5.0667] | -3.3320 |
| B [2.5, 2.6] | [2.1213, 2.9460] | -14.1109 |

Figure: the table is the figure. One claim: the start decides the
destination. Source: original toy, computed 2026-10-06.

Code: the SB22 snippet run for 20 iterations from each start.
Check: both runs' likelihood sequences are non-decreasing
(verified in the script output: B rose -24.86 -> -14.11 ->
-14.11). Expected: monotone in both, different endpoints.

Costs. R restarts multiply the bill by R. Alternatives: k-means++
start, deterministic annealing, Bayesian mixtures that integrate
over the uncertainty. Selection boundary: restarts when the M-step
is cheap. Smarter starts when it is not.

Failure case. The degenerate infinite likelihood above.
Counterexample: a run that "converges" to sigma^2 = 1e-12 on one
point is not a discovery. It is the optimizer falling off the
surface's edge.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (restart rule),
5 (monotone check both runs), 7 (degeneracy counterexample).

Assessment: E14-E15, L04.

---

## SB24, Lec 25 and Tutorial 9: MAP estimate and identifiability

Playlist titles: "Lec 25 MAP Estimate", "Tutorial 9 MAP Estimate".
Leaf: math-ml-U04-C09 (identifiability).

Motivating question: when do two different parameters mean the
same model?

Start from zero. Two examples. First, MAP on a coin: 7 heads in 10
flips, Beta(2, 2) prior. MLE = 0.7. MAP = (7+2-1)/(10+2+2-2) =
8/12 = 0.6667. Posterior mean = 9/14 = 0.6429. The prior pulls the
estimate toward 0.5, gently. This is U04-C07's MAP in source-block
clothing, with the Tutorial 9 conjugate computation shown.

Second, identifiability in the mixture: at mu = [0, 5] the
incomplete log-likelihood is -9.8125. Swap the labels,
mu = [5, 0]: the log-likelihood is -9.8125 again, to all printed
digits. The likelihood cannot tell the two apart. The parameter
vector is not identifiable: the map from parameters to
distributions is many-to-one (K! permutations here). Any point
estimate must break the symmetry by convention (order the means)
or the "answer" is arbitrary.

Mental model. Identifiability asks whether the data can separate
two parameter values. If p(x | theta_1) = p(x | theta_2) for all x,
no sample size separates them: the likelihood ridge never
sharpens. MAP with an asymmetric prior can select one ridge point,
but the selection comes from the prior, not the data. Report
which is which. Shell 2.

Variables. theta parameter vector. Identifiable: the map theta ->
p(x | theta) is one-to-one. Label switching: permuting component
labels leaves the mixture unchanged. Assumptions: the likelihood
is the only information. Priors add information and change the
question.

Why it exists. Non-identifiable models are common: mixtures,
factor models with rotations, neural nets with permuted hidden
units. Inference still works if you ask identified questions
(predictions, not labels). It breaks if you ask "what is mu_1".

Computed example, same objects. Computed 2026-10-06:

| quantity | value |
|---|---|
| MLE (7/10) | 0.7000 |
| MAP Beta(2,2) | 0.6667 |
| posterior mean | 0.6429 |
| loglik mu=[0,5] | -9.8125 |
| loglik mu=[5,0] | -9.8125 |

Figure: the table is the figure. Two claims: the prior pulls
gently. The swap changes nothing. Source: original toy.

Code.

```python
k, n = 7, 10
map_est = (k + 2 - 1) / (n + 2 + 2 - 2)
print(map_est, k / n)  # 0.6666666666666666 0.7
```

Check. (8)/(12) = 0.6667. MLE 0.7. Expected output:
0.6666666666666666 0.7.

Costs. MAP with conjugate priors: O(1) extra. Alternatives:
report the posterior over the identified quantities only.
Selection boundary: use MAP to stabilize non-identifiable fits.
Do not mistake the prior's choice for the data's verdict.

Failure case. A symmetric prior on a mixture does not fix label
switching: the posterior has K! symmetric modes and the MCMC chain
wanders between them. Counterexample: averaging mu_1 over the
chain gives ~2.5, the midpoint, which is neither component. The
"posterior mean" of a non-identified parameter is meaningless.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (the swap test),
5 (0.6667 and -9.8125 checks), 7 (symmetric-posterior
counterexample).

Assessment: E16-E17, L05.

---

## SB25, Lec 26: Parzen window

Playlist title: "Lec 26 Parzen Window". Deepens math-ml-U04-C10
(density estimation, taught in lesson-04a).

Motivating question: can you estimate a density without naming its
family?

Start from zero. Points [0.5, 1.5, 2.5, 4.0]. Fix bandwidth h = 0.5
and the Gaussian kernel. The Parzen estimate at x = 1.5:
p_hat = (1/(4 * 0.5)) * sum_i K((1.5 - x_i)/0.5). Kernel terms:
K(2) = 0.0540, K(0) = 0.3989, K(-2) = 0.0540, K(-5) ~ 0. Sum =
0.5069. p_hat(1.5) = 0.5069/2 = 0.2535. Each point spreads its
mass in a Gaussian puff of width h. The estimate is the average
puff.

Mental model. Parzen (kernel density) estimation is democracy: no
parametric candidate wins outright, so every point votes with a
smooth ballot around itself. h sets the ballot's reach. Small h:
spiky, high variance. Large h: flat, high bias. The bias-variance
knob returns, now as a bandwidth. Shell 2.

Variables. h bandwidth. K kernel, integrates to 1. p_hat(x) =
(1/(n h)) sum_i K((x - x_i)/h). Assumptions: the kernel is fixed,
h is fixed before seeing the evaluation point.

Why it exists. Histograms (lesson-04a SB07b) depend on bin edges.
Parzen removes the edges and keeps the smoothing idea. It is the
nonparametric density estimator you reach for first.

Computed example, same objects. Computed 2026-10-06: p_hat(1.5) =
0.2535.

Code.

```python
from math import sqrt, pi, exp
xp = [0.5, 1.5, 2.5, 4.0]
h = 0.5
xq = 1.5
K = lambda u: exp(-0.5 * u * u) / sqrt(2 * pi)
terms = [K((xq - v) / h) for v in xp]
print([round(t, 4) for t in terms], sum(terms) / (4 * h))
# [0.054, 0.3989, 0.054, 0.0] 0.25346285007366176
```

Check. K(0) = 1/sqrt(2 pi) = 0.3989. K(2) = 0.3989 * e^-2 =
0.0540. Sum 0.5069, over 2 = 0.2535. Expected output:
[0.054, 0.3989, 0.054, 0.0] 0.25346285007366176.

Costs. One evaluation: O(n). n evaluations: O(n^2). Alternatives:
histograms (cheaper, edge-dependent), Gaussian mixtures
(parametric, need K). Selection boundary: Parzen for quick
nonparametric looks at small n. Mixtures when you need a compact
model.

Failure case. In high dimensions the puffs cover nothing: with
fixed n, the estimate at most points is ~0. Counterexample: 4
points in 10 dimensions. h = 0.5 reaches nobody. The curse of
dimensionality eats kernel methods first.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (the puff rule),
5 (0.2535 check), 7 (high-dimension counterexample).

---

## SB26, Lec 27: nearest neighbor classifier

Playlist title: "Lec 27 Nearest Neighbor Classifier". Deepens
math-ml-U05-C09 (capacity).

Motivating question: what is the simplest classifier with zero
training error?

Start from zero. Train points [0, 1, 2, 3], labels [0, 0, 1, 1].
1-NN: predict the label of the nearest train point. Training
error: 0.0, because each point is its own nearest neighbor. New
point x = 2.4: nearest is x = 2, label 1. The decision boundary
sits at the midpoints: 0.5, 1.5, 2.5. 3-NN at x = 1.5: neighbors
0, 1, 2 with labels 0, 0, 1. Majority vote: 0. The vote smooths the
boundary. k is the capacity knob: k = 1 memorizes, k = n predicts
the majority class everywhere.

Mental model. k-NN is memory with a vote. No training phase, no
parameters, just the dataset and a distance. Capacity is 1/k:
small k means flexible, large k means stiff. It is the purest
demonstration of U05 C09: the train error is always 0 at k = 1,
and the test error is whatever the noise dictates. Shell 2.

Variables. k neighbor count. d(x, x') distance, here |x - x'|.
Assumptions: the distance means something (features on comparable
scales). The labels near x resemble the label at x (smoothness).

Why it exists. It is the baseline every fancier classifier must
beat, and the cleanest capacity demo in the course. It also shows
why "zero training error" is not an achievement.

Computed example, same objects. Computed 2026-10-06: 1-NN train
error 0.0. NN of 2.4 is x = 2. 3-NN vote at 1.5: 1/3 positive,
predicts 0.

Code.

```python
xtr = [0., 1., 2., 3.]
ytr = [0, 0, 1, 1]
pred_tr = [ytr[min(range(4), key=lambda i: abs(xtr[i] - v))] for v in xtr]
print(sum(p != t for p, t in zip(pred_tr, ytr)) / 4)  # 0.0
```

Check. Each point's nearest neighbor is itself (distance 0).
Error 0. Expected output: 0.0.

Costs. No training. One query: O(n) distances, O(n log n) with an
index. Alternatives: parametric classifiers (logistic) that
compress the data into weights. Selection boundary: k-NN for small
n and meaningful distances. Parametric when n is large or
distances are meaningless (raw pixels without features).

Failure case. Irrelevant features dominate the distance: with 100
noise dimensions and 1 signal dimension, "nearest" is random.
Counterexample: the toy with x replaced by 100-D noise. 1-NN
becomes a coin flip. Distance needs a representation first.

Shells: 0 (question), 1 (toy), 2 (definitions), 3 (the vote rule),
5 (0.0 check), 7 (noise-dimension counterexample).

---

## Exercises E01-E18

E01. State the Bayes decision rule for 0-1 loss and the assumption
it needs.
E02. Compute the Bayes error for priors 0.5/0.5,
p(x|0) = N(0,1), p(x|1) = N(2,1). Show the boundary.
E03. A rule uses threshold x = 1.5 on the SB18 toy. Compute its
error and the excess over Bayes.
E04. With asymmetric costs (false negative costs 10x false
positive), is argmax posterior still optimal? What changes?
E05. From the SB20 table, which threshold does Neyman-Pearson pick
at alpha = 0.1, and what TPR does it achieve?
E06. Compute AUC = 0.9167 from the ROC points by the trapezoid
rule. Show the terms.
E07. Why is a threshold tuned on training scores optimistic? Give
the small-sample reason from the toy.
E08. Minmax picks t = 0.5 on the toy. Construct a cost model under
which t = 0.3 wins instead.
E09. Write the incomplete likelihood for the 6-point mixture toy.
Why does the log block direct maximization?
E10. Compute the responsibility of component 1 for x = 0.2 at the
SB22 initialization. Show the numbers.
E11. After EM iteration 1 the log-likelihood is -3.4450, up from
-13.0089. Name the inequality that guarantees the rise.
E12. Start B ends at log-likelihood -14.1109 versus -3.3320 for
start A. Both climbs were monotone. Explain in two sentences.
E13. Write the M-step update for mu_1 in terms of responsibilities.
E14. From the SB23 table, state the procedural defense against bad
starts.
E15. Explain the degeneracy: with mu_1 = 0.2 and sigma_1^2 -> 0,
what happens to the likelihood and why?
E16. Compute the MAP estimate for 7 heads in 10 flips with a
Beta(2, 2) prior. Compare with the MLE.
E17. Show numerically that swapping the mixture labels leaves the
log-likelihood unchanged. What does this prove?
E18. Compute the Parzen estimate at x = 1.5 for the SB25 toy by
hand. Show the four kernel terms.

## Deep ladders L01-L05

L01, Bayes. (1) Define the Bayes classifier. (2) Toy: the 0.1587
computation. (3) Derive: the pointwise risk argument. (4) Debug: a
teammate's "Bayes" classifier uses estimated densities and beats
0.1587 on test data. Explain. (5) Transfer: costs are asymmetric.
Derive the new rule.
L02, thresholds. (1) Define TPR, FPR, ROC, AUC. (2) Toy: the SB20
table. (3) Derive: the NP constrained optimum from the table.
(4) Debug: deployed FPR breaks the 0.1 budget though validation
said 0. Name two causes. (5) Transfer: fraud detection with a
daily analyst budget. Your threshold protocol.
L03, EM. (1) Define the E and M steps. (2) Toy: one hand
iteration. (3) Derive: the Jensen bound chain. (4) Debug: the
likelihood decreases between iterations. Name the bug class.
(5) Transfer: the M-step has no closed form. Your options.
L04, GMM practice. (1) Define responsibilities. (2) Toy: the two
starts. (3) Justify: why restarts are the standard defense.
(4) Debug: sigma^2 hits 1e-12 on one component. Diagnose.
(5) Transfer: 1M points, 50 components. Your EM budget plan.
L05, MAP and identifiability. (1) Define identifiability. (2) Toy:
the swap test. (3) Derive: the Beta-Bernoulli MAP. (4) Debug: the
MCMC posterior mean of mu_1 is 2.5. Diagnose. (5) Transfer: you
must report one clustering. How do you break the symmetry?

## Not-yet-understood dependencies

- The playlist's spoken derivations for Lec 22-23 are uninspected
  (G2). The Jensen chain above is the standard argument, taught as
  bridge content, not as the instructor's words.
- Tutorial 8's actual computations are uninspected (G6). The SB23
  numbers are my own EM run, not the tutorial's.
- U09's remaining leaves (k-means, PCA, distance/scaling,
  reconstruction, factor models, held-out evaluation) are RUN 5
  scope. The EM-taught rows are marked now. RUN 5 must not
  re-teach them.
