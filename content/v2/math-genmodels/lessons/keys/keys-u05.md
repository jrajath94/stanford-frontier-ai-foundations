# Answer keys: U05

Unit: math-genmodels-U05. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers below. Red flags name the common failure. Rubrics say
what earns full credit.

## B1

Strong: an implicit distribution is defined by a sampling
procedure x = g(z), with no tractable density formula. Sampling
needs only the forward pass. Scoring needs the integral over all
z mapping to x, which is intractable. Red flag: claiming the
discriminator output is a likelihood. Rubric: the definition plus
the sampling-versus-scoring asymmetry.

## B2

Strong: r(0) = p(0)/q(0) = 0.3989/0.2420 = 1.6487. Real data is
about 1.65 times as likely at x = 0. Red flag: inverting the
ratio. Rubric: both densities and the quotient.

## B3

Strong: D*(x) = p(x)/(p(x) + q(x)). At x = 0: 0.3989/0.6409 =
0.6225. Red flag: writing D* = p/q. Rubric: the formula and the
number.

## B4

Strong: V = E_p[log D(x)] + E_q[log(1 - D(x))], max over D, min
over G. At D*: V* = -log 4 + 2 JSD(p, q) = -1.1635 on the toy.
Red flag: dropping the max over D before plugging in. Rubric:
the objective and the JSD identity.

## B5

Strong: W1 = max over 1-Lipschitz f of E_p[f] - E_q[f]. The
Lipschitz bound keeps the max finite and makes the value a true
distance. Red flag: omitting the constraint. Rubric: the dual
plus why the constraint is load-bearing.

## B6

Strong: mode collapse is the generator covering a strict subset
of the data modes. Recall sees it (missing modes score 0).
Precision does not (the covered mode looks sharp). Red flag:
calling blurry samples mode collapse. Rubric: the definition
plus the precision/recall split.

## L1 ladder

1. r(x) = p(x)/q(x), unitless, the pointwise comparison of the
   two laws.
2. Pointwise: d/dD [p log D + q log(1-D)] = p/D - q/(1-D) = 0
   gives D = p/(p+q). Concave in D, so global max.
3. p/(p+q): divide top and bottom by q to get r/(1+r).
4. Plug D* in: E_p[log(p/(p+q))] + E_q[log(q/(p+q))] = -log 4 +
   2 JSD. On the toy: -1.1635.
5. With D away from D*, max_D V is not attained, so V is some
   other moving target, not JSD. The generator then optimizes a
   quantity with no divergence meaning.

## L2 ladder

1. On disjoint supports JSD = log 2 constant, so its gradient in
   the generator parameters is zero: no learning signal.
2. Disjoint toy: JSD = 0.6931, W1 = 5.0. JSD is saturated, W1
   measures the gap.
3. W1 moves mass: shrinking the shift shrinks the transport
   cost linearly, so the critic gradient points the generator
   toward the data even with no overlap.
4. Bilinear game: update matrix eigenvalues 1 +/- 0.1 i,
   magnitude sqrt(1.01) per step. 20 steps: norm 1.4142 to
   1.5622. Rotation plus expansion, never lands.
5. WGAN-GP. The laws start far apart, JSD saturates, W1 keeps
   the signal. The penalty enforces the dual constraint without
   clipping damage.

## A1

Strong: for each x, maximize f(D) = p log D + q log(1 - D).
f'(D) = p/D - q/(1-D). Set to zero: p(1-D) = qD, so D =
p/(p+q). f''(D) = -p/D^2 - q/(1-D)^2 < 0 for p, q > 0: strictly
concave, unique global maximizer. Red flag: differentiating
under the expectation without the pointwise argument. Rubric:
the derivative, the solve, the concavity check.

## A2

Strong: disjoint supports: on supp(p), m = p/2, so
p log(p/m) = p log 2. Integrate: KL(p||m) = log 2. Same for q.
JSD = log 2. For 1-D Gaussians with equal sigma: W1 = integral
|F_p - F_q| = |mu1 - mu2|, because the CDFs are horizontal
shifts of each other and the area between them is the shift.
Toy: 1.0. Red flag: confusing W1 with W2 (which would be the
same here but differs in general). Rubric: both proofs with the
toy numbers.

## D1

Strong: bug: the critic violates the Lipschitz constraint while
the penalty reads near zero. Likely cause: the penalty is
computed on the wrong points (e.g. only real samples, not
interpolations), or grad is taken with create_graph off so the
penalty is detached and never enforced. Then the critic grows
steep and the "W1" estimate is an unbounded overestimate. Fix:
compute the penalty on eps-interpolated points with gradients
attached, lambda = 10. Test: a 1-Lipschitz audit on a grid
(max |f(x)-f(y)|/|x-y| <= 1) and the estimate must never exceed
an exact W1 computed on 1-D sorted samples. Red flag: lowering
lambda to hide the symptom. Rubric: the mechanism (unbounded
critic), the fix, the two tests.

## T1

Strong: D* = p/(p+q) needs a division. Replace with the logit
form: the classifier can output the log-ratio s = log p - log q
via a difference of scores, and D = sigmoid(s) needs only exp.
On 8-bit fixed point, quantize s and use a lookup table for
sigmoid. What breaks: the ratio estimate loses dynamic range.
tails saturate and the 1.6487 becomes coarse. Calibrate the
table on the operating range. Red flag: quantizing D directly
and losing the gradient near 0 and 1. Rubric: the logit
redesign plus the named breakage.

## T2

Strong: streaming removes the replay of old fake samples, so
the discriminator forgets past generator behavior: catastrophic
forgetting of the ratio. Schedule: keep a small reservoir of
past fakes (bounded memory), or slow the D learning rate so it
cannot overfit the current window. Evaluation: fix a held-out
real set and a frozen reference generator set. Track
precision/recall against those, not the streaming loss. Worse
failure mode: mode collapse, because D never sees the modes G
abandoned two windows ago. Red flag: trusting the streaming
loss curve. Rubric: the reservoir or slow-D fix, the frozen
evaluation, the named worsened mode.

## R1

Strong: attack 1: perfect held-out accuracy means the laws are
disjoint or the discriminator memorized. Either way JSD is
saturated at log 2 and the generator gets zero gradient
(C05). Attack 2: accuracy measures the classifier, not the
generator. A collapsed generator (C09) also yields high accuracy
while missing modes. Fair experiment: freeze the generator,
measure sample precision and recall against the data modes over
several seeds, and report both with the discriminator accuracy
as a diagnostic, not a verdict. Red flag: accepting accuracy as
a convergence certificate. Rubric: both attacks plus the
precision/recall experiment.
