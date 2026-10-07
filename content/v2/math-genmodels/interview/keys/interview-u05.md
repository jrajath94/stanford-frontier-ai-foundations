# Interview keys: U05

Unit: math-genmodels-U05. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice. Strong answers, red flags, rubrics,
remediation.

## Q1

Strong: a GAN defines its law implicitly through x = g(z). A
likelihood needs p(x), which needs the integral over all z with
g(z) = x. That integral is intractable for a neural g. Red flag:
quoting the discriminator output as a likelihood. Rubric: the
implicit definition plus the missing integral. Remediation: U05-C01.

## Q2

Strong: D*(x) = p(x)/(p(x) + q(x)), the Bayes posterior for the
real-versus-fake task. Divide top and bottom by q: D* =
r/(1+r), a monotone function of the ratio r = p/q. Red flag:
D* = p/q. Rubric: the formula and the ratio rewrite.
Remediation: U05-C02, C03.

## Q3

Strong: V = E_p[log D] + E_q[log(1-D)]. At D*: E_p[log
p/(p+q)] + E_q[log q/(p+q)] = -log 4 + 2 JSD. The -log 4 is the
value at p = q, where both logs are log(1/2). Red flag: sign
errors in the algebra. Rubric: both lines with the -log 4
explained. Remediation: U05-C04.

## Q4

Strong: JSD is bounded by log 2. On disjoint supports it sits at
the bound with zero gradient. W1 measures transport cost, which
shrinks linearly as the laws approach, so its gradient never
dies. Red flag: "W1 is just better" without the mechanism.
Rubric: the bound argument plus the transport mechanism.
Remediation: U05-C05, C06.

## Q5

Strong: the Kantorovich dual needs the max over 1-Lipschitz f.
The constraint keeps the value finite and a true distance. Without
it the critic steepens without bound and the objective explodes.
Red flag: treating the constraint as a regularizer. Rubric: the
dual role, not a tuning trick. Remediation: U05-C07.

## Q6

Strong: the penalty is soft (allows slope != 1 off the
interpolation lines) and capacity-preserving. Clipping is hard
and damages capacity when c is small. Pick the penalty when
critic capacity matters. pick clipping (or spectral norm) when
you need a cheap hard guarantee. Red flag: claiming clipping is
always worse. Rubric: two differences plus the selection rule.
Remediation: U05-C07, C08.

## L1 ladder

1. V(D,G) = E_{x~p}[log D(x)] + E_{z}[log(1 - D(g(z)))].
2. Pointwise max: p/D - q/(1-D) = 0 gives D* = p/(p+q),
   concave, global.
3. D* = r/(1+r) with r = p/q.
4. Plug in: -log 4 + 2 JSD. Toy: JSD = 0.1114, V* = -1.1635.
5. Perfect accuracy means disjoint laws or memorization. JSD =
   log 2, gradient zero. The generator gets no signal. Strong
   answers name both the saturation and the memorization
   reading. Red flag: "the game converged."

## L2 ladder

1. Mode collapse: the generator covers a strict subset of modes.
   Toy: all mass at 0, both true modes empty.
2. Collapsed: precision 1.0, recall 0.0 on the left mode. The
   pair exposes what one number hides.
3. Eigenvalues 1 +/- 0.1 i, magnitude sqrt(1.01) per step:
   rotation plus expansion, radius 1.4142 to 1.5622 in 20
   steps.
4. One player's loss can fall while the joint state spirals
   outward. The loss is not a progress metric in a game.
5. Frozen held-out real set plus frozen reference fakes.
   precision and recall over seeds. discriminator accuracy as
   a diagnostic only. Red flag: trusting any single number.

## A1

Strong: start from D*, write E_p[log p - log(p+q)] +
E_q[log q - log(p+q)]. Add and subtract log 2 inside each log:
each term becomes KL to the mixture minus log 2. Sum: KL(p||m)
+ KL(q||m) - 2 log 2 = 2 JSD - log 4. Red flag: dropping the
log 2 shifts. Rubric: the full algebra with the mixture named.

## A2

Strong: W1 = integral |F_p - F_q|. Equal-variance Gaussians have
F_q(x) = F_p(x - s) with shift s = mu2 - mu1. The area between
the curves is |s| times 1. The critic f(x) = sign(s) x is
1-Lipschitz and attains E_p[f] - E_q[f] = |s|. Red flag:
hand-waving the attainment. Rubric: the integral plus the
attaining critic.

## D1

Strong: candidates: (1) the penalty is detached (create_graph
off), so it reads 0 and never constrains: test with a steep
critic probe, the penalty must rise. (2) the critic
overpowered the generator after step 10k (k schedule drift):
test held-out D accuracy, if 1.0 the game saturated. (3) lambda
too small relative to the critic learning rate: test the max
grid slope, it must stay near 1. Order: detachment first
(constant 0.0 is the tell), then saturation, then lambda.
Red flag: restarting training without a diagnosis. Rubric: the
three causes in order with one test each.

## T1

Strong: output the logit s = log p - log q as a score
difference. D = sigmoid(s) via a lookup table. 8-bit fixed
point quantizes s. the table needs no division. Cost: dynamic
range compresses, tails saturate, the 1.6487 ratio becomes
coarse. Calibrate the table on the operating range. Red flag:
quantizing D directly. Rubric: the logit redesign plus the
range cost.

## T2

Strong: keep a bounded reservoir of past fakes or slow D so it
cannot overfit the current window. Freeze a held-out real set
and a reference fake set for evaluation. track precision/recall
there. Worse: mode collapse, since D forgets abandoned modes.
Red flag: streaming the loss as the metric. Rubric: the
reservoir or slow-D fix, frozen evaluation, the named mode.

## R1

Strong: attack 1: FID measures feature mean/covariance match,
not density estimation. GANs have no density to estimate, so
"better density estimator" is a category error. Attack 2: FID
can be gamed and misses mode dropping (C12, U08-C05). Fair
experiment: compare on held-out likelihood where both models
have densities (VAE bound vs a flow), or compare sample
precision and recall separately with a fixed feature
extractor and matched compute. Red flag: accepting FID as a
density verdict. Rubric: both attacks plus the redesigned
comparison.
