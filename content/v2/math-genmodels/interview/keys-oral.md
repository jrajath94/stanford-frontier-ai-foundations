# Oral defense keys

Course: math-genmodels. Date: 2026-10-06. Baseline: October 6, 2026.

Strong answers, red flags, rubrics. Each follow-up builds on the
last. A defense passes when the learner answers all 8 without
notes.

## O1

1. p(x) = integral p(x|z) p(z) dz. Intractable: the integral
   has no closed form for neural decoders.
2. Marginal 0.60, posterior (0.8, 0.2), ELBO -0.5390 at q =
   (0.7, 0.3) (U02 toy).
3. log p = log E_q[p(x|z)p(z)/q]: Jensen moves log inside
   because log is concave: the inequality enters exactly
   there.
4. Gap = KL(q||posterior). Zero iff q equals the posterior.
5. eps = rng.standard_normal(). z = mu + sigma x eps.
   g = d loss / d z, the upstream scalar. dmu, dsigma =
   g, g x eps: dz/dmu = 1, dz/dsigma = eps. Three lines: the
   gradient flows through the deterministic map, not through
   the sampler.
6. ELBO: O(1) per sample. Exact: O(2) paths on the toy, O(K)
   generally.
7. q too weak (mean-field gap, U02-C11) or decoder too
   strong (ignores z, U03-C05).
8. A higher ELBO can be a looser bound on a worse model:
   -1.70 bound versus -1.7371 exact proves nothing about
   the true ranking (U08-C02). Red flag: "higher is better."

## O2

1. One-to-one and onto. z^2: g(1) = g(-1) = 1, no inverse.
2. J = [[1, 0], [0, 1.25]] at (0.5, -1).
3. Expand along the first row: only the diagonal entry
   survives. Induct on D.
4. log p(y) = -2.4629 - 0.2231 = -2.6860. The minus: the
   inverse map's Jacobian corrects the base score.
5. forward/inverse as in U04-C05. Round-trip 1000 points,
   max error < 1e-12.
6. Dense: O(D^3) = 1B ops at D = 1024. Triangular: O(D) =
   1024 logs.
7. -2.2398 = -2.4629 + 0.2231: the sign is flipped. Fix the
   minus. Test against a histogram.
8. Exactness costs invertibility and expressivity limits
   (one affine layer stays Gaussian, U04-C11). VAEs win on
   architecture freedom. The fair test is held-out NLL at
   matched budgets. Red flag: "exact always wins."

## O3

1. x = g(z): a sampler with no density formula. It cannot
   score p(x).
2. r(0) = 1.6487, D*(0) = 0.6225.
3. d/dD [p log D + q log(1-D)] = 0 gives p/(p+q).
   Second derivative negative: concave, global max.
4. Plug D* in: -log 4 + 2 JSD = -1.1635 on the toy.
5. For each G step: k D steps on real/fake batches, then
   one G step on -log D(g(z)) (non-saturating).
6. Two network updates per cycle versus one for a VAE.
   The k multiplier makes D the expensive player.
7. Perfect accuracy means disjoint laws or memorization:
   JSD = log 2, gradient zero (U05-C05). The generator gets
   no signal.
8. FID measures feature moments, not densities (U08-C05),
   and GANs have no density to estimate: category error.
   Fair: precision/recall plus likelihood where densities
   exist. Red flag: "FID decides."

## O4

1. s(x) = grad log p(x). N(0,1): -x, so -1 at x = 1.
2. x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps.
   Toy: 1.5174.
3. Tweedie: E[x_0|x] = x + sigma^2 score. Train the net to
   predict eps: the eps prediction is the score up to
   scale. The trace never appears because the model is
   never differentiated.
4. x0_hat = 1.6917, x_1 = 1.7630, deterministic.
5. x += 0.5 x delta x score + sqrt(delta) x z. Stationary:
   exactly p as delta -> 0.
6. DDPM-1000: 1000 calls. DDIM-50: 50. Heun-50: 100 (2 per
   step).
7. Tweedie single-step test: denoise a grid with the net.
   Sharp there but mushy after full sampling proves the
   sampler (schedule mismatch), not the model.
8. The ODE likelihood needs a noise floor (U06-C12): the
   ranking can flip with it. And likelihood is one
   projection: pair it with samples and cost. Red flag:
   "likelihood decides."

## O5

1. p(abc) = p(a)p(b|a)p(c|a,b) = 0.05.
2. exp(NLL/L) = 2.7144: effective branching factor.
3. Row [1, 1, 0]: positions 1 and 2 only.
4. Scores [[0.7071,-inf],[0,0.7071]], weights [1,0] and
   [0.3302, 0.6698], outputs [2,3] and [3.3395, 4.3395].
5. k_new, v_new = proj(x_t). append to cache. attend over
   the cache. sample x_{t+1}.
6. 16.8M mults per head. 256 KB cache per layer per
   sequence in fp32.
7. Off-by-one label shift: the model predicts x_t from x_t
   (copying). Test: shift inputs by one, outputs must
   shift. constant sequences must not give near-zero loss.
8. Pseudo-perplexity sums to 1.0639 on the toy, not 1.0:
   not a likelihood (U07-C10). And tokenizations differ:
   incomparable scales. Red flag: "lower is better."

## O6

1. Sum of log densities on unseen points. A: -10.1073, B:
   -70.949.
2. Precision: samples on the manifold. Recall: modes
   covered. A: 0.4708/1.0. B: 0.9555/0.5.
3. Truth bimodal, model A Gaussian, matched moments, FID
   0: moments are blind past order two.
4. NN distance per sample to the training set. Memorizer
   0.0066, honest 0.4666.
5. Fix budgets (params, flops, inference cost). Tune each
   model symmetrically. Freeze metrics. Report with seed
   uncertainty. Decide by pre-registered rule.
6. The suite costs 5x evaluation (seeds) plus sample
   generation: still dwarfed by training. FID alone is
   cheap and blind.
7. Inception never saw X-rays: the features are
   irrelevant. Fix: domain features plus human ratings.
   Test: two sets humans rank differently must separate on
   the metric.
8. Every metric has a blind spot (U08-C01..C06): the suite
   covers them. One number is one blind spot with
   confidence. Red flag: "the leaderboard decides."

## O7

1. ELBO = E_q[log p(x|z)] - KL(q||p(z)): reconstruction
   minus rate.
2. KL = 0.9013 nats on the toy (U03-C04).
3. The decoder is strong enough to model x without z, so
   optimization kills the KL: q collapses to the prior and
   z carries nothing. Mechanism: the rate term is pure cost
   when the decoder does not need z.
4. Beta scales the rate price: beta < 1 buys more rate for
   less distortion budget, moving up the frontier.
5. eps = standard_normal(d). z = mu + sigma x eps.
6. VAE: one forward-backward per batch. Flow: one inverse
   pass plus D diagonal logs per layer: same order, exact.
7. Collapse with a memorizing decoder. Fixes: beta < 1 or
   KL annealing (objective). weaker decoder (architecture).
   Both buy rate at distortion cost.
8. The gap has three sources: amortization (one net for all
   x), family (q too simple), optimization. Only the third
   is "just optimization." Red flag: "train longer."

## O8

1. E_p[0.5 ||s_theta - s_p||^2]: squared score error.
2. Cross term E[s_theta . s_p] = -E[div s_theta] by parts.
   Boundary: p s_theta -> 0 at infinity.
3. Denoising trains on noisy samples: the model is never
   differentiated, so div s_theta never appears. Tweedie
   links the noise prediction to the noisy score.
4. Objective -a + 0.5 a^2: min at a = 1, value -0.5.
5. obj(a) = mean(-a + 0.5 a^2 x^2) over the 5,000 draws.
   grid over a.
6. O(D) backward passes naive at D = 1024. Sliced: one
   random projection per step.
7. Score matching only sees data: far away the field is
   extrapolation. Fix: the noise schedule (C04) covers
   space at many scales.
8. The boundary term must vanish (tails), and p must be
   smooth with full support (C12: boundaries break it).
   Red flag: "it just works."

## O9

1. Path x_t = (1-t) x_0 + t x_1. Target velocity u = x_1 -
   x_0 = 2 at every t.
2. E[||v_theta(x_t,t) - (x_1-x_0)||^2]. Toy: (2-1.8)^2 =
   0.04.
3. The conditional targets average over endpoint pairs to
   the marginal velocity field. regression on the
   conditionals recovers the mean, which is what the ODE
   needs.
4. DDPM learns the noise (score). flow matching learns the
   velocity directly. Same path idea, different
   parameterization.
5. Sample x_0, x_1, t. Form x_t. Predict v. Squared error.
   Step.
6. One ODE solver step per stage: Euler 1 call, Heun 2.
   50 steps = 50-100 calls.
7. Crossing paths: independent couplings make conditional
   targets conflict. Fix: OT couplings (straighter paths).
8. The path is a choice (straight is arbitrary), and the
   ODE still needs the score-like smoothness (C09
   assumptions survive). Red flag: "no assumptions."

## O10

1. Best likelihood, best samples, best metric value, best
   efficiency. Four different claims.
2. Bound -1.70 versus exact -1.7371: the truth lies in
   [-1.70, infinity). Undecided.
3. Precision 0.9555 with recall 0.5: sharp and incomplete.
   "Best samples" needs both.
4. Matched moments, FID 0, wrong shape: the metric is
   blind past order two.
5. Same params, same flops, same inference cost, symmetric
   tuning. Else the budget won, not the idea.
6. Metrics: held-out NLL (exact), precision/recall, MMD.
   Budgets matched. 5 seeds. Rule: win 2 of 3 with
   non-overlapping error bars.
7. Report it: the falsified hypothesis, the numbers, the
   mechanism. It counts because it rules out a design
   direction (the capstone's GP finding).
8. A field that cannot publish "this does not work" keeps
   retrying what does not work. Falsification is the
   filter: pre-registered, reported, permanent. Red flag:
   "file-drawer it."
