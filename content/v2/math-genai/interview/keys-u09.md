# Interview keys, U09 score-based models and autoregressive LMs

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4,
float64. Ground truth: compute_run5b.py. Interview
provenance: role-derived practice, not employer material.

## Breadth

B1. s(x) = d/dx log p(x). The derivative kills log
Z, so no normalizer is needed.
B2. (1.5 - 1.7)/0.04 = -5.0.
B3. x <- x + (a/2) s(x) + sqrt(a) z. Drift climbs
log p, noise explores and sets the stationary
distribution to p.
B4. p(a,b,c) = p(a)p(b|a)p(c|b). NLL =
-log2(0.6) - log2(0.5) = 1.73697 bits.
B5. Lower triangular ones. Row 1 allows positions
0 and 1.
B6. 603979776 bytes (0.60 GB) at 1024,
9663676416 bytes (9.66 GB) at 4096.

## Deep ladder D1

D1.1. The gradient of the log density: the
uphill-pointing vector field.
D1.2. w_2 ~ 1 near the right mode: s ~ -(1.0 -
1.5)/0.25 = 2.0, exact 1.999926.
D1.3. p(x_tilde|x) = N(x, sigma^2), d/dx_tilde
log p = -(x_tilde - x)/sigma^2 = (x -
x_tilde)/sigma^2.
D1.4. Not broken: std matches (1.5873 vs
1.5811), mixing-limited (56 crossings), so the
mean shows the starting side. Fix: larger a or
annealing.
D1.5. A mode at 0 fills the valley: crossings
rise sharply, the mean converges faster. The
valley was the barrier, removing it removes the
pathology.

## Deep ladder D2

D2.1. Train each factor on the true prefix,
never the model's own.
D2.2. TF: 1.7370/2 = 0.8685/token. Free-run:
3.2675/2 = 1.6338/token. Gap 0.7653.
D2.3. Position i with mask sees exactly x_<i,
so its output is p(x_{i+1}|x_<i>), all i run in
one pass.
D2.4. Mask off by one (allows j <= i+1). Fix:
strict j <= i, test row 0 == [1, 0, 0].
D2.5. Attack: free-run training is sequential
(n steps, high variance) and diverges early,
teacher forcing is the stable curriculum.
"Obsolete" confuses the bias (real) with the
remedy (impractical). Strong answer names
scheduled sampling as the interpolation.

## Analytical/quantitative

A1. No ranking: different vocabs, different
event spaces. Comparable: bits-per-byte, or
perplexity only within one tokenizer.
A2. 12 12 8192^2 4 = 38654705664 bytes =
38.65 GB. Config: 12 layers, fp16, sliding
window 2048: 38.65/2/4 = 4.83 GB. Fits.

## Implementation/debug

I1. Bug: causal mask off by one, position 0
sees x_1. The loss lies because predicting a
visible token is trivial (copying). Fix:
mask j <= i strictly. Prevention: assert row 0
== [1, 0, 0] and assert eval perplexity is
sane before trusting train loss.

## Changed-constraint scenarios

S1. ODE solver: fewer evals, discretization
error the price. Distillation: one eval,
needs retraining and can inherit teacher
bias. Both trade evals for fidelity.
S2. Change: softmax O(V), perplexity bounds
[1, 40000]. Unchanged: chain rule form,
mask math, the gap definition. The bigram
numbers scale: rows become 40000-wide.

## Research-critique

R1. Likelihood buys comparability on fixed
vocabs, not sample quality or the right
inductive bias. Design: one image set, one
text set, a diffusion model and an AR model
each, metrics likelihood (where defined) +
FID/human quality. Predict the ranking flips
by modality. Red flag: "strictly better"
without naming the metric.
