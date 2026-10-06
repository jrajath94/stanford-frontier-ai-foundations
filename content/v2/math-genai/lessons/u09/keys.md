# Answer keys, lesson 09 (score-based models and autoregressive LMs)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run5b.py.

## E01

s(x) = -(w_1 (x + 1.5) + w_2 (x - 1.5))/0.25. At x =
1.0: N_1 = gauss(1.0, -1.5, 0.25), N_2 = gauss(1.0,
1.5, 0.25). w_2 ~ 1 (x near the right mode), so s ~
-(1.0 - 1.5)/0.25 = 2.0. Exact: 1.999926.

## E02

s = d/dx (log p_unnorm - log Z) = (d/dx log
p_unnorm) - 0. The derivative kills the constant
log Z, so the score never needs the normalizer.

## E03

Target = (x - x_tilde)/sigma^2 = (-1.5 + 1.9)/0.04
= 0.4/0.04 = 10.0. Points right, toward the clean
-1.5 from -1.9.

## E04

s(0.6) = 3.591048. x_1 = 0.6 + 0.05 3.591048 +
0.31622777 0.5 = 0.6 + 0.17955240 + 0.15811389 =
0.937666.

## E05

Not wrong: slow. 56 valley crossings in 20000
steps at a = 0.1 means the chain rarely switches
modes, so a 20000-step average still shows the
starting side's bias (frac(x > 0) = 0.56, mean
0.161). At a = 0.5 (414 crossings) the mean is
0.043. The sampler is correct but mixing-limited,
this is why diffusion anneals noise.

## E06

s(-1.0) = -1.999926. Sampling update x <-
x + 0.5 beta s h with beta = 0.1, h =
0.1: dx = 0.05 (-1.999926) 0.1 =
-0.00999963. x_1 = -1.00999963, toward
the mode at -1.5. Sign note: the
forward-time ODE is dx = -0.5 beta s
dt, sampling integrates it backward,
which flips the sign to +0.5 beta s h.

## E07

-log2(0.6) - log2(0.5) = 0.73696559 + 1.0 =
1.73696559 bits. Verified.

## E08

2^{1.73696559/2} = 2^{0.86848280} = 1.82574186.
Bounds: [1, 4] for V = 4.

## E09

1.6966 bits for the second token only. The
two-token version (3.2675) also scores the
sampled first token (-log2 P(x_1|a)), which
adds the model's own first-token uncertainty.
They differ because they score different
random objects.

## E10

[[1,0,0],[1,1,0],[1,1,1]] (lower triangular).
Row 1 (second position) allows positions 0
and 1.

## E11

Scores row 2: [0.7071, 0.7071, 1.4142]. Exp:
[2.0281, 2.0281, 4.1133]. Sum 8.1695.
Normalized: [0.2483, 0.2483, 0.5035].

## E12

Row 0 of A is [1, 0, 0] by the mask (only
self visible), so O[0] = 1 X[0] + 0 + 0 =
X[0]. No computation of A needed beyond the
mask.

## E13

X changes (P removed), so S, A, O all change.
The mask stays (it is constant). Measured:
row 1 changes by 0.0983 max.

## E14

z/0.5 = [4, 2, 1, 0.2]. Softmax: [0.8282,
0.1121, 0.0412, 0.0185]. First entry
verified.

## E15

Sorted desc: [0.5745, 0.2114, 0.1282, 0.0859].
Cumsum: 0.5745, 0.7859, 0.9141. Keep while
cumsum <= 0.9: indices [0, 1]. Renorm:
[0.7311, 0.2689, 0, 0].

## E16

12 12 2048^2 4 = 2415919104 bytes = 2.42 GB.

## E17

Error: perplexities on different vocabs are
incomparable (different event spaces). Repair:
bits-per-byte, or compare only within one
tokenizer.

## E18

Greedy: entropy 0 (one token). T = 2.0:
entropy 1.9017 bits. T = 2.0 higher.

## E19

Row 0 = softmax([s00, s01]) = [0.5, 0.5, 0]
on the toy (equal scores). Position 0 sees
x_1 during training: the loss for predicting
x_1 collapses, generation breaks (copies a
future that does not exist).

## E20

(a) Images 64x64: score/diffusion. Order-free
continuous data, the AR order is arbitrary
and O(n^2) on 4096 tokens is heavy, the
score culture's 56-crossing mixing is less
relevant for unimodal-per-patch images.
(b) Code: AR. Exact likelihood (1.82574 toy
perplexity) for model comparison, natural
left-to-right order, teacher forcing trains
fast.

## L01

Score: grad log p, no normalizer (E02).
DSM target: (x - x_tilde)/sigma^2, computed
-5.0 and 10.0. Five Langevin steps: the
trajectory in C03. Slow mixing: 56
crossings, mean 0.161 not 0 (E05). ODE:
deterministic, ~50 evals vs 20000.
Annealing experiment: noise ladder high to
low, predict crossings beat any fixed a.

## L02

Factor: p(a,b,c) = p(a)p(b|a)p(c|b). NLL
1.73697, ppl 1.82574. Exposure: 0.8685 ->
1.6338 bits/token, gap 0.7653. Mask: lower
triangular, row 1 allows 0,1. Forward: O
rows computed, O[0] = X[0] by the mask.
Mask bug: row 0 [0.5,0.5,0], loss
collapses, generation breaks. RNN: O(n)
sequential, O(d) state vs O(n^2) parallel.
Order ablation: reverse the factorization
order, predict worse modelling on text.

## L03

T/top-p: three distributions, entropies
0.8754/1.6174/1.9017. Memory: 0.60 GB at
1024, 9.66 at 4096, 2.42 at 2048. Vocab
bug: incomparable perplexities, use
bits-per-byte. Cultures: table in C12
with the four anchor numbers. Hybrid:
VQ + AR prior, predict exact likelihood
but worse samples at equal budget.

## L04

eps to score: s_hat = -eps_hat/sqrt(1 -
ab_t), toy -0.84738932, Tweedie and eps
forms agree to 1e-12. One net, three
samplers: the marginals are shared, only
the reverse rule changes (DDPM kernel,
DDIM jumps, Langevin/ODE on the score).
ODE sampling step: 1.00999963 (toward the
mode). Critique: DDIM eta = 0 IS the
probability-flow ODE discretization,
"not score-based" is false. Evidence: the
update uses x0_hat from eps_hat, i.e. the
score.

## L05

Score culture assumptions: score
learnable (needs noise ladder), Langevin
mixes (fails in deep valleys: 56
crossings), ODE discretization fine.
Break the ladder (sigma -> 0): valley
score garbage, samples miss modes. AR
assumptions: fixed order is natural,
teacher forcing transfers, O(n^2)
affordable. Break the mask: loss lies,
generation dies. Visible failures:
mode-missing samples vs gibberish text.

## Implementation and debug task

Reference: score_mix, langevin_step,
causal attention, transformer forward as
in compute_run5b.py. Bug 1: slow mixing,
not non-convergence, fixes: larger a
(0.1 -> 0.5: 56 -> 414 crossings) or
annealed ladder. Bug 2: mask off by one
(allows j <= i+1), fix: strict j <= i,
test row 0 == [1, 0, 0].

## Changed-constraint scenarios

S1. Changed: scores near 0 (the new mode
at 0 has s = 0 there too, but now it is
a mode not a valley), DSM targets for
x_tilde near 0, crossing counts (three
modes mix differently). Unchanged: the
formulas, the mask, all AR numbers. s(0):
still 0 by symmetry (weights shift but
the odd symmetry about 0 holds if the
third mode is centered at 0).
S2. 38.65 GB needed, 12 GB available.
Levers: fp16 (halves to 19.33 GB, still
over), 6 layers (halves again to 9.66
GB: fits), sliding window w = 2048
(quarters the n^2 term: 9.66 GB at full
12 layers). Concrete: 12 layers, fp16,
window 2048: 38.65/2/4 = 4.83 GB.
Fits with headroom, quality cost is the
window.

## Research-critique question

Strong answer: exact likelihood buys
comparability on fixed vocabs and no
variational gap, it does not buy sample
quality (perplexity 1.83 can still read
badly) or the right bias for images
(order-free data). Test: one image
dataset, one text dataset, both a
score/diffusion model and an AR model,
metrics: likelihood (where defined) +
sample quality (FID/human). Predict the
ranking flips by modality. Red flags:
"strictly better" without a metric,
confusing the map (likelihood) with the
territory (usefulness).
