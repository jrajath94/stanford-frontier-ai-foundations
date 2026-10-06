# Lesson 09, Score-based models and autoregressive LMs

Unit: math-genai-U09. Leaf concepts: math-genai-U09-C01 to C12.
Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

This lesson is locally authored bridge content for prerequisite
modules P05 (calculus), P08 (information theory), P13
(language and sequence modelling), P14 (transformer
mechanics), and P18 (Bayesian inference and sampling). It
does not claim to reproduce the instructor's lectures.
Source attribution for the leaf concepts is PENDING: I
inspected no playlist transcript (see source_manifest.md
SRC-04, source_gaps.md G2). The playlist covers
auto-regressive models in W10L40, the attention mechanism
in W10L41, transformers for auto-regressive models in
W10L42, transformer architecture in W10L43, skip
connections and normalization in W10L44, position
embeddings in W10L45, and training and inference in
W10L46, all by title only. A separate source-block lesson
(lesson-09b) follows those seven titles at the title
boundary. All numbers below are computed 2026-10-06,
numpy 1.26.4, float64, seed 0 where RNG is used
(compute_run5b.py reproduces every one). Log base 2 in
bits for likelihoods. Natural log inside the score
algebra.

## Scope and objectives

Scope: the score function, denoising score matching,
Langevin dynamics, the SDE/ODE bridge, the
autoregressive chain rule, teacher forcing, the causal
mask, the transformer forward pass, likelihood
evaluation, sampling, context budgets, and the
representation differences between the two cultures.

Objectives: after this lesson the learner can write
the score of a mixture, compute a denoising score
matching target, run Langevin steps by hand, state
the probability-flow ODE, factor a sequence with the
chain rule, compute teacher-forced and free-run
losses, apply a causal mask, do a full tiny
transformer forward pass by hand, compute NLL and
perplexity, compare sampling schemes, do the
attention memory arithmetic, and argue when each
culture wins.

Dependencies: U07 (diffusion, the score reading of
eps_hat from U08-C07), U01 (densities), R25
(Gaussian facts), R28 (sequence modelling basics,
the P13 local bridge), R29 (attention basics, the
P14 local bridge).

## How to read this lesson

Mechanisms A to D, each with shells 0 to 10, then
one section per leaf concept with the full contract.
Figures carry one claim each. The audit table lives
in visual_audit.md.

The running toys. Score toy: p(x) = 0.5 N(-1.5,
0.25) + 0.5 N(1.5, 0.25), a symmetric two-mode
density. AR toy: vocabulary {a, b, c, d} (ids 0..3),
bigram matrix P (rows: from-token, cols: to-token),
sequence [a, b, c] = [0, 1, 2]. Transformer toy: d
= 4, one head, identity QKV, token embeddings E and
positional rows P as defined in C08. Logits [2.0,
1.0, 0.5, 0.1] for the sampling sections.

---

## Mechanism A, learning to point uphill

Shell 0. The question: the density p(x) is
unknown, but suppose you could learn its gradient
field, the score s(x) = grad log p(x). Could you
sample without ever writing p(x) down? What
would change if you learned p(x) instead? The
observable result that would change: learning p
needs the normalizer, learning s dodges it.

Shell 1. The toy: the two-mode mixture. s(0) =
0, s(1.0) = 1.999926, s(1.5) = 0, s(-1.0) =
-1.999926. The score is zero at the modes and
at the valley, positive left of the right mode
(pointing right, uphill), negative right of it.
Denoising score matching target at x_tilde =
1.7 from clean x = 1.5 with noise var 0.04:
target = -5.0. Langevin, 5 steps from 0: 0.0,
0.039759, 0.060272, 0.354732, 0.608576,
0.617063, drifting toward the right mode.

Shell 2. Objects: s(x) = grad_x log p(x), the
score. The noise-perturbed density p_t. The
DSM target (x - x_tilde)/sigma^2. The
Langevin step size a. Units: s in 1/x.

Shell 3. One rule: match the score of the
noise-perturbed density, then follow it
uphill with noise (Langevin). Justified
assumption: perturbing with Gaussian noise
makes the score learnable by regression,
without noise, low-density regions have no
training signal.

Shell 4. Derive the algorithm: DSM loss =
E[||s_theta(x_tilde) - (x - x_tilde)/sigma^2
||^2]. Langevin: x <- x + (a/2) s(x) +
sqrt(a) z. Probability-flow ODE (forward
time): dx = -0.5 beta(t) s(x, t) dt.
Sampling integrates it backward: with
h > 0, x <- x + 0.5 beta(t) s(x, t) h.

Shell 5. Check the invariant: the DSM
target at the clean point is exact: (1.5 -
1.7)/0.04 = -5.0. The loss for s_hat = -4
is 1.0. At the mixture modes the analytic
score is 0 (verified < 1e-6).

Shell 6. Change ONE factor: Langevin step
0.1 -> 0.5. Predict: more crossings, mean
closer to 0. Measured over 20000 steps:
crossings 56 -> 414, frac(x > 0) 0.56 ->
0.517, mean 0.161 -> 0.043. Controls: same
seed, same density.

Shell 7. Counterexample: run Langevin on
the unperturbed mixture with a tiny step
from x = 0. The score at 0 is exactly 0,
so the chain sits in the valley and never
finds a mode. Score matching needs the
noise perturbation, the valley has no
signal without it.

Shell 8. Compare: Langevin (SDE, 20000
steps, 56 crossings) versus the ODE
(sampling step from 1.0: 1.00999963,
toward the mode, deterministic, ~50
evals). Equal goal: samples from p. The
ODE is cheaper and reproducible,
Langevin explores more.

Shell 9. Falsifiable extension: anneal
the noise level during Langevin (high to
low) and count crossings. Predict:
annealing beats any fixed step on the
crossing count. This is the historical
bridge to diffusion schedules.

Shell 10. Production: the score net is
the DDPM epsilon net in disguise (U08-C07:
s_hat = -eps_hat/sqrt(1 - ab_t)). The
stakeholder decision: one trained net
serves DDPM sampling, DDIM sampling, and
score-based sampling. The sampler is the
product choice, not the net.

### C01, the score definition

Motivating question: what is the score,
and why is it easier than the density?

Start from zero. For a density p(x), the
score is s(x) = d/dx log p(x). It points
uphill: where p rises, s > 0, at modes, s
= 0. It never needs the normalizer: log p
= log p_unnorm - log Z, and d/dx kills
log Z. Two densities differing only by Z
share the score.

Mental model: the score is a vector field
of arrows. Follow the arrows and you
climb to a mode. The arrows do not tell
you the height (the density value), only
the slope.

Variables: x scalar in the toy. s(x) in
1/x units.

Derivation for the mixture. p(x) = 0.5
N(x, -1.5, 0.25) + 0.5 N(x, 1.5, 0.25).
d/dx log p = (0.5 N_1' + 0.5 N_2')/(0.5
N_1 + 0.5 N_2), N_i' = -N_i (x - mu_i)/
0.25. So s(x) = -(w_1 (x + 1.5) + w_2 (x
- 1.5))/0.25 with w_i the posterior
component weights.

Computed numbers. s(0) = 0 (symmetry).
s(1.0) = 1.999926: left of the right
mode, pointing right. s(1.5) ~ 0 (the
mode, < 1e-6). s(-1.0) = -1.999926
(mirror). p(0) = 0.008864: the valley is
deep, which is why mixing is slow.

Code: score_mix in compute_run5b.py.

Checks. (1) s(0) = 0 exactly by symmetry.
(2) s(1.5) ~ 0 (mode). (3) Antisymmetry:
s(-1.0) = -s(1.0).

Costs. O(1) per evaluation on the toy.

Alternatives. Learn p directly: needs the
normalizer (intractable in general).
Learn the energy E = -log p_unnorm: same
gradients, same sampling, different
parameterization. Selection boundary:
score when only gradients are needed
(sampling), density when probabilities
are needed (likelihood).

Failure case. s(x) = 0 at the valley x
= 0 too: following the score from exactly
0 goes nowhere. Stationary points are
not all modes. Langevin noise rescues
this, deterministically following the
score does not.

Research extension. The score of a
high-dimensional image density cannot be
plotted. Open: visualize 2-D slices of a
trained score field and check for
spurious attractors. Falsifiable: a
slice through two training images shows
an attractor at neither.

### C02, denoising score matching

Motivating question: how do you learn the
score when you only have samples?

Start from zero. Perturb: x_tilde = x +
sigma z, z ~ N(0, 1). The score of the
perturbed density at x_tilde has a closed
form given the clean x: (x - x_tilde)/
sigma^2. Proof sketch: p(x_tilde | x) is
Gaussian, and E over the posterior of x
gives the marginal score, the
single-sample version is exact for the
conditional. Train s_theta(x_tilde) to
match it by squared error. No normalizer,
no MCMC, just regression.

Mental model: add noise, then learn to
point from the noisy point back toward
the clean one. The target is the arrow
from x_tilde to x, scaled by 1/sigma^2.

Variables: sigma the noise level, x the
clean sample, x_tilde the noisy point.

Computed numbers. x = 1.5, x_tilde =
1.7, sigma^2 = 0.04: target = (1.5 -
1.7)/0.04 = -5.0. Loss for s_hat = -4:
(-4 + 5)^2 = 1.0. Multiple noise levels:
average the loss over sigma in a set,
small sigma gives sharp targets, large
sigma gives coverage of the valley.

Code: target = (x - x_tilde) / sigma**2.

Checks. (1) Target points from noisy to
clean (sign). (2) At x_tilde = x the
target is 0. (3) Scaling: halving sigma
quadruples the target magnitude.

Costs. One extra noise draw per sample,
the loss is O(d).

Alternatives. Sliced score matching:
random projections, no perturbation
needed, higher variance. Explicit score
matching: needs the true score,
unavailable. Selection boundary: DSM
when Gaussian perturbation is natural
(diffusion), sliced when it is not.

Failure case. sigma too small: the
perturbed density equals the clean one,
valley regions get no samples, the
learned score is garbage there (shell 7
revisited). sigma too large: everything
is one blur, fine detail lost. Use a
ladder of sigmas.

Research extension. The single-sample
target is unbiased for the conditional
score, not the marginal. Measure the
bias-variance of the learned score at
the valley x = 0 vs the mode x = 1.5 as
sigma varies. Falsifiable: valley bias
falls as sigma grows, mode bias rises.

### C03, Langevin dynamics

Motivating question: you have the score.
How do you get samples?

Start from zero. Gradient ascent on log
p would climb to a mode and stick.
Langevin adds the right noise: x <- x +
(a/2) s(x) + sqrt(a) z. The drift climbs,
the noise explores, and the stationary
distribution is p (for small a). It is
MCMC with the score as the driver.

Mental model: a drunk hiker who mostly
walks uphill. The uphill bias finds the
modes, the drunkenness visits them in
proportion to their mass.

Variables: a the step size, z ~ N(0, 1).

Computed numbers. a = 0.1, start 0,
seed 0: 0.0 -> 0.039759 -> 0.060272 ->
0.354732 -> 0.608576 -> 0.617063. The
chain leaves the valley and climbs toward
1.5. Mixing over 20000 steps, seed 1: a
= 0.1 gives mean 0.1610, std 1.5873
(true 1.5811), frac(x > 0) = 0.56, only
56 valley crossings. a = 0.5: mean
0.0428, std 1.6385, frac 0.517, 414
crossings. Small steps mix slowly across
the valley: the measured mean 0.161 is
not 0 because 20000 steps are not enough
at a = 0.1. This is the honest
predicted-vs-measured gap, and it is the
reason diffusion anneals the noise.

Code: the loop in compute_run5b.py.

Checks. (1) Stationary std matches the
mixture std 1.5811 within noise. (2)
Mean -> 0 as crossings grow. (3) At a =
0.5 the chain still does not diverge
(the score is contractive here).

Costs. Thousands of steps per
independent sample on multimodal
targets. Each step is one score eval.

Alternatives. HMC: gradient + momentum,
better mixing, needs tuning. Annealed
Langevin: the practical fix, and the
ancestor of diffusion schedules.
Selection boundary: Langevin for
unimodal or lightly multimodal targets,
annealing when valleys are deep.

Failure case. a too large: the Euler
discretization diverges (the chain
explodes instead of sampling). The toy
tolerates 0.5, it does not tolerate
every a. Tune a or use Metropolis
adjustment.

Research extension. Count crossings vs a
on the toy and find the a that maximizes
crossings per eval. Falsifiable: the
optimum is interior (too small crawls,
too large rejects).

### C04, the SDE/ODE bridge

Motivating question: what continuous-time
object sits behind DDPM, DDIM, and
Langevin?

Start from zero. The DDPM forward chain
is a discretization of an SDE: dx =
f(x, t) dt + g(t) dw. Every such SDE has
a deterministic twin, the probability-
flow ODE, with the same marginals:
dx = [f(x, t) - 0.5 g(t)^2 s(x, t)] dt.
DDIM with eta = 0 discretizes the ODE,
eta = 1 discretizes the SDE, Langevin is
the SDE at fixed noise.

Mental model: one river (the SDE), one
canal (the ODE). Same water levels at
every bridge (the marginals), different
journeys. The score steers both.

Variables: f drift, g diffusion
coefficient, w Brownian motion, s(x, t)
the time-dependent score. Sign warning:
the ODE above runs in forward time
(data to noise). Sampling runs it
backward (noise to data), which flips
the sign of the update.

Computed numbers. VP toy: f = 0, g =
sqrt(beta), beta = 0.1. Sampling step
with h = 0.1: x <- x + 0.5 0.1 s(x)
0.1. From x = 1.0: s(1.0) = 1.999926,
dx = +0.00999963, x = 1.00999963:
toward the mode at 1.5, uphill on log
p. Deterministic, no z. Eval budget:
Langevin needed 20000 steps for 56
crossings, the ODE needs ~50 evals for
a full trajectory (protocol counts, not
a benchmark).

Code: x = x + 0.5 * beta * score(x) * h
(h > 0 steps from noise toward data).

Checks. (1) h -> 0 recovers the
continuous path. (2) The sampling
update moves uphill on log p: from 1.0
it goes to 1.00999963, from -1.0 to
-1.00999963. (3) eta = 0 DDIM matches
the ODE discretization structurally.

Costs. ODE: tens of evals, no noise.
SDE: thousands, with noise.

Alternatives. Higher-order ODE solvers:
fewer evals, more code. Exact SDE
simulation: unbiased, expensive.
Selection boundary: ODE for speed and
reproducibility, SDE when stochasticity
is the product (diversity).

Failure case. Large dt on a stiff score
(the valley walls): Euler overshoots and
oscillates. The toy score is gentle,
image scores are not. Adaptive step
sizes exist for this reason.

Research extension. Measure the
discretization error of Euler vs a
second-order solver on the toy ODE at
fixed eval budget. Falsifiable: the
second-order solver wins at every
budget.

---

## Mechanism B, predict next, one token at a time

Shell 0. The question: the score culture
learns a vector field. What if instead you
factor the joint distribution into a
sequence of next-token predictions? What
would change if the factorization order
changed? The observable result: the same
joint, written in a different order, needs
different conditionals.

Shell 1. The toy: bigram P, sequence [a,
b, c]. NLL = -log2 P(b|a) - log2 P(c|b)
= -log2(0.6) - log2(0.5) = 0.73697 + 1.0
= 1.73697 bits. Perplexity = 2^{0.86848}
= 1.82574. Teacher-forced: 0.8685
bits/token. Free-run expected: 1.6338
bits/token. Exposure gap: 0.7653
bits/token.

Shell 2. Objects: the vocab {a, b, c, d},
the bigram matrix P (4x4, rows sum to 1),
the chain-rule factors p(x_i | x_<i).
Units: bits.

Shell 3. One rule: p(x_1..x_n) =
prod_i p(x_i | x_<i). Any order works
mathematically, the natural order (left
to right for text) is a modelling
choice. Teacher forcing trains each
factor on true prefixes. Justified
assumption: the factors share parameters
(one net), so the factorization is a
training protocol, not n separate
models.

Shell 4. Derive the algorithm: NLL = -sum
log p(x_i | x_<i), perplexity =
2^{NLL/n}. Free-run: sample x_1 ~ p(.),
then x_2 ~ p(. | x_1_hat). The causal
mask enforces the conditioning in one
parallel forward pass.

Shell 5. Check the invariant: rows of P
sum to 1 (0.1+0.6+0.2+0.1 = 1.0 etc.).
The chain rule with the true conditionals
recovers the joint exactly (C05).

Shell 6. Change ONE factor: train on
true prefixes, test on model prefixes.
Predict: loss rises. Measured: 0.8685 ->
1.6338 bits/token, gap 0.7653. Controls:
same P, same sequence.

Shell 7. Counterexample: remove the
causal mask. Position 1 sees x_2 during
training, the loss collapses to ~0, and
generation fails completely: the net
learned to copy the future, which does
not exist at test time. The mask is
load-bearing.

Shell 8. Compare: teacher forcing
(parallel, fast, biased) vs free-run
training (sequential, slow, unbiased).
Equal budget: teacher forcing wins on
wall-clock, scheduled sampling
interpolates.

Shell 9. Falsifiable extension: measure
the exposure gap vs sequence length on
the bigram toy. Predict: the gap grows
with n (errors compound).

Shell 10. Production: teacher forcing is
how every AR LM trains, the exposure
gap is why decoding strategies (C10) and
evals (U10-C11) matter. The stakeholder
decision: the training loss is not the
product metric, sample quality is.

### C05, the autoregressive chain rule

Motivating question: how do you write a
joint distribution as a training
objective?

Start from zero. p(a, b, c) = p(a) p(b |
a) p(c | a, b). Each factor is a
classifier over the vocab given the
prefix. The log-likelihood is a sum, so
training is n small classification
problems, not one giant joint problem.
The bigram toy truncates the history to
one token: p(c | a, b) = P(c | b).

Mental model: the chain rule turns
"write a sequence" into "always guess
the next token". The model never sees
the whole sequence at once during
training, it sees every prefix.

Variables: x_i tokens, P the 4x4 bigram
matrix, n the sequence length.

Computed numbers. P(b|a) = 0.6, P(c|b)
= 0.5. NLL = -log2(0.6) - log2(0.5) =
0.73697 + 1.0 = 1.73697 bits (the first
token a has no factor in the bigram
model, a unigram p(a) would add one).
Perplexity = 2^{1.73697/2} = 1.82574:
the model is as confused as a 1.83-sided
die per token.

Code:

```python
nll = -np.log2(P[0, 1]) - np.log2(P[1, 2])
ppl = 2.0 ** (nll / 2.0)
```

Checks. (1) Rows sum to 1. (2) NLL >= 0.
(3) Perplexity in [1, 4] for a 4-word
vocab.

Costs. O(n V) per sequence for the
softmaxes, the bigram lookup is O(1).

Alternatives. Full joint table: exact,
needs V^n entries (impossible). Markov
order k: 4^{k+1} params, the classical
compromise. Selection boundary: neural
AR when histories are long, low-order
Markov when data is scarce.

Failure case. Factorizing in a bad
order (right to left for text): valid
math, terrible modelling. The order is
a modelling choice, and the natural
order carries the inductive bias.

Research extension. Measure perplexity
vs Markov order k = 1..4 on a real
corpus toy. Falsifiable: gains diminish
and the neural model beats every fixed
k past some data size.

### C06, teacher forcing

Motivating question: why train on truth
and test on your own guesses?

Start from zero. Teacher forcing feeds
the true prefix at every step during
training: the loss for x_2 uses the true
x_1. This parallelizes (all positions at
once with the mask) and stabilizes
(gradients do not pass through sampling).
At test time the model feeds its own
x_1_hat, which may be wrong, then x_2 is
conditioned on a prefix the model never
saw in training. That mismatch is
exposure bias.

Mental model: training wheels. The
model learns to ride while you hold the
seat. At test time you let go, and the
first wobble compounds.

Variables: x_i true tokens, x_i_hat
sampled tokens.

Computed numbers. Teacher-forced NLL on
[a, b, c]: 1.7370 bits, 0.8685/token.
Free-run expected NLL: E over x_1 ~
P(.|a) of [-log2 P(x_1|a) - log2
P(c|x_1)] = 0.1(3.3219+2.3219) +
0.6(0.7370+1.0) + 0.2(2.3219+3.3219) +
0.1(3.3219+2.0) = 3.2675 bits,
1.6338/token. Exposure gap: 0.7653
bits/token. When the model samples d
(prob 0.1), the next factor is uniform
and costs 2.0 bits for c.

Code: the expectation loop in
compute_run5b.py.

Checks. (1) Free-run >= teacher-forced
here (0.7653 gap). (2) The gap comes
from wrong-prefix mass: x_1 = c and d
are the expensive cases. (3) Gap = 0
only if the model is perfect.

Costs. Teacher forcing: fully parallel.
Free-run: sequential, n steps.

Alternatives. Scheduled sampling: mix
true and sampled prefixes during
training. Professor forcing: match
hidden dynamics. Selection boundary:
teacher forcing for pretraining speed,
mixing when the gap bites (long
generations).

Failure case. Pure free-run training
from scratch: slow, high-variance, often
diverges early. The curriculum is:
teacher-force first, then correct.

Research extension. The gap-vs-length
curve (shell 9). Falsifiable: linear or
worse growth on the bigram toy as n
grows.

### C07, the causal mask

Motivating question: how does one forward
pass train all positions without
leaking the future?

Start from zero. Attention lets position
i read positions 0..n-1. The causal mask
sets scores for j > i to -inf before the
softmax, so position i attends only to
0..i. One forward pass then computes all
n next-token predictions in parallel,
each with exactly its training-time
prefix.

Mental model: blinders. Each position
sees its past, never its future. The
mask is what makes parallel training
honest.

Variables: the 3x3 mask (lower
triangular ones), scores QK^T/sqrt(d).

Computed numbers. Q = K = [[1,0],[0,1],
[1,1]]. Scores/sqrt(2): [[0.7071, 0,
0.7071],[0, 0.7071, 0.7071],[0.7071,
0.7071, 1.4142]]. Masked softmax rows:
[1, 0, 0], [0.3302, 0.6698, 0],
[0.2483, 0.2483, 0.5035]. Position 0
sees only itself, position 2 sees all
three with weights summing to 1.

Code: masked = where(tril, scores,
-1e9), softmax rows.

Checks. (1) Row 0 = [1, 0, 0] exactly.
(2) Rows sum to 1. (3) Upper triangle
exactly 0.

Costs. The mask is free (a constant).
Attention itself is O(n^2), see C11.

Alternatives. PrefixLM mask: bidirectional
on a prefix, causal after. No mask:
bidirectional (BERT-style), not for
generation. Selection boundary: causal
for AR generation, bidirectional for
understanding.

Failure case. Mask off by one (allow j
<= i+1): position i sees x_{i+1}, the
loss for predicting x_{i+1} collapses,
and eval perplexity looks miraculous
until generation. Test: row 0 must be
[1, 0, 0], not [0.5, 0.5, 0].

Research extension. Measure the loss
drop from a one-off mask bug on the toy
transformer. Falsifiable: the bug
always helps training loss and always
hurts generation.

---

## Mechanism C, the transformer

Shell 0. The question: the chain rule
needs p(x_i | x_<i) for long histories.
What architecture computes all of them
in parallel without forgetting the
order? What would change if positions
had no encoding? The observable result:
without position info, "a b" and "b a"
get the same treatment.

Shell 1. The toy: d = 4, one head,
identity QKV. E rows: a = [1,0,1,0], b
= [0,1,0,1], c = [1,1,0,0]. Pos rows:
0.1 i [1,1,1,1]. X = E + Pos. Output
rows: [1,0,1,0], [0.3214, 0.8294,
0.3214, 0.8294], [0.8617, 0.8964,
0.3581, 0.3928]. Row 0 equals X row 0
(the mask at work). With-pos vs no-pos
row 1 differs by 0.0983.

Shell 2. Objects: E token embeddings, P
positional encodings, W_Q/W_K/W_V
projections, A attention weights, O the
output. Shapes: (n, d) throughout with
n = 3, d = 4.

Shell 3. One rule: attention is a
data-dependent weighted average, the
causal mask restricts the average to
the past, the positional encoding tells
identical tokens apart by position.
Justified assumption: identity QKV keeps
the toy hand-checkable, real nets learn
the projections.

Shell 4. Derive the algorithm: Q = X
W_Q, K = X W_K, V = X W_V, S = QK^T /
sqrt(d), A = softmax(mask(S)), O = A V.
Then: residual, norm, feedforward (stated,
not computed on the toy).

Shell 5. Check the invariant: row 0 of
O equals X row 0 exactly (only
self-attention allowed). With-pos and
no-pos outputs differ (positions
matter).

Shell 6. Change ONE factor: remove P.
Predict: output row 1 changes.
Measured: max |diff| 0.0983. The
attention weights also change because X
changed.

Shell 7. Counterexample: W_Q = W_K = 0.
All scores 0, attention uniform over the
past, the net cannot distinguish
positions by content. Content-based
addressing needs nonzero projections.

Shell 8. Compare: transformer vs RNN on
the toy task. The transformer is
parallel across n, the RNN is
sequential. Equal params: the
transformer trains faster, the RNN
generates with O(1) memory per step
(the transformer needs the KV cache,
U10-C02).

Shell 9. Falsifiable extension: train
the toy transformer (one head, d = 4)
on bigram sequences and measure next-
token accuracy with and without P.
Predict: without P, accuracy drops on
any task where order matters.

Shell 10. Production: the transformer
is the AR workhorse, its O(n^2)
attention is the serving bottleneck
(C11). The stakeholder decision: context
length is a product spec with a memory
price tag.

### C08, the transformer

Motivating question: what computes
p(x_i | x_<i) in one parallel pass?

Start from zero. Embed each token: X =
E + P, shape (3, 4). Project: Q, K, V
= X W (identity here, so Q = K = V =
X). Score: S = XX^T/2. Mask causal,
softmax rows: A. Output: O = A X.
Position 0 output = X row 0 (it can
only see itself). Position 1 mixes
rows 0-1 with [0.3302, 0.6698].
Position 2 mixes all three with
[0.2483, 0.2483, 0.5035]. Then (not on
the toy): residual add, layer norm,
feedforward, repeat per layer, a final
linear + softmax gives next-token
probs.

Mental model: attention is a
content-addressable memory read. The
query asks "which past positions matter
for predicting next", the keys answer,
the values deliver. The mask enforces
honesty, the positions break symmetry.

Variables and shapes: n = 3, d = 4, one
head. E (3,4), P (3,4), W (4,4), A
(3,3), O (3,4).

Computed numbers. O rows: [1, 0, 1, 0],
[0.3214, 0.8294, 0.3214, 0.8294],
[0.8617, 0.8964, 0.3581, 0.3928]. Row 0
== X row 0 exactly. Removing P changes
row 1 by 0.0983 max.

Code: the block in compute_run5b.py.

Checks. (1) O[0] == X[0]. (2) A rows sum
to 1. (3) With-pos != no-pos.

Costs. Attention: O(n^2 d). Projections:
O(n d^2). The toy: trivial.

Alternatives. RNN/LSTM: O(n d^2)
sequential, O(d) state. State-space
models (W12L54 title): subquadratic,
not computed here. Selection boundary:
transformer when parallelism and long
range matter, alternatives when memory
per step must be O(1).

Failure case. Forgetting the 1/sqrt(d)
scale: with large d the softmax
saturates, gradients vanish, training
stalls. The toy d = 4 hides this, d =
64 does not.

Research extension. The single-head toy
cannot do induction ("a b ... a -> b").
Two heads can. Falsifiable: a two-head
toy learns the copy task, one head
cannot.

---

## Mechanism D, likelihood, sampling, budgets, and the two cultures

Shell 0. The question: both cultures
generate. How do you score them, sample
from them, and pay for them? What would
change if likelihood were the only
metric? The observable result: the
score culture has no likelihood, the AR
culture has exact likelihood but pays
O(n^2).

Shell 1. The toy: NLL 1.73697 bits,
perplexity 1.82574. Sampling logits
[2.0, 1.0, 0.5, 0.1]: T = 0.5 ->
[0.8282, 0.1121, 0.0412, 0.0185],
entropy 0.8754 bits, T = 2.0 ->
[0.4056, 0.2460, 0.1916, 0.1569],
entropy 1.9017 bits. Top-p 0.9 keeps
[0, 1], renorm [0.7311, 0.2689].
Attention memory: n = 1024 -> 0.60 GB,
n = 4096 -> 9.66 GB (12 heads, 12
layers, fp32).

Shell 2. Objects: NLL, perplexity,
temperature T, top-p threshold, the
attention matrix (n, n) per head-layer.
Units: bits, bytes.

Shell 3. One rule: AR likelihood is a
sum of logs (exact, comparable across
models on the same vocab). Sampling is a
separate choice from training: the same
logits give greedy, temperature, top-k,
top-p. Cost is quadratic in n for full
attention.

Shell 4. Derive the algorithm:
perplexity = 2^{NLL/n}. Temperature:
softmax(z/T). Top-p: sort desc, keep
the smallest set with cumsum >= p,
renormalize. Memory: heads x layers x
n^2 x 4 bytes.

Shell 5. Check the invariant:
perplexity in [1, V]. Top-p kept set is
nonempty. Memory formula: 12 12 1024^2
4 = 603979776 bytes = 0.60 GB.

Shell 6. Change ONE factor: T 1.0 ->
0.5. Predict: sharper, lower entropy.
Measured: entropy 1.6174 -> 0.8754
bits, pmax 0.5745 -> 0.8282.

Shell 7. Counterexample: compare
perplexities across different vocabs.
Meaningless: the event spaces differ.
Perplexity is comparable only on
identical tokenization.

Shell 8. Compare: score culture (no
likelihood, parallel-ish sampling via
ODE, needs noise ladder) vs AR culture
(exact likelihood, sequential sampling,
O(n^2) memory). Neither dominates, the
modality and the metric choose.

Shell 9. Falsifiable extension: measure
perplexity vs n on the bigram toy with
longer sequences. Predict: flat (the
bigram has no long range), a transformer
would beat it past some n on real text.

Shell 10. Production: context length is
priced in GB (C11), sampling knobs (T,
top-p) are product knobs (U10-C01). The
stakeholder decision: ship the (n, T,
top-p) triple that passes the quality
gate at acceptable cost.

### C09, likelihood evaluation

Motivating question: how do you score
an AR model?

Start from zero. NLL = -sum_i log2
p(x_i | x_<i). Perplexity = 2^{NLL/n}:
"the effective branching factor". On
[a, b, c]: NLL = 1.73697 bits,
perplexity 1.82574. Lower is better,
bounded below by 1 (perfect) and above
by V = 4 (uniform).

Mental model: perplexity translates
bits into "how many options the model
seriously considers". 1.83 means nearly
decided, 4 means guessing.

Variables: n tokens, V vocab size.

Computed numbers: 1.73697 bits total,
0.86848 bits/token, perplexity 1.82574.

Checks. (1) 1 <= ppl <= 4. (2) Uniform
model gives ppl = 4. (3) Same vocab
required for comparisons.

Costs. One forward pass per sequence
(with the mask).

Alternatives. Bits-per-byte: vocab-free
comparison. Task accuracy: the product
metric. Selection boundary: perplexity
for model selection on fixed tokenizer,
task metrics for ship decisions.

Failure case. Reporting perplexity on
different tokenizers side by side. The
numbers are incomparable, bits-per-byte
repairs it.

Research extension. Perplexity vs
downstream accuracy correlation on a
benchmark suite. Falsifiable: the
correlation is strong but imperfect,
some low-perplexity models fail tasks.

### C10, sampling

Motivating question: the logits are
fixed. How many samplers can you build?

Start from zero. Logits z = [2.0, 1.0,
0.5, 0.1]. Greedy: argmax = a.
Temperature: softmax(z/T). T = 0.5:
[0.8282, 0.1121, 0.0412, 0.0185],
entropy 0.8754. T = 1: [0.5745, 0.2114,
0.1282, 0.0859], entropy 1.6174. T =
2: [0.4056, 0.2460, 0.1916, 0.1569],
entropy 1.9017. Top-k = 2: keep a, b,
renormalize. Top-p = 0.9: sorted cumsum
0.5745, 0.7859, 0.9141: keep [a, b]
(the first index crossing 0.9 is b...
check: cumsum after a = 0.5745 < 0.9,
after b = 0.7859 < 0.9, after c =
0.9141 >= 0.9: keep [a, b, c]?
The audit script keeps indices with
cumsum <= 0.9 then ensures nonempty:
kept [0, 1], renorm [0.7311, 0.2689].
Note the convention choice: keep-while-
cumsum-below-p drops c. Conventions
vary, the lesson states this one.

Mental model: temperature sets the
adventure level, top-p/top-k cut the
long tail of bad tokens. Same logits,
different products.

Variables: T > 0, k int, p in (0, 1].

Computed numbers: the three
distributions and entropies above,
top-p 0.9 -> [0.7311, 0.2689, 0, 0].

Code: softmax(z/T), the top-p block in
compute_run5b.py.

Checks. (1) Probs sum to 1 after every
op. (2) T -> 0 recovers greedy. (3)
Entropy rises in T.

Costs. O(V log V) for the sort in
top-p, O(V) otherwise.

Alternatives. Beam search: explores k
paths, deterministic-ish. Typical
sampling: information-theoretic
truncation. Selection boundary:
temperature + top-p for chat, greedy
for factual, beam for structured
outputs.

Failure case. T huge with no truncation:
the tail (d at 0.1569) produces
incoherence. Truncation exists because
the tail is where the model is
clueless.

Research extension. Measure the
quality/diversity curve over (T, p) on
a fixed eval. Falsifiable: the best (T,
p) for quality differs from the best
for diversity, mirroring U08-C10.

### C11, context budgets

Motivating question: what does a long
context cost?

Start from zero. One attention matrix
per head per layer: n^2 entries. fp32:
4 bytes each. Toy model: 12 heads, 12
layers. n = 1024: 12 12 1024^2 4 =
603979776 bytes = 0.60 GB. n = 4096:
16x = 9663676416 bytes = 9.66 GB.
Quadrupling n multiplies memory by 16.

Mental model: context is rented by the
square. Twice the context, four times
the attention memory.

Variables: n tokens, h heads, L
layers, b bytes per entry.

Computed numbers: 0.60 GB at 1024,
9.66 GB at 4096. The KV cache (U10-C02)
adds 2 L h d n b bytes on top, linear
in n.

Code: bytes_ = L * h * n * n * 4.

Checks. (1) Ratio 4096/1024 = 16
exactly. (2) Units: bytes. (3) This is
the attention matrix only, not the
weights or the cache.

Costs. This section is the cost model.

Alternatives. Flash attention:
recomputes instead of storing (less
memory, same FLOPs). Sliding window:
O(n w). Linear attention: O(n).
Selection boundary: full attention
when quality needs it and n is small,
alternatives when n is the product.

Failure case. Budgeting weights only
and forgetting activations: at n =
4096 the attention matrix (9.66 GB)
can exceed the weights. Budget all
three: weights, cache, attention.

Research extension. Measure the real
memory vs n on one GPU and find where
the quadratic term dominates. 
Falsifiable: the crossover n is
measurable and model-dependent.

### C12, representation differences

Motivating question: when does each
culture win?

Start from zero. The score culture
represents p via its gradient field.
Strengths: no normalizer, natural for
continuous data, one net serves many
samplers (U08). Costs: no exact
likelihood, sampling needs many steps
or an ODE solver, training needs the
noise ladder. The AR culture represents
p via ordered conditionals. Strengths:
exact likelihood (1.73697 bits on the
toy), simple training (teacher
forcing), any order-free generation via
the mask. Costs: fixed order, O(n^2)
memory, sequential sampling, exposure
bias (0.7653 bits/token on the toy).

Mental model: the score culture learns
the score field, the AR culture learns
the tour. Landscapes suit images
(order-free, continuous). Tours suit
text (ordered, discrete).

The comparison table:

| axis | score | AR |
|---|---|---|
| likelihood | none exact | exact sum of logs |
| sampling | parallel-ish (ODE) | sequential |
| order | none needed | fixed, load-bearing |
| training signal | denoising regression | next-token CE |
| failure mode | slow mixing (56 crossings) | exposure bias (0.77 b/tok) |
| data fit | continuous | discrete sequences |

Computed anchors: the crossing count
(56 at a = 0.1), the exposure gap
(0.7653), the perplexity (1.82574),
the memory (0.60 GB at n = 1024).

Checks. Every number in the table is
computed above.

Alternatives. Hybrids: AR in a learned
latent space (VQ-VAE + AR prior,
U06-C07), diffusion on discrete tokens.
Selection boundary: images -> score/
diffusion, text -> AR, latents -> both.

Failure case. Picking by fashion: the
cultures solve different problems.
Match the representation to the data
and the metric, not to the trend.

Research extension. A hybrid on the
toy: VQ the mixture into 4 codes, AR
over codes, compare likelihood to the
score model. Falsifiable: the hybrid
gets exact likelihood but worse samples
than the score model at equal budget.

## Not yet understood (for the next builder)

1. SDE solver error: the ODE Euler
step is one step, error vs dt
unmeasured.
2. The transformer never trains: one
forward pass only, no gradient, no
learned weights.
3. Exposure gap measured on the bigram
toy only, scheduled sampling not
implemented.
4. Annealed Langevin (shell 9) not run,
the crossing-count prediction is
untested.
5. State-space models (W12L54): title
only, no math.

## Lesson exercises (questions. Keys in lessons/u09/keys.md)

E01. Write s(x) for the mixture and
compute s(1.0) by hand. Verify
1.999926.
E02. Explain why the score needs no
normalizer. Give the one-line algebra.
E03. Compute the DSM target for x =
-1.5, x_tilde = -1.9, sigma^2 = 0.04.
E04. One Langevin step from x = 0.6, a
= 0.1, z = 0.5: compute x_1 by hand.
E05. Explain the 20000-step mean 0.161
at a = 0.1: is the sampler wrong? Use
the crossing count.
E06. ODE sampling step from x = -1.0, h =
0.1, beta = 0.1: compute x_1. Verify
-1.00999963 and state the direction.
E07. Compute the NLL of [a, b, c] in
bits. Verify 1.73697.
E08. Compute perplexity. Verify
1.82574. State the bounds.
E09. Compute the free-run expected NLL
for the second token only. Compare to
the two-token version: why do they
differ?
E10. Write the 3x3 causal mask. What
does row 1 allow?
E11. Compute the masked attention row
2 by hand. Verify [0.2483, 0.2483,
0.5035].
E12. In the transformer toy, prove O[0]
== X[0] without computing A.
E13. Remove P: which numbers change
(X, A, O) and which stay (mask)?
E14. Temperature: compute softmax(z/0.5)
for z = [2.0, 1.0, 0.5, 0.1]. Verify
the first entry 0.8282.
E15. Top-p 0.9: list the kept indices
under the lesson's convention. Verify
[0, 1].
E16. Compute the attention bytes for n
= 2048, 12 heads, 12 layers, fp32.
E17. A colleague compares perplexity
3.2 (vocab 4) with 15.0 (vocab 50000).
Name the error and the repair.
E18. Greedy vs T = 2.0: which has
higher entropy? Compute both.
E19. The mask bug (allow j <= i+1):
compute row 0 of A and explain the
consequence.
E20. Pick score vs AR for (a) 64x64
images, (b) code completion. Justify
each in two sentences with one toy
number.

## Deep oral ladders (questions. Keys in lessons/u09/keys.md)

L01. Score to samples: define the
score, derive the DSM target, compute
it, run 5 Langevin steps, diagnose
slow mixing, compare to the ODE,
design the annealing experiment.
L02. Chain rule to transformer: factor
the joint, compute NLL/perplexity,
quantify exposure bias, apply the mask,
do the forward pass, debug the mask
bug, compare to an RNN, design the
order-ablation.
L03. Sampling and budgets: define
temperature/top-p, compute the three
distributions, do the memory
arithmetic, debug the vocab-comparison
error, compare the two cultures,
design the hybrid experiment.
L04. The bridge: connect eps_hat to
s_hat (U08-C07 numbers), explain why
one net serves three samplers, run
the ODE step, critique "DDIM is not
score-based".
L05. End to end: from mixture samples
to text: name every assumption each
culture makes, break one per culture,
and predict the visible failure.

## Implementation and debug task

Implement score_mix, one Langevin step,
the causal masked attention, and the
tiny transformer forward from memory.
Then debug: a colleague's Langevin
chain on the mixture converges to one
mode and stays there over 2000 steps.
Name the cause (slow mixing across the
valley, 56 crossings per 20000 steps at
a = 0.1) and two fixes (larger a,
annealed noise ladder). Second bug:
their transformer's row 0 is [0.5, 0.5,
0]. Name the bug (mask off by one) and
the fix.

## Changed-constraint scenarios

S1. The mixture gains a third mode at
x = 0 (weight 0.2, var 0.25,
renormalized). Which lesson numbers
change (scores, DSM targets near 0,
crossing counts) and which do not (the
formulas, the mask, the AR numbers)?
Recompute s(0) qualitatively: zero or
not? Justify.
S2. The product needs 8192-token
contexts on a 12 GB card with the toy
transformer shape (12 heads, 12
layers, fp32). The attention matrix
alone needs 38.65 GB. Name three
levers (fewer layers/heads, fp16,
sliding window) and compute the memory
for one concrete feasible config.

## Research-critique question

"Exact likelihood makes AR models
strictly better than score-based models
for science." Attack: name what exact
likelihood buys (model comparison on
fixed vocab, no variational gap) and
what it does not buy (sample quality,
the right inductive bias for images),
then design the experiment that would
test the claim on one dataset with both
metrics.
