# Answer keys, lesson 08 (diffusion variants and implementation)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run5b.py.

## E01

x_{t-1} = sqrt(ab_{t-1}) x0_hat + sqrt(1 - ab_{t-1} -
sigma^2) eps_hat + sigma z. x0_hat = (x_t - sqrt(1 -
ab_t) eps_hat)/sqrt(ab_t). sigma = eta sqrt((1 -
ab_{t-1})/(1 - ab_t)) sqrt(1 - ab_t/ab_{t-1}). z ~
N(0, 1). eta in [0, 1].

## E02

sigma = 0.5 sqrt((1 - 0.440873)/(1 - 0.363563)) sqrt(1 -
0.363563/0.440873) = 0.5 0.937260 0.418708 = 0.19624903.
sigma^2 = 0.03851368.

## E03

Var(x_{t-1} | x_0) = (1 - ab_{t-1} - sigma^2) Var(eps) +
sigma^2 Var(z) = 1 - ab_{t-1}. The eps term carries the
old noise, the z term the new noise, they sum to the
DDPM marginal variance. At eta = 1, sigma^2 = ((1 -
ab_{t-1})/(1 - ab_t))(1 - ab_t/ab_{t-1}), which is the
posterior variance of the jump.

## E04

0.15405473 is the posterior variance of the 10-step
jump 100 -> 90: (1 - ab_90)/(1 - ab_100) (1 -
ab_100/ab_90). 0.01976684 is the single-step posterior
variance beta_tilde_100. Different objects: the
subsequence jump versus one DDPM step.

## E05

x0_hat = (1.60480905 - 0.7977685 0.4)/0.6029626 =
(1.60480905 - 0.3191074)/0.6029626 = 2.13230847. x_90 =
sqrt(0.440873) 2.13230847 + sqrt(0.559127) 0.4 =
0.663991 2.13230847 + 0.747744 0.4 = 1.41580734 +
0.2990976 = 1.71491474.

## E06

Predict: identical, max diff 0. Measured: max |diff| <
1e-12. Same x_T, same net, no noise term: the map is a
pure function.

## E07

Measured: max |diff| > 1e-6. The z draws differ by
seed, so the trajectories differ. One sentence: eta =
1 restores per-jump randomness while eta = 0 removes
it.

## E08

tau_late = [1, 20, 39, 55, 69, 80, 88, 95, 98, 100].
Strictly increasing: each entry exceeds the last.
Ends at 100. ab at tau_late: 0.999900 down to
0.363563, strictly decreasing.

## E09

u1: C_in = 32, C_out = 8, k = 3. Params = 32 8 9 + 8 =
2304 + 8 = 2312.

## E10

FLOPs = 2 C_in C_out k^2 H W = 2 8 16 9 4 4 = 36864.

## E11

d = 8, i = 0: ang = 50/1 = 50. sin(50) = -0.262375,
cos(50) = 0.964966. Entries 0, 1 verified.

## E12

sin(0) = 0, cos(0) = 1 for every frequency. t = 0
means no noise added, the embedding is the fixed
vector [0, 1, 0, 1, ...], the "clean" clock position.

## E13

w(t) = beta_t/(2 ab_t (1 - ab_t)). t=1: 1e-4/(2
0.9999 1e-4) = 0.50005. t=50: 0.0099495/(2 0.77718
0.22282) = 0.02873. t=100: 0.02/(2 0.36356 0.63644)
= 0.04322.

## E14

Overweight ratio = 1/w(t): t=1: 1/0.50005 = 2.00. t=50:
1/0.02873 = 34.81. t=100: 1/0.04322 = 23.14. The
uniform weight most overweights t = 50 (34.8x) and
least overweights t = 1 (2.0x). The ELBO weight's
minimum is where the uniform weight distorts most.

## E15

s_hat = -0.4/sqrt(1 - 0.77718008) = -0.4/0.4720356 =
-0.84738932.

## E16

x0_hat = (1.99917538 + 0.22281992 (-0.84738932))/
0.8815753 = (1.99917538 - 0.18881349)/0.8815753 =
1.81036189/0.8815753 = 2.05354466. Matches the eps
formula to 1e-12.

## E17

eps_tilde = 0.35 + 2 (0.45 - 0.35) = 0.55. x0_hat =
(1.99917538 - 0.4720356 0.55)/0.8815753 =
(1.99917538 - 0.25961958)/0.8815753 = 1.7395558/
0.8815753 = 1.973228.

## E18

eps_tilde = 0.4 - 1.0 0.4720356 0.3 = 0.4 - 0.1416107
= 0.25838859.

## E19

Per forward 156672 FLOPs. DDIM-10: 10 156672 =
1566720. CFG doubles: 3133440 FLOPs = 3.13344
MFLOP.

## E20

Assert 4 (round-trip) fails first: x0_hat built
with ab[t] uses the wrong noise level, so eps_rec
!= eps_hat. Asserts 1-3 still pass (schedule,
bounds, determinism are unaffected by the
indexing).

## L01

Marginals: q(x_t | x_0) fixed, process free.
Reverse: x_{t-1} = sqrt(ab_{t-1}) x0_hat +
sqrt(1 - ab_{t-1} - sigma^2) eps_hat + sigma z.
One jump computed in E05. Marginal identity in
E03. eta = 0 deterministic (E06), eta = 1
stochastic (E07). Negative root: eta > 1 or
wrong ab order, caught by assert 2. Markov
critique: DDPM's Markov chain is sufficient,
not necessary, only marginals train the net.
S-sweep: endpoint error vs S in {5, 10, 20,
50, 100}, predict diminishing returns past
20.

## L02

Blocks: d1 1->8, d2 8->16, bn 16->16, u1
32->8, u2 16->1, all k3. Params: 80, 1168,
2320, 2312, 145 = 6025. Embedding: sinusoid,
emb(50) computed, emb(0) = [0,1,...]. t must
enter because the denoising task changes
with noise level, without it the net is
time-blind. Shape check: skip spatial dims
match (8x8, 4x4), channel sums 32 and 16.
Debug: channel mismatch means the concat
was forgotten in the count. Plain stack:
same params, no fine-detail path. Two-level
experiment: train with/without embedding on
t in {10, 90}, predict the ablated net
cannot beat the task average.

## L03

ELBO weight: beta_t/(2 ab_t (1 - ab_t)),
ratio curve 17.41 at t=1, 1.50 at t=100
(normalized). Tweedie: x0_hat = (x_t + (1
- ab_t) s_hat)/sqrt(ab_t). eps to score:
s_hat = -eps_hat/sqrt(1 - ab_t). CFG:
eps_u + g(eps_c - eps_u), g=2 gives 0.55
and 1.973228. g=10: linear extrapolation,
x0_hat far off-manifold (C10 overshoot).
Classifier guidance: needs a noisy
classifier plus gradients, CFG: one net,
double evals. Weighting experiment: five
schemes, measure sample quality vs
likelihood, predict no scheme wins both.

## L04

Hit-rate: P(|x0_hat - 2.0| < 0.5), g = 0..3
gives 0.000, 1.000, 0.000, 0.000. FLOPs:
156672/eval, DDIM-10 1.56672 MFLOP,
+CFG 3.13344 MFLOP. Battery: six asserts,
all green on the toy. E-018 bug: assert 4
fails first. DDIM-10 vs DDPM-100: 10x
fewer evals, discretization error the
price. Net-vs-steps: 2x net half steps at
fixed FLOPs, predict no difference on the
toy (constant net), real difference on
real data.

## L05

Two jumps fully: 100 -> 90 computed
(x_90 = 1.71491474), 90 -> 80 analogous
with ab_80 = 0.523788. Assumptions: the
net output is fixed at 0.4 (wrong: true
0.5), eta = 0 (no correction noise), tau
uniform. Break the net assumption (make
eps_hat t-dependent): the trajectory
changes at every jump and the endpoint
bias shrinks. Samples: sharper (no noise)
but biased toward the net's systematic
error.

## Implementation and debug task

Reference: ddim_step as in C01 plus the
six asserts. Cause 1: an RNG call inside
the eta = 0 path (e.g. z drawn then
multiplied by sigma = 0 still advances
the RNG and changes later draws, or z
added unconditionally). Caught by assert
3. Cause 2: x_T redrawn between the two
runs (seed reset but draw repeated).
Caught by fixing x_T and comparing, or
by assert 3 with x_T logged.

## Changed-constraint scenarios

S1. Changed numbers: ab_tau (cosine
schedule values), w(t) (different beta
profile), sigma per jump. Unchanged: the
formulas, the battery, the marginal
identity. The ratio w(1)/w(50): the
cosine schedule keeps SNR high longer,
so beta_1 is smaller relative to the
middle, the ratio shrinks. Justify: less
early noise means less ELBO weight
concentration at t = 1.
S2. Candidates: (a) S=10, eta=0, g=1:
10 evals = 1.56672 MFLOP. (b) S=10,
eta=0, g=2: 20 evals = 3.13344 MFLOP.
(c) S=5, eta=0, g=1: 5 evals = 0.78336
MFLOP. Pick (a): CFG (g=1 costs nothing
extra over conditional) at the quality
sweet spot, (c) if the budget is hard
and samples stay acceptable. State the
latency number the budget implies before
choosing.

## Research-critique question

Strong answer: the argument proves the
training objective is unchanged (same
marginals, same (x_0, x_t, t) triples)
and each jump's marginal matches. It
does not prove the 10-step joint equals
the 100-step joint, nor equal perceptual
quality: discretization error differs.
Test: fix net and x_T, generate with
DDIM-10 and DDPM-100, compare with a
two-sample test and a fixed quality
metric over many seeds. Red flags:
citing "same marginals" as "same
samples", no seed control. Remediation:
read the DDIM paper's Section 4
experiments (title-level pointer, not a
quote).
