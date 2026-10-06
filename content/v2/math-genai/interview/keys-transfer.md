# Transfer set keys, math-genai U01-U10

Date: 2026-10-06. Computed 2026-10-06, numpy
1.26.4, float64. Ground truth:
verify_transfer_keys.py. Interview provenance:
role-derived practice, not employer material.

## T-U01

T1. (a) H(P) = 1.48548 bits. CE(P, Q) = log2 3
= 1.58496 bits. KL(P || Q) = 0.09949 bits.
(b) H(P') = 0.95443 bits. CE(P', Q) is still
1.58496 bits (Q unchanged). KL(P' || Q) =
0.63053 bits. (c) KL(Q || P') is infinite:
q_3 = 1/3 > 0 while p'_3 = 0. Rule: KL(P ||
Q) is finite only where P has support inside
Q's support. the direction matters.
Strong answer: computes all five numbers and
states the support rule with the direction
right. Red flags: symmetric KL, finite answer
for (c). Rubric: 2/2 numbers plus rule. 1/2
numbers only. Remediation: U01-C04, C09.

T2. (a) 0.75. (b) 1.0. The log-likelihood of a
future tail is log 0 = -inf. (c) Laplace
smoothing: (4 + 1)/(4 + 2) = 0.83333. Strong
answer: names smoothing as a prior, not a hack.
Red flags: "MLE is 1.0, so tails are
impossible". Rubric: 2/2 with the -inf stated.
1/2 otherwise. Remediation: U01-C03, C11.

## T-U02

T1. (a) E = 1.0, f(E) = 1.0, E[f] = 2.0, gap =
1.0. (b) E = 2.0, f(E) = 4.0, E[f] = 5.0, gap
= 1.0. (c) The gap stayed fixed. For f(x) =
x^2 the gap is the variance, and variance is
translation-invariant. General fact: the gap
measures spread around the mean, not location.
Strong answer: all eight numbers plus the
variance identification. Red flags: "gap
scales with the mean". Remediation: U02-C01.

T2. (a) D* = [0.6, 0.47368, 0.375]. (b) D*' =
[0.45455, 0.6, 0.5]. (c) Outcome 1 crosses
0.6 -> 0.45455 (above to below), outcome 2
crosses 0.47368 -> 0.6 (below to above),
outcome 3 lands exactly on 0.5. The generator
should decrease q_1, increase q_2, and outcome
3 is at the indifference point. Strong answer:
all six numbers with the directional updates.
Red flags: "D* > 0.5 means the generator is
winning" (it means the discriminator favors
real). Remediation: U02-C04, U03-C03.

## T-U03

T1. (a) V = -1.35179. Check: JS = 0.01725
nats, 2*JS - 2*ln 2 = -1.35179. Match.
(b) Collapsed Q' = [1, 0, 0], stale D: V =
-1.59203. Generator non-saturating: E_Q'[ln
D*] = ln 0.6 = -0.51083, up from -0.74629
before collapse. The stale discriminator
rewards the collapse. (c) Retrained D*' =
[0.33333, 1.0, 1.0]. Generator objective: ln
(1/3) = -1.09861. The retrained
discriminator punishes the collapse. Lesson:
evaluate a generator only against a current
discriminator. a stale D is a broken metric.
Strong answer: all numbers plus the stale-D
lesson in one sentence. Red flags: "collapse
helped, so collapse is good". Remediation:
U03-C06, C11.

T2. (a) Moving mass from outcome 3 to outcome
1 changes the objective by ln 0.6 - ln 0.375
= ln 1.6 = 0.47000 > 0 per unit mass. (b) All
mass flows to outcome 1, the mode of P. Even
with an optimal discriminator, the generator
objective rewards fooling, not covering, so
mode-seeking pressure persists. Strong answer:
the gradient value plus the fooling-vs-cover
distinction. Red flags: "optimal D prevents
mode collapse". Remediation: U03-C06.

## T-U04

T1. (a) eps = 0.01: W1 = 0.1, JS = 0.00348
nats. eps = 0.5: W1 = 5.0, JS = 0.21576 nats.
(b) W1. It is linear in eps with slope 10 at
all scales. JS is near-flat at eps = 0.01
(0.00348 nats of signal) because the
distributions barely overlap, so its gradient
vanishes where the generator needs it most.
Strong answer: all four numbers plus the
gradient argument. Red flags: "JS is fine,
it is nonzero". Remediation: U04-C02, C07.

T2. (a) True W1 = 2. (b) Clipped estimate:
sup over |c| <= 0.5 of c*(0 - 2) = 1.0. Gap =
1.0. (c) Capacity loss from weight clipping:
the box constraint shrinks the function class
below 1-Lipschitz functions. The gradient
penalty constrains ||grad f|| = 1 on
interpolated points instead of boxing the
weights. Strong answer: 1.0, the gap, and the
box-vs-gradient distinction. Red flags:
"clipping enforces 1-Lipschitz exactly".
Remediation: U04-C05, C06.

## T-U05

T1. (a) KL = 0.5*(0.25 + 0.25 - 1 - ln 0.25)
= 0.44315 nats. (b) sigma = 0.1: KL =
0.5*(0.25 + 0.01 - 1 - ln 0.01) = 1.93259
nats. (c) Collapse is mu -> 0, sigma -> 1,
KL -> 0. The optimizer prefers the low-KL
regime because KL is a price, and the cheapest
price is zero. The problem: zero KL means the
posterior ignores x, so the latent carries no
information. Strong answer: both numbers plus
the price logic. Red flags: "collapse means
high KL". Remediation: U05-C07.

T2. (a) 0.05 - 0.04 = 0.01 > 0: keep.
(b) beta = 1.6: 0.05 - 0.064 = -0.014 < 0:
drop. beta = 4: 0.05 - 0.16 = -0.11: drop.
(c) beta* = 1.6007 is the break-even price of
the marginal latent dimension on the lesson
toy: above it, the KL price exceeds the
reconstruction gain. Strong answer: three
decisions plus the break-even reading. Red
flags: "higher beta always helps
disentanglement for free". Remediation:
U05-C08.

## T-U06

T1. (a) Squared distances: [0.40, 0.80, 3.60].
Winner e1. Commitment loss = 0.40, codebook
loss = 0.40. (b) z = (0.1, 0.9): distances
[1.62, 0.02, 2.02]. Winner e2. Both losses =
0.02. (c) The STE passes the decoder gradient
through as if the quantization were the
identity: grad wrt z = grad wrt e_k. e3 gets
no assignments and no gradient, so it drifts
toward dead. Strong answer: all numbers plus
the STE identity statement. Red flags:
"argmin is differentiable". Remediation:
U06-C02, C04.

T2. (a) Code 3 (count 0) is dead: it receives
no STE gradient and only stale EMA updates.
(b) P(unused) = (7/8)^8 = 0.34361. Expected
dead = 8 * 0.34361 = 2.75 codes. (c) One fix:
reset dead codes to random encoder outputs
(or an EMA usage threshold with reset). It
gives the stale code a live starting point
near real data. Strong answer: the mechanism
plus the reset. Red flags: "dead codes fix
themselves". Remediation: U06-C06.

## T-U07

T1. (a) ab_50 = 0.99^50 = 0.60501.
(b) w(50) = 0.01/(2 * 0.60501 * 0.39499) =
0.02092, below the lesson's 0.02873.
(c) The constant schedule destroys signal
faster (0.60501 < 0.77718 at t = 50). The
lower ELBO weight means mid-timesteps get
less training emphasis under the constant
schedule. Strong answer: both numbers plus
the emphasis reading. Red flags: "the
schedule does not affect the loss weights".
Remediation: U07-C02, C10.

T2. (a) 0.03956209 * 2.0 + 0.96013574 * 1.2
= 1.23129. (b) The net output multiplies
0.03956209, the small coefficient: at t = 50
the reverse mean is dominated by x_t, so an
x0-predicting net has its output scaled down
to a whisper. eps prediction keeps the net
output at full strength through the
reweighting. Strong answer: the number plus
the whisper argument. Red flags: "the
parameterization does not matter".
Remediation: U07-C08.

## T-U08

T1. (a) Per forward 156672 FLOPs. (5, 1):
0.78336 MFLOP, feasible. (5, 2): 1.56672
MFLOP, over budget. (10, 1): 1.56672 MFLOP,
over budget. Pick (5, 1). (b) Candidates:
uniform-5 = [20, 40, 60, 80, 100], late-dense-5
= [60, 75, 88, 96, 100]. Predict late-dense-5
degrades less: uniform-5 takes four size-20
jumps, two of them in the high-t regime where
the capstone measured the largest per-jump
errors. Assumption: the learned net's jump
error behaves like the toy's (grows with jump
size and with t). Caveat: on the constant-net
toy the endpoint is schedule-invariant, so
this prediction is about per-jump error, and
a learned net is needed to test endpoints.
Strong answer: the FLOP numbers, the pick,
and the caveated prediction. Red flags:
"more steps always win regardless of
placement". Remediation: U08-C03, C11.

T2. (a) sigma = 0.5 * 0.39249806 =
0.19624903. The jump gains sigma * z =
0.19624903 * z. (b) At eta = 0 the seed did
nothing with x_T fixed (bit-identical
trajectories). At eta = 0.5 the seed drives
the stochastic kick at every jump, so two
seeds give different trajectories from the
same x_T. Strong answer: the number plus the
seed-role change. Red flags: "eta = 0.5 is
half deterministic". Remediation: U08-C02.

## T-U09

T1. (a) sigma^2 = 0.16: at x_tilde = 1.7,
target = (1.5 - 1.7)/0.16 = -1.25. At
x_tilde = 1.9, target = -2.5. (b) Loss =
(-1.0 + 1.25)^2 = 0.0625. (c) Small sigma
dominates: halving sigma quadruples target
magnitudes, so the smallest noise level sets
the loss scale unless the average reweights.
Strong answer: all numbers plus the scaling
rule. Red flags: "the average treats all
sigmas equally". Remediation: U09-C02.

T2. (a) Scores/sqrt(2): [[1.41421, 0.70711,
0.70711], [0.70711, 0.70711, 0], [0.70711, 0,
0.70711]]. Masked rows: [1, 0, 0],
[0.5, 0.5, 0], [0.40111, 0.19778, 0.40111].
(b) Row 0 changes most: it would read
positions 1 and 2, i.e. the future. The model
would train on answers it cannot see at
generation time. Strong answer: the matrix
plus the train/serve mismatch. Red flags:
"the mask only affects efficiency".
Remediation: U09-C07.

## T-U10

T1. (a) 12.00 MiB / 512 = 24 KiB per token.
n = 1024: 24.00 MiB. Batch 8 at n = 1024:
192 MiB. (b) n = 4096: 96 MiB per sequence.
(c) Decode is memory-bandwidth bound: the
cache bytes move per generated token, so
batch times n sets the throughput ceiling
before the weights do. Strong answer: all
numbers plus the bandwidth argument. Red
flags: "KV cache is negligible next to
weights". Remediation: U10-C02, C04.

T2. (a) s = 0.02: q = [16, -37, 56, -9].
Dequantized: [0.32, -0.74, 1.12, -0.18]. Max
|error| = 0.007 <= 0.01. Bound holds.
(b) s = 0.05: q = [6, -15, 23, -4].
Dequantized: [0.30, -0.75, 1.15, -0.20]. Max
|error| = 0.023 <= 0.025. Bound holds.
(c) The scale trades error against metadata:
finer scale, smaller error, more scales to
store (per-tensor vs per-channel). Strong
answer: all values plus the tradeoff. Red
flags: "INT8 is exact". Remediation:
U10-C03.
