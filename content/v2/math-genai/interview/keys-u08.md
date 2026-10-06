# Interview keys, U08 diffusion variants and implementation

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4,
float64. Ground truth: compute_run5b.py. Interview
provenance: role-derived practice, not employer material.

## Breadth

B1. x_{t-1} = sqrt(ab_{t-1}) x0_hat + sqrt(1 -
ab_{t-1} - sigma^2) eps_hat + sigma z. Contract:
q(x_t | x_0) matches the DDPM marginals for every t,
so the trained epsilon net transfers unchanged.
B2. eta = 0: 0. eta = 1: 0.39249806, squared
0.15405473 = the 100 -> 90 jump posterior
variance (not the single-step beta_tilde_100 =
0.01976684).
B3. u1: 32 8 9 + 8 = 2312 params. Totals: 6025
params, 156672 FLOPs per forward.
B4. sin(50) = -0.262375, cos(50) = 0.964966.
emb(0) = [0, 1, 0, 1, 0, 1, 0, 1].
B5. w(t) = beta_t/(2 ab_t (1 - ab_t)). 0.50005,
0.02873, 0.04322.
B6. eps_tilde = eps_u + g (eps_c - eps_u). g = 0:
eps_u. g = 1: eps_c.

## Deep ladder D1

D1.1. Keep the DDPM marginals q(x_t | x_0), drop
the Markov requirement, any consistent reverse
reuses the trained net.
D1.2. x_100 = 1.60480905, x0_hat = 2.13230847,
x_90 = 1.71491474 (see keys.md E05).
D1.3. Var = (1 - ab_{t-1} - sigma^2) + sigma^2
= 1 - ab_{t-1}. At eta = 1, sigma^2 = ((1 -
ab_{t-1})/(1 - ab_t))(1 - ab_t/ab_{t-1}), the
jump posterior variance.
D1.4. An RNG call inside the eta = 0 path (z
drawn even though multiplied by sigma = 0)
advances the RNG and changes later draws, or
x_T redrawn. Caught by assert 3.
D1.5. tau = [20, 40, 60, 80, 100] (uniform) or
late-dense. Discretization error grows, the
endpoint bias from the constant net grows.

## Deep ladder D2

D2.1. ELBO: w(t) = beta_t/(2 ab_t (1 - ab_t)).
Simplified: 1 everywhere.
D2.2. 1/w: 2.00, 34.81, 23.14. Most distortion
at t = 50.
D2.3. s_hat = -eps_hat/sqrt(1 - ab_t),
substitute into x0_hat = (x_t - sqrt(1 - ab_t)
eps_hat)/sqrt(ab_t) to get (x_t + (1 - ab_t)
s_hat)/sqrt(ab_t).
D2.4. The loss never sees g, g acts only at
sampling. g = 10 linearly extrapolates x0_hat
off-manifold (C10 overshoot). Strong answer
names the train/serve split.
D2.5. Attack: g > 1 extrapolates past the
conditional prediction, it is not a model but
a sampling-time exaggeration. Evidence: x0_hat
moves linearly past the conditional value
(1.973228 at g = 2 vs 2.026772 at g = 1).
Red flag: calling it "stronger conditioning"
without noting extrapolation.

## Analytical/quantitative

A1. (a) 15.6672 MFLOP. (b) 1.56672 MFLOP. (c)
3.13344 MFLOP. Ratios to (a): 1, 0.1, 0.2.
A2. Linear interpolation between (1, 1.000)
and (2, 0.000): hit-rate 0.5 at g = 1.5.
Assumption: the hit-rate falls linearly in g,
which the mechanism does not guarantee (the
true curve needs measurement).

## Implementation/debug

I1. Asserts 1-3 check the schedule, the sigma
bound, and determinism: none touches which ab
index x0_hat uses. Assert 4 (round-trip)
compares eps_rec to eps_hat and fails. Fix:
x0_hat = (x_t - sqrt(1 - ab[t-1]) eps_hat) /
sqrt(ab[t-1]) for 1-indexed t.

## Changed-constraint scenarios

S1. v = sqrt(ab_t) eps - sqrt(1 - ab_t) x_0,
so x_0 = (sqrt(ab_t) eps_hat... solve: x0_hat
= (x_t - sqrt(1 - ab_t) v_hat... from v_hat
= sqrt(ab_t) eps - sqrt(1 - ab_t) x_0 and x_t
= sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps:
x0_hat = sqrt(ab_t) x_t - sqrt(1 - ab_t)
v_hat. Tweedie unchanged (it needs s_hat,
convert v_hat to eps_hat first). Battery:
assert 5 (Tweedie = eps) becomes a v-round-
trip, the rest unchanged.
S2. Per eval 156672 FLOPs, budget 2e6.
Feasible: (S=10, g=1): 1.56672 MFLOP.
(S=5, g=2): 1.56672 MFLOP. (S=12, g=1):
1.880064 MFLOP. Pick (S=10, g=1): the
guidance sweet spot with 10 jumps, more
steps beat CFG-doubled evals at fixed
budget absent quality data. Justify with
the hit-rate peak at g = 1.

## Research-critique

R1. Proved: same training objective (same
marginals, same triples), per-jump
marginals match. Not proved: equal joint
samples or equal quality at 10 vs 100
steps. Test: fix net and x_T, sample both,
two-sample test plus a fixed quality
metric over many seeds. Red flags: "same
marginals" quoted as "same samples", no
seed control. Remediation: title-level
pointer to the DDIM paper experiments,
not a quote.
