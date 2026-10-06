# Interview bank, U08 diffusion variants and implementation

Date: 2026-10-06. Questions only. Keys in
interview/keys-u08.md. Closed-book. Do not read the keys
first. The running toy: T = 100, linear betas 1e-4 to
0.02, x_0 = 2.0, eps = 0.5, eps_hat = 0.4, tau = [10,
..., 100].

## Breadth (6)

B1. Write the DDIM reverse step for a jump t ->
t-1. State the marginal contract in one sentence.
B2. Give sigma for the 100 -> 90 jump at eta = 0
and eta = 1. Explain the eta = 1 value.
B3. Count the params and FLOPs of the toy U-Net
u1 block (32 -> 8, k3). State the totals.
B4. Compute emb(50) entries 0-1 for d = 8. What
is emb(0)?
B5. Write the ELBO loss weight w(t). Give w(1),
w(50), w(100).
B6. Write the CFG formula. What do g = 0 and g =
1 give?

## Deep ladder D1, the DDIM construction (5 follow-ups)

D1.1. Define the DDIM marginal contract in one
sentence.
D1.2. Toy: compute the 100 -> 90 jump at eta = 0
(x_100, x0_hat, x_90).
D1.3. Derive the marginal variance identity and
show eta = 1 gives the jump posterior variance.
D1.4. Implement/debug: eta = 0 but trajectories
differ across seeds with fixed x_T. Name the
cause.
D1.5. Changed constraint: tau must have S = 5.
Propose the schedule and name what degrades.

## Deep ladder D2, steering (5 follow-ups)

D2.1. Define the ELBO weight and the simplified
weight in one sentence each.
D2.2. Toy: compute 1/w(t) at t = 1, 50, 100.
Where does the uniform weight distort most?
D2.3. Derive the Tweedie x0_hat from s_hat.
D2.4. Implement/debug: g = 10 gives garbage
samples but the loss is fine. Explain.
D2.5. Research critique: "CFG with g > 1 is
just a stronger conditional model." Attack it.

## Analytical/quantitative (2)

A1. The toy U-Net forward costs 156672 FLOPs.
Compute per-sample FLOPs for (a) DDPM-100, (b)
DDIM-10, (c) DDIM-10 with CFG. Give the three
ratios to (a).
A2. From the hit-rate table (g = 0..3: 0.000,
1.000, 0.000, 0.000), estimate the largest g
with hit-rate above 0.5 under linear
interpolation between g = 1 and g = 2. State
the assumption.

## Implementation/debug (1)

I1. A colleague's ddim_step passes asserts
1-3 but fails assert 4 (round-trip). They
indexed ab[t] instead of ab[t-1] in x0_hat.
Explain why asserts 1-3 pass anyway, and
write the one-line fix.

## Changed-constraint scenarios (2)

S1. The net now outputs v instead of eps.
Rewrite x0_hat and the Tweedie conversion.
Which battery asserts change?
S2. Serving budget: 2 MFLOP per sample, toy
U-Net fixed. List feasible (S, g) configs
and pick one. Justify.

## Research-critique (1)

R1. "The DDIM marginal-matching argument
proves DDIM-10 samples equal DDPM-100
samples." Attack the claim: what is proved,
what is not, and what experiment tests the
strong version?
