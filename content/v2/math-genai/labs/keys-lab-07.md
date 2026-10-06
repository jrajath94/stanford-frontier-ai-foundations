# Lab keys 07, diffusion-variant mechanics

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4,
float64, seed 0 where RNG is used. Ground truth:
compute_run5b.py.

## Task 1

(a) Predict: eta = 0 -> sigma = 0 exactly. eta = 1
-> sigma = sqrt(jump posterior var), about 0.39.
(b) Measured: x_100 = 1.60480905, x0_hat =
2.13230847, x_90 = 1.71491474. sigma(eta=1) =
0.39249806.
(c) 0.15405473 + 0.40507252 = 0.55912725 = 1 -
ab_90. Identity holds to 1e-12.
(d) eta = 1.5 at the 100 -> 90 jump: sigma =
0.58874709, 1 - ab_90 - sigma^2 = 0.212504 >
0, no break yet. At the 20 -> 10 jump: sigma =
0.130261, 1 - ab_10 - sigma^2 = -0.006966 <
0: the square root fails. Assert 2 (sigma <=
sqrt(1 - ab_{t-1})) catches it. eta > 1 is
outside the contract, early jumps break first
because 1 - ab_{t-1} is smallest there.

## Task 2

(a) max |diff| < 1e-12, bit-identical.
(b) max |diff| > 1e-6, genuinely different.
(c) Deterministic means the trajectory is a pure
function of (x_T, net): reproducible. It does not
mean constant: x_T is still drawn, so samples
vary.
(d) Yes, the sample changes: x_T is the
randomness source. Determinism removes per-step
noise, not the initial draw.

## Task 3

(a) ang = 50/1 = 50: sin = -0.262375, cos =
0.964966.
(b) Verified: [0, 1, 0, 1, 0, 1, 0, 1].
(c) cos(49, 50) = 0.8838, cos(10, 90) = 0.3594.
(d) The net loses time information: its output
no longer depends on t, so it cannot do
different denoising at different noise levels.

## Task 4

(a) Predict: linear in g, x0_hat falls as g
rises.
(b) Measured: g=0: 2.080317. g=1: 2.026772. g=2:
1.973228. g=3: 1.919683.
(c) Hit-rates: 0.000, 1.000, 0.000, 0.000.
(d) The blend x0_hat(g) = 0.3 + g 1.7 is linear,
so the mean walks past the target 2.0 for g >
1. The peak is at g = 1 because that is where
the mean equals the conditional target, the
noise (0.15) is small, so the peak is sharp.

## Task 5

(a) All six green on the correct implementation.
(b) Assert 4 (round-trip) fails first:
eps_rec != 0.4.
(c) DDIM-10 g=1: 1566720 FLOPs = 1.56672 MFLOP.
DDIM-10 + CFG: 3133440 = 3.13344 MFLOP.
DDPM-100: 15667200 = 15.6672 MFLOP.
(d) Ship DDIM-10 g=1 if 1.56672 MFLOP fits the
budget with headroom, evidence still needed: a
fixed quality metric (FID or human eval) on
held-out prompts comparing the three configs,
because FLOPs do not measure quality.
