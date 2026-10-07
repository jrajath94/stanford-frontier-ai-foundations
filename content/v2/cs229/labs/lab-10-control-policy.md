# Lab 10: LQR, Kalman filter, REINFORCE, PPO clipping

Date: 2026-10-06. Unit: cs229-U16.
Work the tasks, then check keys-lab-10.md. Run all code.

Setup: numpy only. No other installs.

## Task 1: 1d LQR by hand

A = B = U = W = 1, T = 2, Sigma = 0.

(a) Compute Phi_2, Phi_1, Phi_0 with
the Riccati recursion.
(b) Compute the corrected gains L_1
and L_0, and the closed-loop
trajectory from s_0 = 5.
(c) Compute the gain the notes'
unsigned printed formula gives for
L_1, and state in one sentence why
it is wrong.

## Task 2: Kalman predict and update

1d system: s_{t+1} = s_t + w_t,
w_t ~ N(0, 1). y_t = s_t + v_t,
v_t ~ N(0, 1). Prior: s_{0|0} = 0,
Sigma_{0|0} = 1. Observation y_1 =
1.5.

(a) Predict: s_{1|0}, Sigma_{1|0}.
(b) Kalman gain K, updated s_{1|1},
Sigma_{1|1}.
(c) One sentence: what happens to K
if the sensor noise variance grows
to 100?

## Task 3: REINFORCE and PPO numbers

Two-armed bandit, theta = 0,
pi = [0.5, 0.5], rewards [1, 0].

(a) True gradient of eta at theta =
0, and the two-sample estimate from
SL-07.
(b) Per-sample estimates with
baseline B = 0.5. Report the
variance with and without baseline.
(c) PPO clip cases, eps = 0.2:
(Ahat, r) = (1, 1.3), (1, 0.5),
(-1, 1.3), (-1, 0.5). Report C in
each case and mark capped or not.

## Deliverable

A short log: the three result blocks
with numbers. The numbers must match
keys-lab-10.md within the stated
tolerance.
