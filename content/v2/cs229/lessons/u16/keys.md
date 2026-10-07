# Keys: Lesson 16, control and policy optimization

## Breadth recall

E01: The payoff is a finite sum, so no
discount is needed to keep it finite.
Gamma existed to bound the infinite
sum.

E02: s_{t+1} = A_t s_t + B_t a_t +
w_t with w_t ~ N(0, Sigma_t), and
R^(t) = -s_t^T U_t s_t - a_t^T W_t
a_t with U_t, W_t positive definite.

E03: The Riccati recursion computes
Phi_t and Psi_t backward from Phi_T
= -U_T, Psi_T = 0. V*_t(s) = s^T
Phi_t s + Psi_t: quadratic, with Phi_t
negative semidefinite.

E04: Phi_t and L_t never involve
Sigma_t. The noise enters only Psi_t
through -tr(Sigma_t Phi_{t+1}), so it
changes the value, not the policy.

E05: (1) Roll a naive controller for
a nominal trajectory. (2) Linearize
dynamics and quadratize reward around
each point. (3) Solve the LQR
problem for an improved policy.
(4) Roll the new policy on the true
dynamics, repeat from (2).

E06: grad eta = (1/n) sum_i (sum_t
grad log pi_theta(a_t|s_t)) f(tau_i):
average over sampled trajectories of
score times payoff.

## Deep oral ladders

L01: (1) s_{t+1} = A s_t + B a_t +
w_t, R = -s^T U s - a^T W a, T = 2.
(2) Phi_2 = -1. Phi_1 = -1.5. L_1 =
-0.5. (3) -2Wa + 2B^T Phi(As + Ba) =
0 gives a* = -(B^T Phi B - W)^{-1}
B^T Phi A s. The notes omit the
leading minus. (4) Script: Phi stays
[-1, -1.5, -1.6], all negative.
(5) Tabular DP: exact on a grid,
needs discretization, k^d states.
LQR: closed form, continuous, needs
linearity. (6) Unsigned gain formula,
or W not positive definite (a free
direction with no cost). (7) The
linear model is valid near the
linearization point only. Swing-up
lives far from any fixed point,
one linearization cannot cover the
trajectory. (8) Collect n trials
from a coverage policy, least
squares for A, B on (s_t, a_t) ->
s_{t+1}, sample covariance of
residuals for Sigma.

L02: (1) eta(theta) = E_{tau~P_theta}
[f(tau)], f = sum gamma^t R.
(2) pi = [0.5, 0.5], grad log pi =
[0.5, -0.5], payoffs [1, 0],
estimate (0.5 + 0)/2 = 0.25. (3)
grad E[f] = grad int P f = int grad
P f = int P (grad log P) f =
E[(grad log P) f]. (4) Script:
Monte Carlo mean 0.25 within
tolerance over 20000 samples.
(5) Without baseline the two sample
values are 0.5 and 0 (variance
0.0625). With B = 0.5 both are 0.25
(variance 0). (6) Variance: the mean
is right but the noise drowns the
signal. Add a baseline, shorten the
horizon, or move to PPO clipping.
(7) The ratio corrects the action
distribution only. The state
distribution stays d_old, so the
surrogate is local to theta_old.
(8) Sample a batch from theta_old,
run a few epochs (for example 4)
with the clipped surrogate, monitor
clipped fraction and KL from
theta_old, then resample.

## Analytical exercises

E07: From -2 W a + 2 B^T Phi (A s +
B a) = 0: (B^T Phi B - W) a = -B^T
Phi A s, so a* = -(B^T Phi B -
W)^{-1} B^T Phi A s. On the 1d toy:
-( (-1-1)^{-1} )(-1) = -0.5. The
notes print +0.5. Direct check: max_a
-s^2 - a^2 - (s+a)^2 gives a =
-s/2, confirming -0.5. The printed
formula pushes the state away from
zero.

E08: E_{a~pi}[grad log pi(a|s)] =
sum_a pi(a|s) grad pi(a|s)/pi(a|s) =
grad sum_a pi(a|s) = grad 1 = 0.
In (21.10) the weight of step t is
sum_{j>=t} gamma^j R_j. Subtract
gamma^t B(s_t): the extra term is
E[grad log pi(a_t|s_t) B(s_t)] =
E_{history}[B(s_t) E_{a_t}[grad log
pi | history]] = 0 by the identity.
Hence (21.11) is unbiased.

## Failure diagnosis

E09: The notes call J_IS a local
surrogate: it corrects actions but
keeps the old state distribution.
Too many epochs move theta far from
theta_old, the state distribution
shifts, and the surrogate optimizes
a fiction. Diagnose: check KL from
theta_old and the clipped fraction.
Fix: fewer epochs per batch, smaller
eps, resample more often.

## Counterfactual comparison

E10: Team A fails if the nominal
trajectory leaves the valid
linearization region (for example a
bad warm start). Team B fails
wherever the maneuver leaves hover:
one linearization cannot cover a
perching trajectory. The deciding
check: roll each controller on the
true nonlinear dynamics from the
real initial state and compare
closed-loop cost. Simulation on the
linear model proves nothing.

## Research question

E11: Claim: beyond 8 epochs per
batch on the MuJoCo-style walker
task (state the simulator), the
mean KL from theta_old exceeds 0.05
and true returns stop tracking the
clipped surrogate. Metric: per-epoch
KL(theta || theta_old) and the gap
between surrogate gain and realized
return. Baseline: 4 epochs per batch.

## Implementation task

E12: Accept if: Riccati gives Phi =
[-1, -1.5, -1.6] and L = [-0.5,
-0.6] within 1e-9, Kalman gives K =
2/3, s = 1.0, Sigma = 2/3 within
1e-9, bandit Monte Carlo mean is
0.25 within 0.01 and baseline
variance is below no-baseline
variance, the four clip cases give
1.2, 0.5, -1.3, -0.8 within 1e-9.
