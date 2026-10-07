# Keys: Lesson 15, reasoning and reinforcement learning foundations

## Breadth recall

E01: A trace splits one hard prediction
into a sequence of smaller predictions.
It buys more compute (more tokens to
think) and more memory (intermediate
results logged in the context for later
tokens to condition on).

E02: J_R(theta) = E_{x~D,
y~pi_theta(.|x)}[R(x, y)]. J_beta(theta)
= J_R(theta) - beta E_{x~D}[D_KL(
pi_theta(.|x) || pi_ref(.|x))].

E03: A verifier maps a completion to a
scalar reward, for example 1 if the
extracted answer is correct else 0.
Valid: it tests the real property (not
a proxy), it resists gaming (hidden
tests), and format penalties do not
dominate the correctness signal.

E04: (S, A, {P_sa}, gamma, R): states,
actions, transition distributions,
discount factor in [0, 1), reward
function.

E05: V^pi(s) = R(s) + gamma sum_{s'}
P_{s pi(s)}(s') V^pi(s'). V*(s) = R(s)
+ max_a gamma sum_{s'} P_{sa}(s')
V*(s').

E06: Value iteration: V(s) := R(s) +
max_a gamma sum_{s'} P_{sa}(s') V(s'),
repeated to convergence. Policy
iteration: alternate exact evaluation
V := V^pi (linear solve) with greedy
improvement pi(s) := argmax_a
sum_{s'} P_{sa}(s') V(s').

## Deep oral ladders

L01: (1) Tuple as in E04, payoff
R(s_0) + gamma R(s_1) + ... (2) V(B)
= 1/(1-0.9) = 10, V(A) = 0.9 * 10 = 9.
(3) V^pi(s) = E[R(s_0) + gamma *
(rest) | s_0 = s] = R(s) + gamma
E_{s'~P}[V^pi(s')]. (4) Script: errors
1.0, 0.9, 0.81, ... ratio 0.9 per
sweep. (5) Value iteration: O(|S|^2
|A|) per sweep, many sweeps. Policy
iteration: O(|S|^3) linear solve per
iteration, few iterations. (6) gamma
>= 1, or a sign error (maximizing
cost), or rewards unbounded. (7) Real
problems never hand you P_sa. The
estimation error in 19.3 becomes the
dominant term. (8) Collect n trials
from a coverage policy, accumulate
counts per (19.5), warm-start value
iteration on the estimated model,
evaluate the greedy policy on held-out
start states.

L02: (1) State s_t = (x, y_{<t}),
action a_t = y_t, deterministic
transition s_{t+1} = (x, y_{<=t}),
reward 0 until the terminal verifier
reward R(x, y). (2) Rbar = 0.5, s_R
= 0.5, advantages = [1, -1, -1, 1].
(3) r_t = pi_theta / pi_thetaold per
(18.4). C_t = min(r_t Ahat_t,
clip(r_t) Ahat_t), objective E[C_t].
(4) Script checks: Ahat = 1, r = 1.3,
eps = 0.2 gives min(1.3, 1.2) = 1.2.
Ahat = -1, r = 1.3 gives min(-1.3,
-1.2) = -1.3. (5) PPO uses a learned
value baseline V_old(s_t), GRPO uses
the group-relative advantage from G
completions to the same prompt, no
critic. (6) The verifier is invalid:
it rewards format or a proxy, not the
task. (7) The KL penalty limits drift
from pi_ref, which bounds reward
hacking. It cannot fix a verifier
that measures the wrong thing. (8)
Fix the model, vary RL training steps
at fixed sampling budget versus vary
sampling budget (trace length,
samples) at fixed training. Plot
accuracy on each axis separately, as
in Figure 18.1.

## Analytical exercises

E07: V^pi(s) = E[sum_{t>=0} gamma^t
R(s_t) | s_0 = s, pi]. Split t = 0
from t >= 1: = R(s) + E[sum_{t>=1}
gamma^t R(s_t)] = R(s) + gamma
E_{s_1}[E[sum_{t>=0} gamma^t
R(s_{t+1}) | s_1]] = R(s) + gamma
sum_{s'} P_{s pi(s)}(s') V^pi(s').

E08: Rbar = (1+0+0+1)/4 = 0.5. s_R^2
= ((0.5)^2 + (-0.5)^2 + (-0.5)^2 +
(0.5)^2)/4 = 0.25, s_R = 0.5.
Ahat = ([1,0,0,1] - 0.5) / (0.5 +
1e-8) = [1, -1, -1, 1].

## Failure diagnosis

E09: The notes state fitted value
iteration has no convergence
guarantee, and convergence of the
regression loop is not convergence
to V*. The likely cause is feature
misspecification: theta^T phi(s)
cannot represent V*, so the Bellman
targets are fit on sampled states
while the greedy policy fails
elsewhere. Check: enlarge the
feature map and watch the policy
return, not the regression loss.

## Counterfactual comparison

E10: Team A: 50^4 = 6.25e6 states,
about 50 MB at 8 bytes per value.
Exact on the grid, piecewise
constant, chatters at cell
boundaries. Team B: a handful of
parameters, smooth values, but the
quadratic features may miss the true
V*. Memory: B wins by far. Smoothness:
B wins if the features are rich
enough. The deciding check: closed-
loop return of each controller on the
real (nonlinear) pendulum, not the
regression residual.

## Research question

E11: Claim: on a 1d continuous MDP
whose true V* is quadratic, fitted
value iteration with linear features
achieves at most half the return of
the same algorithm with quadratic
features, measured over 20 seeds.
Baseline: tabular value iteration on
a fine grid, which must beat both.

## Implementation task

E12: Accept if: value iteration
reaches V = [9, 10] within 1e-6 of
the closed form after enough sweeps
with error ratio 0.9 per sweep, the
count estimator returns [0.7, 0.3]
then [0.6, 0.4] after accumulation
and [0.5, 0.5] for the unvisited
pair, GRPO advantages equal
[1, -1, -1, 1] within 1e-6.
