# Keys: interview U16

## Breadth

Q01: The time left changes the best
action. With few steps remaining a
near small reward beats a far large
one. The policy must be indexed by
t.

Q02: Linear dynamics s_{t+1} = A_t
s_t + B_t a_t + w_t with Gaussian
noise, quadratic rewards -s^T U s -
a^T W a with U, W positive definite,
finite horizon T.

Q03: Quadratic: V*_t(s) = s^T Phi_t
s + Psi_t. The Bellman backup of a
quadratic under linear dynamics and
quadratic reward is quadratic, so
the shape closes under the
recursion.

Q04: Phi_t and L_t never involve
Sigma_t. Noise enters only Psi_t
via -tr(Sigma_t Phi_{t+1}), which
changes the value, not the policy.

Q05: (1) Nominal trajectory from a
naive controller. (2) Linearize
dynamics, quadratize reward around
each point. (3) LQR solve for an
improved policy. (4) Roll out on the
true dynamics, repeat from (2).

Q06: (1/n) sum_i (sum_t grad log
pi_theta(a_t|s_t)) f(tau_i).

## Deep ladders

L01: F1: As in Q02. F2: -2Wa +
2B^T Phi(As + Ba) = 0, so a* =
-(B^T Phi B - W)^{-1} B^T Phi A s.
F3: Phi_1 = -1.5, L_1 = -0.5. F4:
max_a -s^2 - a^2 - (s+a)^2 gives a
= -s/2. The printed +0.5 maximizes
in the wrong direction. F5: W not
positive definite (free direction),
linearization invalid far from the
nominal point, or no stabilizing
policy exists (closed-loop
eigenvalues outside the unit
circle).

L02: F1: grad int P f = int grad P
f = int P (grad log P) f. F2: log
P_{s_t a_t}(s_{t+1}) does not depend
on theta, so its gradient is zero.
F3: sum_a pi grad log pi = sum_a
grad pi = grad 1 = 0. F4: The extra
term is E[grad log pi(a_t|s_t)
B(s_t)] = E[B(s_t) E[grad log pi |
history]] = 0. F5: The ratio
corrects the action distribution.
The state distribution stays d_old:
uncorrected, bounded by keeping the
excursion small.

## Analytical

A01: Substitute a* = L_t s into the
Bellman backup: Phi_t = A^T(Phi -
Phi B(B^T Phi B - W)^{-1}B^T Phi)A
- U. Toy: (-1 - (-1)(-1/2)(-1)) - 1
= -1.5.

A02: No baseline: values 0.5 (prob
0.5) and 0 (prob 0.5), mean 0.25,
variance 0.0625. B = 0.5: values
0.25 and 0.25, mean 0.25, variance
0. Mean unchanged because E[grad log
pi B] = B E[grad log pi] = 0.

## Implementation and debug

D01: Correct: Phi = [-1, -1.5,
-1.6], trajectory 5 -> 2 -> 1.
Unsigned: L = [+0.5, +0.6],
trajectory 5 -> 7.5 -> 12: the
controller amplifies the state and
diverges. The sign check on Phi
passes in both cases (the bug is in
the gain, not the recursion), so the
test must also check the closed-loop
eigenvalues or a rollout.

## Changed-constraint scenarios

S01: One linearization at the start
fails: the task lives far from any
fixed point. Order: (1) a feasible
warm-start trajectory (even a
hand-designed swing), (2) DDP
iterations around it, (3) LQR only
as the inner solver of DDP step 3.
If DDP diverges, shorten the
horizon or reshape the reward to
keep iterates near the nominal
path.

S02: More epochs per batch (for
example 10), since samples are the
bottleneck. Guards: the clipped
fraction (must stay well above zero
but below saturation) and KL(theta
|| theta_old) (must stay small, for
example below 0.02). If KL grows
while returns fall, cut epochs and
resample.

## Research critique

R01: The claim overreaches. The
notes call the PPO objective a local
surrogate: the ratio corrects only
the action factor while states come
from d_old. Fully off-policy
methods reuse arbitrary old data
with state-distribution corrections.
PPO resamples after a few epochs
precisely because the data must stay
near on-policy. Strong answer:
"PPO is approximately on-policy
with a bounded excursion." Red
flag: treating the clipped
surrogate as a global objective.
