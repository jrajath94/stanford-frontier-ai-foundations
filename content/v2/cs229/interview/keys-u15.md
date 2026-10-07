# Keys: interview U15

## Breadth

Q01: More compute (more tokens to
think) and more memory (intermediate
results in context). One hard
prediction becomes a sequence of
smaller predictions that condition on
each other.

Q02: Few-shot includes hand-written
(x, z, a) demonstrations. Zero-shot
replaces them with an instruction
such as "Let us think step by step".

Q03: J_R(theta) = E_{x~D,
y~pi_theta(.|x)}[R(x, y)]. x: prompt.
y: completion with trace and answer.
R: verifier reward in [0, 1]. pi_theta:
the language model as policy.

Q04: It keeps pi_theta close to the
reference model pi_ref, limiting drift
and reward hacking. It cannot fix a
verifier that measures the wrong
thing.

Q05: (S, A, {P_sa}, gamma, R): states,
actions, transition distributions,
discount factor in [0, 1), reward
function.

Q06: V^pi(s) = R(s) + gamma sum_{s'}
P_{s pi(s)}(s') V^pi(s'). V*(s) =
R(s) + max_a gamma sum_{s'} P_{sa}(s')
V*(s').

## Deep ladders

L01: F1: V^pi(s) = E[sum gamma^t
R(s_t) | s_0 = s, pi]. F2: Split off
t = 0: R(s) + gamma E_{s'}[V^pi(s')].
F3: V(s) := R(s) + max_a gamma
sum_{s'} P_{sa}(s') V(s'). Styles:
synchronous (all states, then
overwrite) and asynchronous (one at a
time). F4: Sweep 1: V = [0, 1].
Sweep 2: V = [0.9, 1.9]. F5: Value
iteration: cheap sweep O(|S|^2|A|),
many sweeps, asymptotic convergence.
Policy iteration: expensive linear
solve O(|S|^3) per iteration, few
iterations, exact in finite time with
an exact solver.

L02: F1: s_t = (x, y_{<t}), a_t =
y_t, s_{t+1} = (x, y_{<=t})
deterministic, reward 0 until terminal
R(x, y). F2: The verifier only scores
the finished completion. There is no
per-token ground truth. F3: Ahat_i =
(R_i - Rbar) / (s_R + epsilon).
Rbar = 0.5, s_R = 0.5, advantages
[1, -1, -1, 1]. F4: PPO uses a learned
value baseline V_old(s_t). GRPO uses
the group-relative advantage from G
completions to the same prompt, no
critic. F5: Verifier invalid: it
rewards a proxy (format, length) not
the task. Check task accuracy on a
stricter verifier.

## Analytical

A01: V*(s) = max_pi E[R(s_0) + gamma
sum_{t>=1} ... | s_0 = s] = R(s) +
max_pi gamma E_{s_1}[V^pi(s_1)].
The max over policies of the future
term equals max_a gamma sum_{s'}
P_{sa}(s') max_{pi'} V^{pi'}(s') =
max_a gamma sum_{s'} P_{sa}(s')
V*(s'), because after the first action
the remaining policy choice is
independent. This gives (19.2).

A02: E[grad log pi] = sum_a pi(a|s)
grad pi(a|s) / pi(a|s) = grad sum_a
pi(a|s) = grad 1 = 0. A baseline
B(s) contributes E[grad log pi(a|s)
B(s)] = B(s) E[grad log pi] = 0, so
the mean gradient is unchanged while
variance can drop.

## Implementation and debug

D01: Correct run: max errors 10, 9,
8.1, ..., ratio 0.9. With gamma =
1.0 the errors stop shrinking: V(B)
grows by 1 per sweep without bound
(V(B) = t after t sweeps) and the
"error" against any finite target
grows linearly. The contraction needs
gamma < 1.

## Changed-constraint scenarios

S01: RLVR on format-only reward trains
format, not correctness. Change: add
a correctness check to the verifier
(extracted answer match, hidden unit
tests) or drop RLVR for SFT on
verified traces. Monitor: task
accuracy on a held-out strict
verifier, KL from pi_ref, and trace
length (a hacking model writes long
empty traces).

S02: Without a simulator, fitted
value iteration loses its sampling
oracle. Discretization survives: it
needs only the ability to estimate
P_sbar a from real experience (19.3
counts), no simulator. What breaks:
the k^d curse at d = 12, and the
piecewise-constant policy. Honest
answer: neither classical method is
good here, this is where function
approximation with learned models
(U16 policy methods) takes over.

## Research critique

R01: The claim is false. The notes
state explicitly that fitted value
iteration cannot be proved to always
converge. The regression step
projects the Bellman target onto the
feature span, and the projection of a
contraction need not be a
contraction. Tabular value iteration
converges because the max-norm
contraction acts on the full vector.
Strong answer names the projection
as the broken link. Red flag: citing
"it works in practice" as a
guarantee.
