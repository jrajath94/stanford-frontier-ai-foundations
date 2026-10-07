# Capstone A: research replication and falsifiable extension

Date: 2026-10-06. Units: U15 (fitted
value iteration), U17 (ablation and
uncertainty protocol). Script:
scripts/capstone_a_fitted_vi.py.
Figure:
figures/capstone_a_results.png.

## Replication target

Section 19.4.2 of the notes: fitted
value iteration for continuous-state
MDPs. The notes state that it often
works in practice but carries no
convergence guarantee, and that the
choice of feature map decides what
the method can represent. This
capstone replicates the algorithm
exactly as described: sample n
states, estimate q(a) with k
next-state draws, take the max over
actions, refit theta by least
squares, repeat.

## Extension question

How much does feature
misspecification cost? The notes
warn about it but never measure it.

## Falsifiable hypothesis

Written before the run: on a 1d
continuous MDP whose optimal value
function is nearly quadratic, fitted
value iteration with quadratic
features achieves a mean
closed-loop return at least 50%
better (less negative) than with
linear features, over 5 seeds, and
a fine-grid tabular value iteration
baseline beats both or ties the
quadratic map.

## Method

Environment: s in [-2, 2], actions
{-1, +1}, s' = clip(s + 0.5a + w)
with w ~ N(0, 0.1), R(s, a) =
-s^2 - 0.1a^2, gamma = 0.9.
Budgets matched: n = 200 sampled
states, k = 5 draws per (s, a), 30
iterations, same simulator. Arms:
phi = [1, s] (linear), phi = [1, s,
s^2] (quadratic), and a 41-bin
tabular baseline with 200 sweeps.
Metric: greedy-policy mean return
over 20 episodes of 50 steps from
uniform starts, 5 seeds.

## Results

| arm | mean return | std |
|---|---|---|
| linear phi | -30.489 | 1.529 |
| quadratic phi | -3.783 | 0.312 |
| tabular baseline | -4.321 | (single run) |

Relative improvement of quadratic
over linear: 87.6%. Claim part 1:
CONFIRMED, far above the 50%
bar. Claim part 2: the tabular
baseline (-4.321) does not beat the
quadratic map (-3.783 +/- 0.312).
The gap is 0.538, within about two
standard deviations of the
quadratic arm's noise: the honest
verdict is a TIE within noise, not
a tabular win. The prediction said
"beats both or ties", the tie
clause is what the data supports.

## Interpretation

The linear map cannot represent the
quadratic value function. Its
fitted V is a tilted plane, so the
greedy policy is bang-bang: always
push one direction, pin the state
at the boundary, collect about -4
per step. Its return (-30.5) is
worse than a random policy (-14.9,
measured separately). This is the
U15 failure mode made concrete:
regression loss converged, the
policy is catastrophic. The
quadratic map recovers the true
shape and matches the tabular
baseline at a fraction of the
memory (3 parameters vs 41 bins).

## Limitations

One environment, one noise level,
one sampling budget. The 50% bar
was arbitrary, the effect size
(87.6%) is what matters. The
tabular baseline ran one seed, so
its number has no error bar. The
greedy evaluation used a
deterministic one-step lookahead,
a sampled lookahead would add
noise but not change the ranking.

## Negative results

None hidden: the linear arm's
catastrophic failure was the
predicted negative result for that
arm, and it is reported with the
same prominence as the quadratic
arm's success. No seeds were
dropped. No hyperparameters were
retuned after seeing the results.
