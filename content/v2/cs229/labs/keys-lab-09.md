# Keys: Lab 09

## Task 1

(a) V(B) = 1 / (1 - 0.9) = 10. V(A) =
0 + 0.9 * 10 = 9.

(b) Sweeps from V = 0:

| sweep | V(A) | V(B) | max error |
|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 10.0000 |
| 1 | 0.0000 | 1.0000 | 9.0000 |
| 2 | 0.9000 | 1.9000 | 8.1000 |
| 3 | 1.7100 | 2.7100 | 7.2900 |
| 4 | 2.4390 | 3.4390 | 6.5610 |
| 5 | 3.0951 | 4.0951 | 5.9049 |

Tolerance: 1e-4 on values.

(c) Error ratio 0.9 between consecutive
sweeps: the Bellman operator is a
gamma-contraction with gamma = 0.9.

## Task 2

(a) After 10 trials: P_hat = [0.7, 0.3].
After 20 trials: numerator [12, 8],
denominator 20, P_hat = [0.6, 0.4].

(b) Fallback: uniform [0.25, 0.25,
0.25, 0.25].

(c) On large |S| the uniform fallback
spreads probability mass over thousands
of unreachable states and dominates the
estimate for every unvisited pair.

## Task 3

(a) Rbar = 0.5. s_R^2 = (0.25 + 0.25 +
0.25 + 0.25) / 4 = 0.25, s_R = 0.5.
Advantages: [1, -1, -1, 1] within 1e-6.

(b) r_t = 1.0 lies inside [0.8, 1.2],
so C_t = r_t * Ahat_t = Ahat_t.
Completion 1: C_t = 1. Completion 2:
C_t = -1.

(c) If all rewards are 1, s_R = 0 and
every advantage is (1 - 1) / epsilon =
0: the group carries no learning
signal.
