# keys-lab-01.md, Lab 01 answer keys

Date: 2026-10-06. Keep separate from the lab file. All numbers
computed 2026-10-06, numpy 1.26.4, float64, seed 0.

## Task 1

(a) r(0) = 0.3/0.6 = 0.5. r(1) = 0.7/0.4 = 1.75.
(b) KL = 0.7*log2(1.75) + 0.3*log2(0.5) = 0.7*0.8074 +
0.3*(-1) = 0.5651 - 0.3 = 0.2651. Reverse KL = 0.4*log2(0.4/0.7)
+ 0.6*log2(2) = -0.3229 + 0.6 = 0.2771. JS = 0.0667. TV =
0.5*(0.3 + 0.3) = 0.3. H^2 = 0.0932.
(c) The C05 snippet reproduces: 0.26514844544032273,
0.2770580311769584, 0.06665370714512762, 0.3,
0.09317133815030676.
(d) f(t) = sqrt(t) gives a negative number (about -0.045).
The non-negativity assert fires. It must: sqrt is concave,
not a valid generator, so the "divergence" is meaningless.

## Task 2

(a) E[X] = 0.55, f(E[X]) = 0.3025. E[f(X)] = 0.5*(0.16 +
0.49) = 0.325. Gap = 0.0225.
(b) Code confirms 0.0225. sqrt: sqrt(0.55) = 0.7416,
0.5*(sqrt(0.4) + sqrt(0.7)) = 0.7346. Reversed, as concave
demands.
(c) f(E): E = -0.5, f(-0.5) = -0.125 + 1.5 = 1.375. E[f]:
0.5*f(-2) + 0.5*f(1) = 0.5*(-2) + 0.5*(-2) = -2.0. The Jensen
direction f(E) <= E[f] reverses: 1.375 > -2.0. Violated
condition: convexity of f.

## Task 3

(a) sigma^2 = 0.6*(0.5-1)^2 + 0.4*(1.75-1)^2 = 0.15 + 0.225 =
0.375, sigma = 0.6124. SE at N = 400: 0.6124/20 = 0.0306.
Predicted before running.
(b) Seed 0, N = 400: measured estimate and error depend on the
draw. Expected |error| around 0.03. (Reference run: the
lesson reports 0.0750 at N = 100, 0.0200 at N = 1000,
0.0011 at N = 10000.)
(c) |error| should sit within about 2 SE (0.061) of zero at
N = 400, and within 0.012 at N = 10000. The lesson's measured
errors do.
(d) "Your number has a standard error around sigma/sqrt(50).
without it the digits are decoration."

## Task 4

(a) Best 0.18320129756895054 nats at (a, b) = (1.3, 0.3).
(b) 0.1832 <= 0.1838 holds. Approximation gap = 0.1837869 -
0.1832013 = 0.0005856 nats.
(c) Analytic: (-0.3873127313836182, -0.687312731383618).
Finite-diff: (-0.38731273122039056, -0.6873127311735061). Max
discrepancy under 1e-9.
(d) Constants give best dual 0 (attained at T = 1). The
shortfall 0.1838 nats is the approximation gap: the class
cannot approach T*.

## Task 5

(a) KL = 0.7*log2(0.7/0.01) + 0.3*log2(0.3/0.99) =
0.7*6.1293 + 0.3*(-1.7225) = 4.2905 - 0.5167 = 3.774 bits
(computed 2026-10-06. Accept 3.77).
(b) numpy issues "RuntimeWarning: divide by zero encountered
in log2" and returns inf for the term, so KL = inf.
(c) assert np.all((p > 0) <= (q > 0)), "support violation".
It blocks the task-1(d)-style silent wrong number: a zero-mass
q can no longer pass, and it guards every ratio before the
division.
