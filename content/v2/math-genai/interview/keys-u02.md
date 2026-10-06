# keys-u02.md, U02 interview answer keys

Date: 2026-10-06. Keep separate from the questions. All numbers
computed 2026-10-06, numpy 1.26.4, float64.

## Breadth

B1. For convex f: f(E[X]) <= E[f(X)]. Conditions: f convex on
the range of X. Both sides finite. The direction flips for
concave f (>=). Strong answer adds: no fixed direction without
convexity or concavity. Red flag: "Jensen always gives <=".
B2. A restricted parameterized set of candidate distributions.
Cost: the approximation gap. The best member may miss the
target. Strong answer: names phi and one concrete miss
(multimodal target vs Bernoulli family). Red flag: confusing
the family with the posterior.
B3. D_f(p || q) = E_q[f(p/q)]. Knob: f. Requirements: convex,
f(1) = 0. Strong answer: derives D_f >= 0 from Jensen in one
line. Red flag: any non-convex f.
B4. sup_T E_p[T] - E_q[f*(T)]. First term needs samples from
p. The second needs samples from q. No densities needed. Strong
answer: states T* = f'(r). Red flag: "the dual needs the
densities".
B5. E_q[p/q] = sum q*(p/q) = sum p = 1. Breaks when supports
mismatch (q = 0 where p > 0): the ratio is undefined. Strong
answer: calls it the first check of any ratio estimate. Red
flag: setting 0/0 = 0.
B6. Approximation (class too small), estimation (finite N),
optimization (maximizer not found). Strong answer: gives the
toy values 0.0006 nats / sigma/sqrt(N) / grid remainder. Red
flag: one undifferentiated "error".

## Deep ladder D1

D1.1. Minimum sufficient: "every chord between two points on
the graph lies above the graph."
D1.2. E[X] = 0.55, f(E[X]) = 0.3025, E[f(X)] = 0.325, gap =
0.0225.
D1.3. log p(x) = log E_q[p(x,z)/q(z)] >= E_q[log p(x,z) - log
q(z)]. Three lines: introduce q, Jensen (log concave), done.
D1.4. Causes: (a) support violation: q = 0 where p(x,z) > 0,
making a term +inf or NaN. (b) sign flip: used <= or coded
E_q[log q - log p(x,z)]. Tell apart: check min q on the
support. If positive, it is the sign.
D1.5. Survives: introduce q, Jensen, the bound form, the gap
= KL identity. The finite sum becomes an integral. The
expectation E_q is now over a continuous q. The support
condition becomes absolute continuity.

## Deep ladder D2

D2.1. f*(s) = sup_t (s t - f(t)). Minimum sufficient, plus:
it is always convex.
D2.2. T* = ln r + 1 = [0.3069, 1.5596]. Dual = E_p[T*] -
E_q[r] = 1.1838 - 1.0 = 0.1838 nats = 0.2651 bits.
D2.3. Pointwise: sup_T(x) of T(x) r(x) - f*(T(x))... in the
E_p/E_q form, sup over T of E_p[T] - E_q[f*(T)] attains
f(r(x)) at each x when T*(x) = f'(r(x)). Taking expectations
gives E_q[f(r)] = D_f.
D2.4. Diagnosis: approximation gap. Constants cannot approach
T*. Best dual 0 < truth 0.2651 bits. The "models match"
claim confuses witness weakness with distribution closeness.
Fix: enrich the class, then re-measure.
D2.5. Attack: samples must grow too. A bigger class with
fixed N inflates the estimation gap (overfit witness:
memorizes the sample piles, dual overshoots). What breaks:
the empirical dual stops tracking the population dual.
Remediation: grow class and N together. Report SE.

## Analytical/quantitative

Q1. f(t) = -log2 t, f(1) = 0. Terms: q(0)*f(r(0)) =
0.6*(-log2 0.5) = 0.6*1 = 0.6. q(1)*f(r(1)) =
0.4*(-log2 1.75) = 0.4*(-0.8074) = -0.3229. Sum = 0.2771
bits. Cross-check: equals KL(q||p) = 0.2771 from the lesson.
Q2. sigma = 0.6124, SE = 0.6124/40 = 0.0153. Measured: 0.0200
at N = 1000 (predicted 0.0194), 0.0011 at N = 10000
(predicted 0.0061). The law holds: errors track sigma/sqrt(N)
within a factor of about 2.

## Implementation/debug

T1. The bug: np.sum(np.log2(p/q)) sums the log-ratios without
weighting by p. Fix: np.sum(p * np.log2(p/q)) = 0.2651. The
general rule: a divergence is an expectation (E_p or E_q of a
function of r), never a bare sum over outcomes. Ship the
fixed version with the support assert from the lesson.

## Changed-constraint scenarios

S1. Survive: Jensen, the dual form, the 1/sqrt(N) law, the
three gaps, the support message. First to change: exact
tables become density ratios. Estimate r(x) with the
classifier route (C04). The dual objective itself is
unchanged.
S2. The number (C05/C06), its standard error (C08), the
support verdict: min q mass on p samples (C07), and the
witness class description (C10). Each maps to its section.

## Research-critique

R1. sqrt is concave, so it is not a valid generator: the
"divergence" can go negative, and -0.045 proves nothing about
the model. The ranking is meaningless. The paper needs a
convex f with f(1) = 0 before any comparison.
