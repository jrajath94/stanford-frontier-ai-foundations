# keys-source-block.md, U02 first source block answer keys

Date: 2026-10-06. Closed-book answers for lesson 02b. Keep
separate from the lesson file. All numbers computed 2026-10-06,
numpy 1.26.4, float64.

## E01

D_f(p || q) = E_q[f(p/q)], f convex, f(1) = 0. KL: f(1) =
1*log2(1) = 0. TV: f(1) = |1-1|/2 = 0.

## E02

r = [0.5, 1.75], q = [0.6, 0.4]. Term at t = 0.5:
0.5*0.5*log2(1.0/1.5) + 0.5*log2(2/1.5) = 0.25*(-0.5850) +
0.5*0.4150 = -0.1462 + 0.2075 = 0.0613. Term at t = 1.75:
0.5*1.75*log2(3.5/2.75) + 0.5*log2(2/2.75) = 0.875*0.3479 +
0.5*(-0.4594) = 0.3044 - 0.2297 = 0.0747. JS = 0.6*0.0613 +
0.4*0.0747 = 0.0368 + 0.0299 = 0.0667 bits.

## E03

min_q max_T E_p[T] - E_q[f*(T)]. The max over T is the
variational step (function optimization replacing the exact
divergence). The min over q is the model fit (move the model
toward the data).

## E04

No. The constant witness class gives inner max 0, but the true
divergence is 0.2651 bits. The shortfall is the approximation
gap (C10): the class cannot approach T*. The min-max value
under a restricted class is a lower estimate, not the truth.

## E05

n = 3: weights a, b, c summing to 1. f(a x + b y + c z) =
f((a+b)*(a x + b y)/(a+b) + c z) <= (a+b) f((a x + b
y)/(a+b)) + c f(z) <= (a+b)((a/(a+b)) f(x) + (b/(a+b)) f(y))
+ c f(z) = a f(x) + b f(y) + c f(z). Two uses of the two-point
case.

## E06

f(x) = x^3 - 3x on {-2, 1} with weights (0.5, 0.5): f(E) =
1.375, E[f] = -2.0, so the Jensen direction f(E) <= E[f] fails.
Violated condition: convexity of f (the chord dips below the
graph).

## L01

f-divergence: one sentence as in E01. Toy: five numbers from
SB01. Derivation: D_f = E_q[f(r)]. Conjugate f*(s) = sup_t (s
t - f(t)). Pointwise sup over T(x) recovers f(r(x)). Hence
sup_T E_p[T] - E_q[f*(T)] = E_q[f(r)] = D_f. Code: the SB02
snippet. Assert |inner - KL_nats| < 1e-12. Source claims:
W1L3/W1L4/W5T10 are title-level only (G2). No transcript
inspected, so any claim about lecture content (which
generators, which proof form) is PENDING. Debug: the citation
is invalid. The classmate cites a title, not inspected
content. Critique: titles do not establish taught content.
Design: an inspected transcript or slide deck showing the JS
generator formula with a timestamp or page anchor would
promote the row. Until then it stays authored with computed
numbers.
