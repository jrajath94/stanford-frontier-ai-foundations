# Lesson 02b, First source block: f-divergence to variational minimization

Unit: math-genai-U02 (source block: W1L3, W1L4, W5T10).
Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

This lesson follows the first coherent source block of the
playlist (SRC-04): W1L3 "f-Divergence", W1L4 "Variational
divergence minimization", and W5T10 "Proof of Jensen's
inequality". Mapping is title-level only: no transcript was
inspected (source_gaps.md G2), so every claim about the
instructor's treatment stays at the title boundary. The
examples below are original toys, not lecture reproductions.
Source attribution for the leaf concepts is PENDING. All
numbers computed 2026-10-06, numpy 1.26.4, float64. Log base 2
in bits unless stated.

Playlist-order note: W5T10 sits in Week 5 of the playlist
while W1L3 and W1L4 are Week 1. Title order is a playlist
fact, not a course claim. The proof belongs logically with
this block because Jensen is the tool both lectures use.

## Scope and objectives

Scope: what the three titles name. The f-divergence family
(W1L3). The variational move that turns divergence
minimization into function optimization (W1L4). The proof of
Jensen's inequality with its conditions (W5T10).

Objectives: after this block the learner can write the
f-divergence definition, name five members with their
generators, state the variational dual, explain why the dual
needs samples not densities, and prove Jensen for n points
with the exact conditions.

Dependencies: U02 lesson C01-C06. This block re-covers the
same ground from the source titles' angle. It adds
provenance, not new mechanisms.

---

## SB01, W1L3 f-Divergence

Motivating question: what does the title "f-Divergence" promise
as one idea?

Start from zero. U01 ended with KL as one ruler among several.
The title promises a family, not a single ruler. The family:
D_f(p || q) = E_q[f(p/q)] with f convex and f(1) = 0.

The one idea. A single knob f selects the ruler. The table
from U02-C05, recomputed here on the running Bernoulli toy
(p = 0.7, q = 0.4):

| Ruler | f(t) | Toy value (bits) |
|---|---|---|
| KL | t log2 t | 0.2651 |
| reverse KL | -log2 t | 0.2771 |
| JS | (t/2)log2(2t/(t+1)) + (1/2)log2(2/(t+1)) | 0.0667 |
| TV | \|t-1\|/2 | 0.3000 |
| Hellinger^2 | (sqrt(t)-1)^2 | 0.0932 |

Two facts the title implies and the math confirms. First,
non-negativity: D_f >= f(E_q[r]) = f(1) = 0 by Jensen, so
every member is a valid divergence. Second, identity: p = q
gives r = 1 everywhere, f(1) = 0, D_f = 0.

What the title does not say (title boundary). Which members
the lecture emphasizes, which proofs it shows, and which
examples it works are not inspected. The table above is
authored.

Implementation.

```python
import numpy as np
q = np.array([0.6, 0.4])
r = np.array([0.5, 1.75])
for name, f in [("KL", lambda t: t*np.log2(t)),
                ("JS", lambda t: 0.5*t*np.log2(2*t/(t+1)) + 0.5*np.log2(2/(t+1))),
                ("TV", lambda t: np.abs(t-1)/2)]:
    print(name, np.sum(q * f(r)))
# KL 0.26514844544032273, JS 0.06665370714512762, TV 0.3
```

Assessment: E01-E02. Keys in lessons/u02/keys-source-block.md.

---

## SB02, W1L4 Variational divergence minimization

Motivating question: what do the three words in the title each
contribute?

Start from zero. "Divergence": D_f(p || q), the ruler.
"Minimization": over the model q (or its parameters), we want
q close to p. "Variational": the calculus-of-variations move:
replace the exact divergence, which needs densities, by an
optimization over functions.

The one idea. D_f(p || q) = sup_T E_p[T] - E_q[f*(T)]. The
outer problem is min over q (minimization). The inner problem
is max over T (variational). Together: min_q max_T. On the
toy with KL in nats, T*(x) = ln r(x) + 1 gives exactly
0.1838 nats. Any smaller T class gives less (U02-C10: the
approximation gap).

Why the title matters for the course. This min-max pattern is
the template for GANs (U03): the discriminator is the witness
T, the generator minimizes over q. W1L4 is the mathematical
setup. W2_L6/L7 play the game.

What the title does not say (title boundary). Whether the
lecture derives the conjugate, which f it uses, and whether it
connects forward to GANs are not inspected.

Implementation.

```python
import numpy as np
p = np.array([0.3, 0.7])
q = np.array([0.6, 0.4])
r = p / q
Tstar = np.log(r) + 1.0
inner = np.sum(p * Tstar) - np.sum(q * np.exp(Tstar - 1.0))
print(inner)  # 0.1837868973868122 nats = min-max value at optimal T
```

Assessment: E03-E04. Keys in lessons/u02/keys-source-block.md.

---

## SB03, W5T10 Proof of Jensen's inequality

Motivating question: what does the proof need, exactly?

Start from zero. Claim: for convex f and weights lambda_i >=
0 summing to 1, f(sum_i lambda_i x_i) <= sum_i lambda_i
f(x_i).

The proof. n = 2 is the definition of convex. Assume the
claim for n points. For n + 1 points, write S = sum_{i=1}^{n}
lambda_i and peel off the last point:

    sum_{i=1}^{n+1} lambda_i x_i
      = S * (sum_{i=1}^{n} (lambda_i/S) x_i) + lambda_{n+1} x_{n+1}

The inner sum is a convex combination (weights lambda_i/S sum
to 1). Apply the two-point case with weights (S,
lambda_{n+1}):

    f(...) <= S * f(inner mean) + lambda_{n+1} f(x_{n+1})

Apply the induction hypothesis to the inner mean:

    <= S * sum_{i=1}^{n} (lambda_i/S) f(x_i) + lambda_{n+1} f(x_{n+1})
     = sum_{i=1}^{n+1} lambda_i f(x_i)

Done. The expectation form follows: E[X] is a convex
combination (integral) of values, E[f(X)] the same weights on
f values.

Conditions used, each load-bearing. Convexity of f (the
two-point step). Weights non-negative summing to 1 (the
combination step). Finiteness of both sides (else the
inequality is not a statement about numbers).

Counterexample outside the conditions. f(x) = x^3 - 3x is not
convex on {-2, 1}: f(E) = 1.375, E[f] = -2.0, reversing the
claimed direction. The proof fails at the two-point step:
the chord dips below the graph.

What the title does not say (title boundary). Whether the
lecture proves the finite or the general measure-theoretic
form is not inspected. The proof above is the standard finite
form, authored.

Implementation. The induction step checked numerically.

```python
import numpy as np
xs = np.array([0.4, 0.7])
lam = np.array([0.5, 0.5])
lhs = (lam @ xs) ** 2
rhs = lam @ (xs ** 2)
assert lhs <= rhs + 1e-12
print(lhs, rhs)  # 0.3025 0.325
```

Assessment: E05-E06 and ladder L01. Keys in
lessons/u02/keys-source-block.md.

---

## Source-block exercises (questions. Answers in keys-source-block.md)

E01. Write the f-divergence definition. Verify f(1) = 0 for
the KL and TV generators.
E02. Compute JS on the running toy from the generator formula.
Show the two terms.
E03. Write the min-max form of variational divergence
minimization. Label which max is variational and which min is
the model fit.
E04. On the toy, a constant witness gives dual value 0. Is the
min-max value 0 the true divergence? Explain with C10.
E05. Prove Jensen for n = 3 directly from the n = 2 case,
without citing induction in general.
E06. Give a function and weights where Jensen's conclusion
fails, and name the exact condition violated.

## Deep oral ladder (questions. Answers in keys-source-block.md)

L01. Define f-divergence in one sentence. Toy: the five toy
numbers. Derive the dual from the conjugate definition.
Implement the dual check. Compare title-level source claims
versus inspected evidence for W1L3/W1L4/W5T10. Debug: a
classmate cites the lecture for the JS generator formula.
Critique: is the citation valid? Design: what artifact would
promote the JS generator row from PENDING to source-confirmed?
