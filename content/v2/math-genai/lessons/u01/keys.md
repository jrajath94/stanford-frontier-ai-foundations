# keys.md, U01 lesson answer keys

Date: 2026-10-06. Closed-book answers for lesson 01. Keep separate
from the lesson file.

## E01

p_data: the unknown law that made the files. p_hat: counts/8, the
dataset as a table. q_uni: the model, 1/4 each. Training may change
only q.

## E02

The memorizer claims P(0000) = 1 and everything else 0. One file
cannot support certainty about the other 15 possible 4-bit
patterns.

## E03

Map die faces 1-6 to patterns (reroll 5, 6, or map them to two
patterns twice). The drawn face is one sample. The distribution is
the full odds table. One draw is not the table.

## E04

l = 8 * log2(1/8) = 8 * (-3) = -24 bits.

## E05

Raw likelihoods are products of n numbers below 1. They underflow
to 0.0 in float64 for large n, so every model scores zero and no
comparison is possible. Logs turn the product into a sum.

## E06

The log-likelihood is -infinity (log(0) = -infinity). The support
check (C04 code) would have caught it: 1010 has model mass 0.

## E07

z = A means "coin A was used for this flip". z is not observed. We
see only heads or tails.

## E08

p(heads) = 0.9*0.9 + 0.1*0.2 = 0.81 + 0.02 = 0.83.

## E09

p(tails) = p(A, tails) + p(B, tails) = 0.05 + 0.40 = 0.45.

## E10

p(B | tails) = p(B, tails)/p(tails) = 0.40/0.45 = 0.8889.

## E11

Fair coin: H = -(0.5*(-1) + 0.5*(-1)) = 1 bit. Certain coin:
H = -(1*0 + 0) = 0 bits. The fair coin is larger: maximum
uncertainty.

## E12

Unnormalized counts give "probabilities" summing to 8. Then
-log(3) is negative, terms flip sign, and the sum is not an
entropy. Entropy needs a real distribution.

## E13

H(p_hat, q_emp) = -sum p_hat log q_emp = 1.9056 bits = H(p_hat).
The gap is 0: the model matches the data odds exactly.

## E14

Cross-entropy prices the model's code on data drawn from p. The
weight must be the data odds p, because those are the frequencies
at which each surprisal is actually paid.

## E15

D_KL = 0.375*log2(0.375/0.25) + 0.125*log2(0.125/0.25) + 0 + 0
= 0.375*0.5850 + 0.125*(-1) = 0.2194 - 0.125 = 0.0944 bits.

## E16

No. D_KL(p_hat || q_uni) = 0.0944, D_KL(q_uni || q_emp) = 0.1038.
Different numbers. KL is not symmetric.

## E17

TV still ranks: TV(p_hat, q_bad) = 0.375. KL gives infinity and
refuses to rank.

## E18

l(theta) = 3 log theta + 7 log(1 - theta). Derivative:
3/theta - 7/(1-theta) = 0 gives theta = 3/10 = 0.3.

## E19

The estimator follows its rule correctly. The lesson is that MLE
with n = 2 carries no reliability. Report the estimate with its
sample size, or add a prior.

## E20

Yes, -1.5 beats -1.7075 on the same held-out set. Check: identical
held-out files for both models, no training files leaked into the
held-out set, the new model covers the held-out support, and the
mean (not total) is compared.

## L01

p_data: unknown law. p_hat: count/8 table. q: fittable formula.
p_hat from counts: divide each count by 8. Sum check: 0.375 + 0.25
+ 0.25 + 0.125 = 1. Memorizer wins training (-15.2451 vs -16.0)
because it copies the data. Table sums to 1.3: a count or mass is
wrong, or the table mixes counts with probabilities. p_hat misleads
when n is small: sampling wobble. Test: collect 8 more files and
compare their histogram to p_hat.

## L02

Likelihood: the model's probability of the observed dataset.
q_uni: (1/4)^8, l = -16 bits. Log step: log of a product is a sum,
monotone, so the argmax is unchanged. Code: sum of math.log2 over
items. O(n) time. Total grows with n. Mean divides by n so datasets
compare. Product underflows to 0.0 near n = 2000 in float64. The
sum does not. Training likelihood favors the memorizer by
construction. Protocol: split held-out data, score mean
log-likelihood there.

## L03

Support: outcomes with positive model mass. log(0) = -infinity, so
one zero-mass observed file kills the whole sum. Code: assert every
observed key has mass > 0. KL on q_bad: infinite. TV on q_bad:
0.375, finite. NaN at step 0: support violation, not a code bug.
Infinite KL is a message: the model denies observed reality. Repair:
add epsilon mass to every outcome (smoothing), the smallest change
that covers the data.

## L04

Entropy: expected surprisal of p. Cross-entropy: p-weighted
surprisal under q. KL: the surcharge, CE minus H. Toy: H = 1.9056,
CE = 2.0, KL = 0.0944. Derivation: sum p log(p/q) = sum p log p
with sign flipped minus sum p log q. Five lines: the three
one-liners from the lesson. O(K). Ratio form localizes blame per
outcome. Subtraction form explains. Negative KL: impossible. So the
code broke an assumption (swapped p and q, or mixed log bases).
Estimate H(p_data): use held-out data and a good model, then read
the cross-entropy as an upper bound.

## L05

Estimation: data to parameter. Evaluation: model to held-out
score. Toy: 7/10 heads gives theta = 0.7 by k/n. Grid search: max
at 0.7, matches formula. Prior-based estimate on n = 2: pulls 1.0
toward the prior mean. Held-out beating training: possible
honestly when the held-out set is easier (fewer rare patterns), but
check for leakage first. Splits: train fits, validation selects
among models, test reports once. Test is touched exactly once.

## Implementation and debug task

Correct version:

```python
import math

def score(counts, model):
    assert abs(sum(model.values()) - 1.0) < 1e-9, "model must sum to 1"
    total = 0.0
    m = 0
    for k, c in counts.items():
        mass = model.get(k, 0.0)
        if mass <= 0.0:
            raise ValueError(f"support violation on {k}")
        total += c * math.log2(mass)
        m += c
    return total / m
```

The bug: the broken version catches the log(0) case and adds 0.0,
which silently treats an impossible file as perfectly predicted.
The fix: raise instead of substituting. Support violations must be
loud.

## Changed-constraint scenarios

S1. Still works: likelihood sums, support checks, entropy/KL
definitions, held-out protocol. Breaks: any table over K outcomes
(O(K) memory is impossible at K = 2^64). First change: drop tables,
use models that give q(x) per item without enumerating the space
(implicit or autoregressive models).
S2. The memorizer still overfits in the only sense that matters:
it claims the 12 unseen patterns have probability 0. Support check
on fresh data exposes it. Entropy of p_hat is unchanged. The risk
is support, not the four seen masses.

## Research-critique question

Flaw 1: zero training loss means memorization, not learning (C01:
q = p_hat always wins training). Expose with held-out
log-likelihood.
Flaw 2: "learned the true distribution" confuses p_hat with p_data
(C01, C11: n = 10,000 still wobbles). Expose with a second sample
or confidence intervals.
Flaw 3: training loss alone cannot detect support collapse or bad
samples (C12: likelihood vs sample protocols). Expose with samples
drawn from the model and inspected.
