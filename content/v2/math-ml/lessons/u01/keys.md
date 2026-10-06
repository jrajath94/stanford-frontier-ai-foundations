# Answer keys, U01 lesson

Date: 2026-10-06. Kept separate from the lesson per the assessment rule.
Test-mode: read the question file first, answer closed-book, then check
here.

## E01

f(-2) = 5, f(-1) = 2, f(0) = 1, f(1) = 2, f(2) = 5. Image = {1, 2, 5}.
|image| = 3. Note the collision: -2 and 2 map to 5.

## E02

Rule on {1, 2}: g(1) = 9 and g(1) = 10. It breaks the contract "each
input maps to exactly one output". Minimum sufficient: one input with
two outputs disqualifies the rule.

## E03

Not a function: id 7 has two prices. Training sees two targets for one
input. Squared error cannot fit both. The gradient pulls toward their
mean while the data claims both are exact. Fix: deduplicate ids or add a
feature that distinguishes the two rows.

## E04

sum_{i=1}^{4} (2i + 3) = 5 + 7 + 9 + 11 = 32.

## E05

sum_{i=1}^{3} 2i = 2 + 4 + 6 = 12.

## E06

Failure: one symbol, two kinds. Fix: name the kind at first use, for
example n for the scalar count and c for the vector of counts. Never
reuse a symbol across kinds in one derivation.

## E07

Slope unit = seconds / meter. Check: distance (meters) times slope
(seconds/meter) = seconds. The units cancel to the target unit.

## E08

Income decides nearly every neighbor pair. Its range (200000) dwarfs the
age range (100), so income differences dominate the distance sum. Fix:
scale features before any distance computation.

## E09

[3, 4, 5]: sum 12, product 60. [3, 4, 6]: sum 13, product 72. Relative
moves: sum 8.33 percent, product 20 percent. The product moved more.

## E10

log10(4 * 25) = log10(100) = 2.0. log10(4) + log10(25) = 0.60206 +
1.39794 = 2.0. Equal.

## E11

Length 8: direct count 64^8 = 281474976710656. Log10 = 14.449439791871097.
Log-sum: 8 * log10(64) = 14.449439791871097. Both survive. Length 400:
direct computation overflows (Python raises OverflowError. Float64 gives
inf with a warning). Log-sum = 400 * log10(64) = 722.4719895935548,
finite. The log path survives at length 400.

## E12

Direct: 0.9^3 = 0.7290000000000001. Log space: 3 * log(0.9) =
-0.31608154697347884. Exp gives 0.7290000000000001. The paths agree.

## E13

The conclusion is wrong. exp(-921.034) underflows to 0.0, but the
log-likelihood -921.034 is a finite, meaningful number. Zero here means
"below float range", not "impossible". Compare models in log space. Never exponentiate a large negative log-likelihood to judge it.

## E14

||[6, 8]|| = sqrt(36 + 64) = sqrt(100) = 10. ||2[3,4]|| = ||[6,8]|| = 10
= 2 * 5. Scaling rule: ||c*v|| = |c| * ||v|| for scalar c.

## E15

v dot w = 1*3 + 2*4 = 11. v + w = [4, 6]. (v+w) dot (v+w) = 16 + 36 = 52.
Right side: 5 + 25 + 2*11 = 52. Verified.

## E16

Shape (4,). Entry j is the sum of column j over the 3 rows.

## E17

Reduce axes (1, 2, 3): mean over channels, height, width per image.
Result shape: (32,). Each entry is the mean pixel value of one image.

## E18

```python
def total(xs):
    s = 0
    for v in xs:
        s = s + v
    return s
```

total([2, 5, 7]) = 14. total([]) = 0. Zero is the right convention: it is
the additive identity, so total(a + b) = total(a) + total(b) still holds
when one list is empty.

## E19

```python
def prod(xs):
    p = 1
    for v in xs:
        p = p * v
    return p
```

With the accumulator at 0, prod([2, 5, 7]) returns 0 instead of 70. The
failing test: assert prod([2, 5, 7]) == 70.

## E20

The truncated plot stretches the 65-thousand-dollar rise to fill the
frame, so a 20 percent climb reads as a dramatic surge. The honest
version is required because the axis limits are part of the claim.

## E21

Two honest options: find the missing row and add the seventh dot, or
plot 6 dots and label the figure "6 of 7 rows. 1 row excluded (reason)".
What is not honest: plotting 6 dots while the caption says 7.

## E22

```python
def check_unit_sum(v, tol=1e-9):
    assert all(t >= 0.0 for t in v), "negative item in %r" % (v,)
    assert abs(sum(v) - 1.0) < tol, "sum is %r" % (sum(v),)
```

Passing: [0.2, 0.3, 0.5]. Failing: [0.2, 0.3, 0.4] (sum 0.9).
[0.6, 0.5, -0.1] (negative item).

## E23

Cause: legitimate float noise exceeds 1e-16, so correct code fails.
Better: 1e-9 for float64-scale sums. Reason: float64 rounding noise
sits near 1e-16 per operation, and a sum of several terms accumulates
a few units of that. 1e-9 clears the noise while still catching real
errors like 0.9 vs 1.0.

## E24

0.1 + 0.2 = 0.30000000000000004, so == gives False. Correct comparison:
abs((0.1 + 0.2) - 0.3) < 1e-9.

## E25

float64 holds about 16 decimal digits. 1e16 uses all 16 for the leading
digits, so the +1 falls below the representable grid and rounds away.
1e16 + 1 evaluates to 1e16, and the subtraction gives 0.0.

## E26

No. default_rng() with no seed draws entropy from the OS, so its
sequence differs on every run. Only the seeded runs match.

## E27

Symptom: every epoch sees the batches in the identical order, because
the generator restarts at the same seed each epoch. Shuffling has no
effect across epochs. The model sees a fixed sequence.

## L01 ladder (strong answers)

1. A function maps each input to exactly one output.
2. f(x) = 2x + 3 on {1, 2, 3, 4} gives {5, 7, 9, 11}. Each input has one
   arrow, no arrow is shared.
3. f(3) = 2*3 + 3 = 9.
4. Code: the list comprehension in C01. Check: image == [5, 7, 9, 11].
5. Relations allow multiple outputs. Models must be functions or
   training has no single target per input.
6. Debug: the table is not a function at key 7. The failure surfaces as
   two different targets for one input. Fix the data (deduplicate or add
   a distinguishing feature), not the model.
7. Critique: "exactly one output" is too strict for one-to-many real
   mappings (one query, many valid answers). The honest fix is to model a
   distribution over outputs, which is U04 content, not to pretend the
   data is functional.

## L02 ladder (strong answers)

1. log_b(x) is the power to which b must be raised to get x.
2. log10(200) = 2.3010299956639813 directly. Log10(2) + log10(100) =
   0.30103 + 2 = 2.3010299956639813.
3. a = b^{log_b(a)}, c = b^{log_b(c)}. A*c = b^{log_b(a)+log_b(c)}, so
   log_b(a*c) = log_b(a) + log_b(c).
4. Code: the math.fsum loop in C05. Check: total within 1e-9 of
   -921.0340371976182. Exp(log(0.125)) recovers 0.125.
5. Direct product: O(n), underflows past a few hundred tiny factors. Log
   sum: O(n), stays finite. Equal budgets, different failure modes.
6. Debug: log(0) raises. Fix A: clamp inputs to [eps, 1] with
   eps = 1e-12. Price: a tiny bias. Fix B: mask out zero entries before
   the log. Price: code complexity and a changed objective on those
   entries.
7. Critique: log space hides modeling problems when the tiny
   probabilities come from a misspecified model rather than from honest
   uncertainty. If every likelihood is astronomically small, the model
   may be wrong, not just the arithmetic. Check calibration (U04-C12)
   before blaming the float.

## D01

Bug: the loop adds t instead of t*t, so total = 3 + 4 = 7 and the
function returns sqrt(7) = 2.64575131 instead of 5.0. Caught by the
first test: assert vector_norm([3.0, 4.0]) == 5.0. Fix: total = total +
t * t. The zero-vector test would pass either way, which is why the
[3, 4] test is the load-bearing one.

## S01

House vector h = [size, rooms], weight vector w = [2, -1]. Norm, dot,
and addition follow the same code with n = 2 items. Shapes stay (2,).
What changes: the units differ per axis (sq ft vs count), so the dot
product mixes units and the norm mixes them too. Honest follow-up: scale
or standardize before any dot or norm, or the size axis dominates.

## S02

0.99^400 = 0.017950553275045137, computed directly with no underflow.
Log space: 400 * log(0.99) = -4.02013434140058. Both paths work here.
Choose log space anyway when the factor count grows, when factors vary
in scale, or when gradients must sum per-example terms. Choose direct
only for a handful of benign factors.

## R01

Counterexample 1: a float32 probability of 1e-40 rounds to 0.0, and the
log-loss then crashes on log(0). Counterexample 2: the C11 sum shows
float32 already errs at 1.2e-7 on a ten-term sum. A million-term loss
accumulates far worse. float32 is safe only when values stay in a benign
range, reductions use pairwise or compensated summation, and no log of a
tiny probability is taken without clamping. The claim as stated is
false. The safe version names these three conditions.
