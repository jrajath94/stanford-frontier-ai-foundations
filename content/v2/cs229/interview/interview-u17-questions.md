# Interview bank: U17, appendices, historical supplements, research synthesis

Date: 2026-10-06. Keys in keys-u17.md.
Provenance: original practice questions
derived from SRC-01 Appendix A and
course synthesis. No employer
attribution.

## Breadth questions

Q01: State the four Gaussian/KL
identities.
Q02: What is the factor analysis
model, and where does its math
survive in the 2026 notes?
Q03: What does the KL chain rule buy
you for sequence models?
Q04: Name the four ablation rules.
Q05: State the three uncertainty
honesty rules.
Q06: Name the six deployment gaps.

## Deep ladders (5 follow-ups each)

L01 (KL identities):
F1: Write Lemma A.1.3.
F2: Compute the KL for N([1,0],I)
vs N([0,0],I).
F3: Derive it from the log
densities.
F4: When does A.1.3 not apply?
F5: Use the chain rule to decompose
a sequence-level KL into token
terms.

L02 (research craft):
F1: Define a matched-budget
ablation.
F2: A paper ablates five things at
once. What is wrong?
F3: Compute the standard error for
the ten-seed returns.
F4: The error bars overlap. What do
you conclude?
F5: Design the acceptance gate for
shipping a course model.

## Analytical exercises

A01: Prove Lemma A.1.1 for the
scalar case from the Gaussian
moment-generating function or
direct convolution.
A02: Show that the perceptron
mistake bound (R/gamma)^2 is
independent of the sample size n,
and explain why that matters for
the online setting.

## Implementation and debug task

D01: Implement the KL Monte Carlo
check. Then change Q to N([0,0],
2I) while keeping the A.1.3 formula.
Report the disagreement and explain
which identity was violated.

## Changed-constraint scenarios

S01: You may not assume Gaussian
noise anywhere. Which identities
survive, and what replaces the
Kalman update?
S02: The reviewer asks for an
ablation but you have compute for
only one extra run. What single
ablation do you run, and what claim
can it support?

## Research critique

R01: "Our method beats the baseline
0.83 to 0.79, so it is better."
Critique with the uncertainty
rules.
