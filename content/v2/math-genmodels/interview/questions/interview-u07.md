# Interview bank: U07 (questions)

Unit: math-genmodels-U07. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u07.md.

## Breadth (6)

Q1. Why is the autoregressive factorization exact while the VAE
bound is not?
Q2. What is exposure bias, and which training choice causes it?
Q3. Why does the causal mask have to be inside every layer?
Q4. What breaks if the attention scale 1/sqrt(d) is removed?
Q5. How does temperature change sampling without changing the
model?
Q6. Why can two models with the same architecture not have
their perplexities compared across tokenizations?

## Deep ladders (2 x 5)

L1 (likelihood):
1. State the chain rule for L = 3.
2. Compute log p(abc) on the toy.
3. Convert to perplexity and interpret it.
4. Explain why the number is exact.
5. A VAE reports -3.4 on the same toy. What can you conclude?

L2 (the transformer):
1. Draw the 3x3 causal mask from memory.
2. Work the attention head: scores, weights, output.
3. Cost attention at L = 512, d = 64.
4. Explain KV caching and its memory price.
5. Sampling is slow at L = 8192. Propose two fixes and their
   costs.

## Analytical exercises (2)

A1. Prove the chain-rule product sums to 1 over all sequences
given valid conditionals.
A2. Derive the masked softmax row for the toy head, showing
where -inf goes.

## Implementation and debug (1)

D1. Loss is excellent, samples are garbage, masks are present.
Name the most likely bug, the fix, and the test.

## Changed-constraint scenarios (2)

T1. Context must reach 32768 on fixed memory. Redesign
attention and name the loss.
T2. You need bidirectional context and exact sampling. Show the
conflict and the closest design.

## Research critique (1)

R1. "Our masked model has lower pseudo-perplexity than their AR
model." Attack with two arguments and propose the fair test.
