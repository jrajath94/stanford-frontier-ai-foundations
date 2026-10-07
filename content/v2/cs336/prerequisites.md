# Prerequisites , cs336 U01-U09

Shared bridge modules P01-P24 live once at
`../shared/prerequisites/` and are linked here, never rebuilt.
Each unit lesson also carries a local remediation block for the exact
objects it uses first.

## Unit prerequisite map

| Unit | Bridges | What the unit assumes on day one |
|------|---------|----------------------------------|
| U01 | P01, P02, P06, P13 | integer arithmetic, Python loops and dicts, conditional probability, text as sequences |
| U02 | P03, P11, P12, P15 | matrix shapes, autodiff graphs, tensor views, FLOPs versus bytes |
| U03 | P05, P11, P12, P14 | chain rule, normalization, broadcasting, attention block layout |
| U04 | P04, P14, P15 | low-rank structure, attention cost, memory hierarchy |
| U05 | P05, P09, P11, P12 | gradients, convexity and smoothness, optimizer state, stable softmax |
| U06 | P12, P15, P16 | dtypes and devices, bandwidth and latency, collectives vocabulary |
| U07 | P12, P15 | tensor programs, SRAM versus HBM, roofline intuition |
| U08 | P12, P16 | autograd semantics, message passing, partitioning |
| U09 | P14, P15, P16 | transformer block math, interconnect topology, consistency |

## Bridge reading order for this course

P01, P02, P03, P06, P05, P11, P12, P13, P14, P04, P09, P15, P16, P08.
(Only the bridges this half uses. P07, P10, P17-P24 serve the second half.)

## Diagnostic

Attempt closed-book. Score each item 0 (cannot start), 1 (partial), or 2
(full). Remediation follows the table. The answer key is
`keys/diagnostic_key.md`, do not open it first.

D1. Write the UTF-8 bytes of the character "é" (U+00E9) as integers.
    Link: `../shared/prerequisites/p13_language.md`.
D2. A Python dict maps pairs to counts. Write code that returns the pair
    with the highest count and its count. Link: `../shared/prerequisites/p02_python.md`.
D3. For events A and B, write P(A|B) in terms of P(B|A), P(A), P(B).
    Link: `../shared/prerequisites/p06_probability.md`.
D4. X has shape (4, 8) and Y has shape (8, 3). Give the shape of X @ Y and
    the FLOP count of the product. Link: `../shared/prerequisites/p03_vectors.md`.
D5. State the chain rule for z = f(g(x)) with one line of working.
    Link: `../shared/prerequisites/p05_calculus.md`.
D6. Draw the forward and backward data flow of y = xW + b and name every
    saved tensor. Link: `../shared/prerequisites/p11_neural_nets.md`.
D7. A tensor has shape (2, 3, 4) and stride (12, 4, 1). How many elements
    does it hold, and is it contiguous? Link: `../shared/prerequisites/p12_pytorch.md`.
D8. Define perplexity of a sequence under a model in one sentence, then
    give its formula. Link: `../shared/prerequisites/p13_language.md`.
D9. Sketch one transformer block and label Q, K, V, the residual path, and
    the norm placement. Link: `../shared/prerequisites/p14_transformer.md`.
D10. A matrix has rank 2. What does that say about its SVD, in one
     sentence? Link: `../shared/prerequisites/p04_spectral.md`.
D11. Write one step of gradient descent for parameters w with learning
     rate eta. Link: `../shared/prerequisites/p09_optimization.md`.
D12. A GPU kernel reads 1 GB and does 10 GFLOP. Is it memory-bound or
     compute-bound on a 1 TB/s, 100 TFLOP/s device? Show the arithmetic.
     Link: `../shared/prerequisites/p15_hardware.md`.
D13. Name the three collectives that move a full gradient to every rank
     and back in data parallel training. Link: `../shared/prerequisites/p16_distributed.md`.
D14. Define cross-entropy H(p, q) for discrete distributions and say what
     it measures. Link: `../shared/prerequisites/p08_information.md`.

## Score guide

- 24-28: start U01 directly.
- 16-23: read the linked bridge sections for missed items first.
- Below 16: work P01 through P14 in order before U01.

## Local remediation rule

Every unit lesson opens with a remediation block that re-teaches the
smallest set of objects the unit uses before any other course does
(for example U01 re-teaches code points and UTF-8 even though P13 covers
them). The block is self-contained and assessed.
