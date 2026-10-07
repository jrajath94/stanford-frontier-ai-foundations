# Interview bank U01: Sequence models and transformer workload

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Define path length. Give the RNN and one-layer attention
values for T = 64.

B2. Write the next-token pretraining objective in one line.
Where do the labels come from?

B3. For n = 16, h = 4, T = 8, batch 1: write the shapes of Q,
the score matrix, and the merged attention output.

B4. State the 6P training FLOP rule and its two parts.

B5. Derive the per-layer parameter formula 12n^2 from the
four attention matrices and the MLP.

B6. Write the fp16 KV-cache byte formula in terms of T, L,
n. Evaluate it for T = 2048, L = 32, n = 4096.

B7. Name the term that makes naive attention quadratic in T,
and the term that keeps the KV cache linear.

B8. A model's loss fell but its task accuracy did not move.
Name two explanations.

## Deep ladder 1: the decode bottleneck

L1a. Define: what is one autoregressive decode step, with
shapes?
L1b. Toy: n = 8, L = 1, prompt of 3 tokens. How many KV
numbers exist after generating 2 tokens, in fp16 bytes?
L1c. Derive: show that per-step cache traffic grows as O(T)
per layer.
L1d. Implement and complexity: write the cache-append
update in numpy and state the per-step FLOP and byte
counts.
L1e. Compare: cached decode versus recompute decode under
equal memory. When does recompute win?
L1f. Debug: per-token time doubles when the prompt
doubles, but FLOPs per step look flat. What is wrong with
the FLOP-only view?
L1g. Critique: state the assumption behind "decode is
bandwidth-bound" and name the regime where it breaks.
L1h. Design: propose a measurement that finds the batch
size where decode flips to compute-bound. Name the
controls.

## Deep ladder 2: the MLP/attention crossover

L2a. Define: split per-layer per-token FLOPs into MLP,
projections, and scores.
L2b. Toy: n = 8. Compute each part at T = 4 and at T = 64.
L2c. Derive: solve for the crossover T where scores equal
MLP plus projections.
L2d. Implement and complexity: write `layer_flops(n, T)`
and state the complexity class of each term in T.
L2e. Compare: at n = 4096, T = 2048, which part dominates
and by what ratio?
L2f. Debug: an attention kernel got 2x faster but
end-to-end time moved 5 percent. Explain with numbers.
L2g. Critique: the derivation assumed d_ff = 4n. How does
SwiGLU change the crossover, and what else must be
recomputed?
L2h. Design: propose a profiling experiment that finds
the measured time crossover on a real GPU. Name the
confounders.

## Analytical exercises

A1. P = 1.3B, D = 300B tokens. Compute training FLOPs via
the 6P rule. Convert to days on 512 devices at 1.5e14
FLOP/s sustained each. Show each step.

A2. B = 4, T = 4096, n = 4096, L = 32, fp16. Compute
residual-stream activation bytes in GiB. Then compute the
naive score-matrix bytes for h = 32. Which term is larger,
and by what factor?

## Implementation / debug task

D1. The snippet below should compute one causal attention
step at position t, but the outputs are wrong. Find the bug,
fix it, and state the invariant that catches it.

```python
import numpy as np
def step(q, K, V, t):
    # q: (d,), K: (T, d), V: (T, d), full sequence
    s = (q @ K.T) / np.sqrt(q.shape[0])
    a = np.exp(s - s.max())
    a = a / a.sum()
    return a @ V
```

## Changed-constraint scenarios

S1. Constraint change: the KV cache may not be stored
(memory cap hit). Derive total FLOPs to generate 100
tokens from a 10-token prompt with P parameters and no
cache. How does the answer scale in the output length?

S2. Constraint change: infinite memory bandwidth but
today's compute. What limits decode latency now? What
limits training throughput? Give one design change for
each.

## Research critique

R1. A paper claims a new attention variant trains 2x
faster than "the transformer baseline" on T = 512,
n = 1024. List the five audit questions you ask before
believing the claim, and for each, state what answer
would invalidate it.
