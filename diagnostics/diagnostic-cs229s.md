# Diagnostic: cs229s readiness

Date: 2026-10-06. 12 questions covering P04, P11, P12,
P13, P14, P15. Keys in `keys-cs229s.md`. Miss more than
4: work the linked bridge modules before U01.

## Questions

D01 (P12). A tensor has shape (2, 4, 8). How many
numbers does it hold? Write the shape of its transpose
over the last two axes.

D02 (P12). Define broadcasting. Give the result shape
of (4, 1) + (8,).

D03 (P11). Write one gradient-descent step for a scalar
parameter. Name each symbol.

D04 (P11). Forward pass costs F FLOPs for a dense
layer. What does the backward pass cost, and why?

D05 (P13). Define perplexity in one sentence. A uniform
model over 1000 tokens has what perplexity?

D06 (P13). What is teacher forcing?

D07 (P14). For n = 8, h = 2, T = 4: write the shapes
of Q, the score matrix, and the merged attention
output at batch 1.

D08 (P14). What does the KV cache store, and why does
it avoid recomputation?

D09 (P15). Define arithmetic intensity with units.
A kernel does 1e9 FLOPs and moves 1e9 bytes. What is
its intensity, and which roofline side is it on at
ridge 150?

D10 (P15). Name the GPU memory levels from fastest to
slowest.

D11 (P04). A matrix has condition number 1e6. In one
sentence, what does that say about solving linear
systems with it?

D12 (P04). What is a low-rank approximation, and when
does it save work?

## Scoring

10-12: proceed to U01. 8-9: review the missed bridges
lightly. Below 8: work the linked modules fully
(see prerequisites.md), then retake.
