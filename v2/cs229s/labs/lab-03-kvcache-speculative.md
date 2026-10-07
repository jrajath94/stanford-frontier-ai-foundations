# Lab 03: KV cache and speculative decoding

Unit: cs229s-U03. Date: 2026-10-06.
Run: `python3 verify_lab03.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-03.md`.

## Objective

Verify U03 accounting on CPU: the two-term training FLOP
formula, decode FLOPs, KV-cache byte ratios, the prefill
identity (T=1 prefill = one decode step), inference
intensities, the batch toy model, speculative decoding
expected yield and speedup, the rejection-sampling
exactness (histogram test), and the speedup surface.

## Setup

```bash
python3 verify_lab03.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Verify the two-term training FLOP split on the toy.
2. Verify decode FLOPs and the attention add-on ratio.
3. Check KV-cache byte ratios in B, T, L, n.
4. Check prefill(T=1) equals one decode step.
5. Check prefill/decode intensities and the roofline
   side at the toy ridge.
6. Check the batch toy model monotonicity.
7. Verify E[k] and speedup numbers, check limits.
8. Histogram test: spec_step output matches p.
9. Sweep the speedup surface, find the bad case.

## Replication proposal (PROPOSED, not executed)

Question: does the speedup formula predict measured
speculative decoding speedup within 20 percent on a real
draft/target pair? Hypothesis: yes, with deviations
explained by the verification slope (C08). Method: one
open draft/target pair sharing a tokenizer, measure a
per position, c, and verification time vs gamma, grid
over gamma, compare measured vs formula speedup. Budget:
one GPU day. Failure criterion: systematic >30 percent
error with no slope explanation.
