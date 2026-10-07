# Lab 01: Sequence workload accounting

Unit: cs229s-U01. Date: 2026-10-06.
Run: `python3 verify_lab01.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-01.md`.

## Objective

Verify the U01 accounting rules on toys with numpy: path
lengths, attention shapes, the 6P rule, parameter counts,
activation memory, KV cache growth, the MLP/attention
crossover, and the `T^2`/`T` scaling split.

## Setup

```bash
python3 -m venv /tmp/lab01venv  # only if numpy is absent
/tmp/lab01venv/bin/pip install numpy
python3 verify_lab01.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Confirm RNN path length grows with `T` while attention
   path length stays 1.
2. Assert every attention shape from `(B, T, n, h)` and
   check softmax rows sum to 1.
3. Verify the 6P rule on a toy linear layer with manual
   gradients.
4. Recompute the toy (1664) and real-scale (6.57B)
   parameter counts.
5. Check activation memory ratios when `T` doubles.
6. Check KV cache grows by `2Ln` numbers per appended
   token.
7. Sweep `T` and locate the MLP/attention crossover near
   `6n`.
8. Confirm scores scale 4x and cache 2x when `T` doubles.
9. Build the toy perplexity/accuracy disagreement case.

## Replication proposal (PROPOSED, not executed)

Question: does the FLOP crossover `T = 6n` predict the
measured time crossover on a real GPU? Hypothesis: the
time crossover sits within 2x of `6n` in FLOP terms.
Method: one small transformer, sweep `T`, profile
per-layer MLP versus attention time, report both
crossovers with hardware and dtype named. Budget: one
GPU hour. Failure criterion: the time crossover differs
by more than 4x with no memory-based explanation.
