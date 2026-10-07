# Lab 07: Contrastive loss, LoRA accounting, retrieval metrics

Date: 2026-10-06. Unit: cs229-U13.
Work the tasks, then check keys-lab-07.md. Run all code.

Setup: numpy only. Seeds as stated per task. No other installs.

## Task 1: SIMCLR loss by hand

Batch B = 4, embedding dimension 6, seed 9. Build
two views per example as in the lesson (shared
component weight 0.8, noise weight 0.6),
normalize to unit norm, and compute the SIMCLR
loss of SL-04. Report the total loss and the
per-example terms. Then move each positive pair
10 percent closer (same spherical step as the
lesson) and report the new loss. State the
monotonicity this confirms.

## Task 2: LoRA parameter table

For a square layer with d = 4096 and for a
rectangular layer with d_out = 11008, d_in =
4096: compute the dense-update parameter count
and the LoRA count for r in {4, 8, 16, 64}.
Report the full table and the reduction ratio
at r = 8 for each layer. State the memory
nuance from SL-03 in one sentence.

## Task 3: retrieval metrics by hand

Judgment set: one query, corpus of 6 documents,
gold relevant set R(q) = {d2, d4, d5} with
grades s* = {d2: 3, d4: 2, d5: 1, others: 0}.
The system returns the ranked list
[d1, d2, d3, d4, d5, d6]. Compute Recall@3,
Recall@5, DCG@5, IDCG@5, NDCG@5 by hand (show
the arithmetic), then verify with code.
State which metric punishes the d1-at-rank-1
miss and which does not.

## Deliverable

A short log: the three result blocks with numbers. No
essay. The numbers must match keys-lab-07.md within the
stated tolerance.
