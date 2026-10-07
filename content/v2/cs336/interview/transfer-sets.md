# Transfer sets , 10 changed-scenario sets

Closed-book. Each set changes one constraint from the course toys.
Full keys in `keys-transfer.md`.

## Set 1 , scaling with a data cap

The fitted law says D/N=20 at C=1e23, but the corpus caps at 3e11
tokens. (a) Allocate. (b) State what breaks in the unconstrained
argument. (c) Name the experiment that would change your answer.

## Set 2 , serving under a memory wall

Batch 64, T=8192, GQA-8 7B model, one 80 GB GPU. (a) Compute the
KV cache. (b) It does not fit with weights. List three fixes in
order of preference. (c) Which fix changes quality, and how do
you check?

## Set 3 , eval with a tiny slice

The safety slice has n=25, scores 0.92. (a) Compute the 95%
interval. (b) Can you ship on this slice? (c) Design the cheapest
valid safety gate.

## Set 4 , dedup at 10x scale

n=1e8 documents, Bloom budget 8 GB. (a) Bits per item and the
optimal k. (b) Compute the FPR. (c) The team wants exact dedup.
Cost it.

## Set 5 , SFT with tool results

A 5-call tool trace: the tool returns 200 tokens per call.
(a) Which tokens carry loss, and why? (b) The tool output
contains user PII. What breaks? (c) Redesign the trace handling.

## Set 6 , RL with uniform groups

GRPO, G=8, 90% of prompts solve at 100%. (a) What happens to the
gradients? (b) Diagnose from the advantage statistics. (c) Fix
the data pipeline.

## Set 7 , async with a slow trainer

Rollouts 200/min, trainer 100/min, queue bound 10k sequences.
(a) When does the queue fill? (b) What happens to staleness?
(c) Set the shed policy.

## Set 8 , multimodal context overflow

12 images at 256 tokens + 2000 text tokens, limit 4096. (a) Does
it fit? (b) Three ways to fit it, with costs. (c) Which
preserves the most task performance, and how would you test?

## Set 9 , guest source surfaces

Slides for G1 appear (40 pages, no recording). (a) Update the
source record. (b) What can now be taught, with what anchors?
(c) What stays unknown?

## Set 10 , the full arc

A 7B run is planned: C=6e21 FLOPs, 40 RPS serving, held-out eval,
RL after SFT. (a) Allocate N/D. (b) Size the serving fleet (HYP
inputs allowed, labeled). (c) List the three gates that must pass
before RL starts.
