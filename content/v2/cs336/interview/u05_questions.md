# U05 interview bank , questions

Closed-book. Keys in `u05_key.md`. Quotas: 6 breadth, 2 deep ladders of
5, 2 analytical, 1 implementation/debug, 2 changed-constraint, 1
research-critique.

## Breadth (6)

B1. Why is the naive softmax cross-entropy numerically unsafe?
B2. What are the inputs and labels for next-token training?
B3. What do Adam's m and v track?
B4. How does AdamW differ from Adam with L2?
B5. What does gradient clipping guarantee, and what does it not?
B6. Name the six components of a resumable training checkpoint.

## Deep ladders (2 x 5)

L1. Adam from scratch.
- L1.1 Write the moment updates.
- L1.2 Toy: one AdamW step by hand on w=[1.0,-0.5], g=[0.3,-0.2],
  lr=1e-3, lam=0.01.
- L1.3 Derive the bias correction from the geometric series.
- L1.4 Implement adamw_step, state the t=1 check.
- L1.5 Compare AdamW with Lion at matched compute, debug the
  transferred-lr divergence, critique the unbiased-gradient
  assumption, propose the lr-sensitivity experiment.

L2. Schedules.
- L2.1 Define warmup, cosine decay, WSD.
- L2.2 Toy: lr at steps 0/50/100/550/1000 for warmup 100, total
  1000, peak 3e-4, floor 3e-5.
- L2.3 Derive why a wrong cosine total under-converges.
- L2.4 Implement both schedules, state the endpoint checks.
- L2.5 Compare cosine with WSD under an early stop, debug the
  half-stopped cosine, critique the known-total assumption, propose
  the extension experiment.

## Analytical exercises (2)

E1. A run uses AdamW, lr=3e-4, global batch 2M tokens via accumulation
with microbatch 8k tokens. (a) How many accumulation steps? (b) A bug
drops the /a division. What is the effective lr? (c) The loss is nan
at step 100. Give the ordered checklist.
E2. Checkpoint at step 5000 resumed with a fresh RNG. Two symptoms are
possible: a loss jump, and a silent divergence. Explain each
mechanism and the one-line fix.

## Implementation/debug task (1)

D1. This clipping function silently changes the training dynamics even
though the loss looks fine. Find the bug and fix it.

```
def clip_bad(grads, c=1.0):
    total = sum((g**2).sum() for g in grads)   # <-- suspect
    if total > c:
        scale = c / total
        return [g * scale for g in grads]
    return grads
```

## Changed-constraint scenarios (2)

S1. The GPU holds only 1/8 of the target batch. Name the change, the
exactness guarantee, and the two things that can break it (loss
reduction convention, per-microbatch randomness).
S2. The run may be extended 2x or stopped at half at any time. Pick
the schedule and justify against cosine.

## Research critique (1)

R1. A paper claims a new optimizer "beats AdamW" with a single run per
method, no lr sweep, and wall-clock plots only. List three gaps and
the measurement for each.
