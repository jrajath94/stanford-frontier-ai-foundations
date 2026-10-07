# U15 interview key

## B1-B6

B1. Pretrain (LM objective), midtrain (continued LM, new mix),
post-train (behavior: SFT/RL).
B2. Stretches the RoPE ruler 50x, needs long-document training.
B3. Train/serve mismatch reads as role confusion.
B4. Loss on assistant tokens only.
B5. 2*d*r per adapted pair.
B6. Weights, optimizer moments, RNG state, data position.

## L1

L1.1 (T*n - tokens)/(T*n).
L1.2 47.6% vs 22.0%, 344 bins.
L1.3 Blocks cross-document attention.
L1.4 Greedy first-fit, no bin overflows.
L1.5 Padding wastes half, debug: mask bug leaks context.
critique: greedy, experiment: canary tokens.

## L2

L2.1 Base-suite loss per checkpoint, after minus before.
L2.2 Mean +0.15 nats.
L2.3 Forgetting is measured, not assumed.
L2.4 The delta, replay flattens it.
L2.5 Replay/LR/freeze, debug: silent capabilities, critique:
toy, experiment: replay sweep.

## E1

(a) 2*4096*16*2*32 = 8.39M. (b) 2*4096^2*32 = 1074M. (c) Full
(or high-r): a new language is a big shift, rank 16 cannot carry
it. Rubric: (a) 1 pt, (b) 1 pt, (c) 2 pts. Red flag: picking
LoRA for maximal change.

## E2

(a) 7,500 bad exposures. (b) 750 bad exposures at 3x cost: worth
it when behavior quality is the product, the tradeoff is
7500->750 bad lessons vs 3x annotation. Rubric: (a) 1 pt, (b) 2
pts, plus the judgment 1 pt.

## D1

Bug: system tokens get mask 1, so the model learns to emit
system instructions. Fix: mask 1 only for assistant. Rubric:
find 2 pts, fix 1 pt, state the consequence 1 pt.

## S1

Isolate: a 100k document in 2048-bins wastes 50 bins of padding
around it and breaks packing efficiency, give it dedicated
batches or truncate to windows.

## S2

Incompatible until proven otherwise: the adapter's base changed.
run the eval suite on v2+adapter before any use.

## R1

Gaps: (1) no data card: the training set is unknown, publish
it. (2) no mask: the objective is unknown, describe it. (3) no
regression check: the win may cost base ability, report base
slices.
