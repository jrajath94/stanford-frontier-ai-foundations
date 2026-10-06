# Interview bank, U10 LLM inference, quantization, and alignment

Date: 2026-10-06. Questions only. Keys in
interview/keys-u10.md. Closed-book. Do not read the keys
first. Toys as in the lesson. C04 numbers are authored
arithmetic, not benchmarks.

## Breadth (6)

B1. Compute softmax(z/T) at T = 0.5 and T = 2.0 for
z = [2.0, 1.0, 0.5, 0.1]. What happens to entropy?
B2. KV bytes for L=12, h=8, d_h=64, n=512, fp16.
What is the per-token increment?
B3. Quantize W's entry 0.5 with s = 0.013725, zp =
87. Give q and w_hat.
B4. 7B fp16 at 2 TB/s: compute the tok/s ceiling.
What changes at INT4?
B5. Write the BT loss, the PPO clipped objective,
and the DPO loss. One line each.
B6. State the C12 audit denominators. Name the 4
title-level concepts.

## Deep ladder D1, decode economics (5 follow-ups)

D1.1. Define the KV cache in one sentence.
D1.2. Toy: compute the 12.00 MiB and the 24576
bytes/token.
D1.3. Derive why prefill is O(n^2) and decode is
O(n) per token.
D1.4. Implement/debug: OOM at half the planned
context. Name the cause.
D1.5. Changed constraint: n = 8192 on a 12 GB
card. Price three levers.

## Deep ladder D2, alignment (5 follow-ups)

D2.1. Define SFT, BT, PPO, DPO in one sentence
each.
D2.2. Toy: compute the BT loss 0.3412 and the DPO
loss 0.6588.
D2.3. Derive the DPO implicit reward from the
KL-regularized objective (sketch).
D2.4. Implement/debug: DPO loss falls but both
log-ratios fall too. Diagnose.
D2.5. Research critique: "lower DPO loss means a
better model." Attack it.

## Analytical/quantitative (2)

A1. PPO: (rho=1.3, A=0.5) and (rho=0.5, A=-0.4),
e=0.2. Compute both objectives and state which
term the min picks.
A2. Reward hacking toy: remove the length bonus
and recompute the winner. Then propose a proxy
without this exploit and name its residual risk.

## Implementation/debug (1)

I1. A colleague's top-p sorts ascending and keeps
the first k by that order. Describe the symptom
on the toy at p = 0.9, name the fix, and write
the assert that prevents recurrence.

## Changed-constraint scenarios (2)

S1. Deploy to a phone: 2 GB RAM. The 7B model is
3.5 GB at INT4. Give a concrete feasible config
with byte math.
S2. Preference labels have 30% flips. Which
degrades faster, PPO or DPO? Justify.

## Research-critique (1)

R1. "The course's alignment coverage is
complete: eight titles." Attack with the audit
and design the re-audit trigger.
