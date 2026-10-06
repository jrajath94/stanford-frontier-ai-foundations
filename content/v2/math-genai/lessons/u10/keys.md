# Answer keys, lesson 10 (LLM inference, quantization, and alignment)

Date: 2026-10-06. Computed 2026-10-06, numpy 1.26.4, float64,
seed 0 where RNG is used. Ground truth: compute_run5b.py.
C04 numbers are authored arithmetic, not benchmarks.

## E01

z/T then softmax. T=0.5: [0.8282, 0.1121, 0.0412,
0.0185], pmax 0.8282. T=1.0: [0.5745, 0.2114, 0.1282,
0.0859], pmax 0.5745. T=2.0: [0.4056, 0.2460, 0.1916,
0.1569], pmax 0.4056. Verified.

## E02

Entropies: 0.8754, 1.6174, 1.9017 bits. Trend:
entropy rises in T, temperature spreads mass from
the argmax to the tail.

## E03

Sorted desc: [0.5745, 0.2114, 0.1282, 0.0859]. Cumsum:
0.5745, 0.7859 (< 0.9), 0.9141. Keep-while-cumsum-
below-p: [0, 1]. Renorm: [0.7311, 0.2689, 0, 0].
Verified.

## E04

2 12 8 64 512 2 = 12582912 bytes = 12.00 MiB.
Per-token increment: 2 12 8 64 2 = 24576 bytes.

## E05

Prefill: 2 12 8 512^2 64 = 3221225472. Decode per
token: 2 12 8 512 64 = 6291456. Ratio 512 = n.
Verified.

## E06

s = 0.01372549, zp = 87. q = round(0.5/0.01372549 +
87) = round(123.428) = 123. w_hat = (123 - 87)
0.01372549 = 0.494118. err = 0.005882 = the max
err over the matrix, within s/2 = 0.006863.

## E07

14e9 bytes / 2e12 B/s = 7.00e-3 s/token = 7.00
ms/token. Ceiling 1/7.00e-3 = 142.9 tok/s.
Authored arithmetic.

## E08

3.5e9/2e12 = 1.75e-3 s = 1.75 ms/token -> 571.4
tok/s. Assumption: decode reads all weights once
per token (dense model), bandwidth 2 TB/s
sustained. Authored arithmetic, not a benchmark.

## E09

softmax([1.5, 0.5, -0.5]): max 1.5, exps [1,
0.367879, 0.135335], sum 1.503215, p0 = 0.665241,
-log = 0.407606. Second token: 0.325397. Total
0.733003 nats. Verified.

## E10

Margin 0.9. P = 1/(1+e^-0.9) = 0.7109. Loss =
-log(0.7109) = 0.3412 nats. Verified.

## E11

Case 1: unclipped 1.3 0.5 = 0.65, clipped
min(1.3,1.2) 0.5 = 0.6. min picks 0.6 (the
clip binds). Case 2: unclipped 0.5 (-0.4) =
-0.2, clipped 0.8 (-0.4) = -0.32. min picks
-0.32 (pessimism). Verified.

## E12

Margin = 0.1 (0.4 - (-0.3)) = 0.07. Loss = -log
sigmoid(0.07) = 0.6588. Implicit rewards: chosen
0.0400, rejected -0.0300. Verified.

## E13

0.5 log2(1.5) + 0.3 log2(0.9) + 0.2 log2(0.6) =
0.292481 - 0.045600 - 0.147393 = 0.0995 bits.
beta scales the KL's price: bigger beta, less
drift allowed.

## E14

Without the bonus: proxy = q. r_A = 0.9, r_B =
0.7. A wins. The hack needed the exploit term.

## E15

Margins 0.9, -0.1, 0.5, 0.1 -> sigmoids 0.7109,
0.4750, 0.6225, 0.5250. Mean = 2.3334/4 =
0.5834. Verified.

## E16

Title-level: C06 (W12L52), C07 (W11L50), C08
(W12L53), C09 (W11L51 adjacent), plus the RL
ramp W11L47-L49 behind C07: 5 concepts.
Authored bridges: C01, C02, C03, C04, C05,
C10, C11: 6 with no title-level source...
count check: C01-C04 (4) + C05 (1) + C10-C11
(2) = 7? The audit table lists C05 as
"authored bridge (adjacent: none)": so
bridges = C01, C02, C03, C04, C05, C10, C11
= 7, title-level = C06, C07, C08, C09 = 4,
audit = 1. Total 4 + 7 + 1 = 12. Correction
to the lesson's shell-1 line ("5 of 12
title-level, 6 bridges"): the correct split
is 4 title-level (C06-C09), 7 authored
bridges, 1 audit. The W11L47-L49 ramp
supports C07 but is not itself a U10
concept row.

## E17

Refutation: W12L52 is "Reward-Modelling"
(title-level), reward hacking (C10) has no
title. A title about the model is not a
title about its failure mode. The audit
table shows C10 as authored bridge.

## E18

z/0 raises ZeroDivisionError (or inf/nan in
numpy). Fix: branch on T == 0 -> argmax.

## E19

Wrong: 12 8 64 512 2 = 6291456 (6.00 MiB),
missing K vs V. Right: double it =
12582912 (12.00 MiB).

## E20

(a) Fixed dataset: DPO. Offline, one loss, no
rollouts, the data already covers the space.
(b) Live feedback: PPO. Online exploration
uses the feedback stream, DPO cannot explore
beyond its pairs.

## L01

T, top-p defined, three distributions and
entropies computed, KV 12.00 MiB and the
512 ratio, z/0 -> argmax branch, greedy
reproducible but loopy vs sampling diverse
but knob-dependent, (T,p) surface experiment
with quality vs diversity split.

## L02

Affine: s = (max-min)/255, zp =
round(-min/s), toy 0.013725/87, err <= s/2
(0.005882 <= 0.006863), roofline 142.9 ->
571.4 tok/s authored, wide-range calibration
6.7x worse error, INT8 simple vs INT4 4x
smaller but grouped, bits-vs-perplexity
study with the near-free-8 prediction.

## L03

SFT 0.733003, BT 0.3412, PPO 0.6/-0.32,
DPO 0.6588, KL 0.0995 bits. DPO assumptions:
BT true, pi_ref good, pairs cover the space,
watch absolute log-ratios, not just margin.
KL prices drift. Drifting policy: check beta
> 0 and the KL trace. PPO online/explores
vs DPO offline/cheap. Beta sweep with the
win-rate vs KL knee.

## L04

Proxy = q + exploit, toy hack computed,
held-out BT eval 0.5834, self-graded eval
(the training reward as judge) is circular,
"teaches hacking" refuted by the audit
(C10 has no title), the audit rerun rule.

## L05

Pretrain (AR, perplexity) -> SFT (demos,
CE, fails on bad demos, check demo quality)
-> reward model (pairs, BT, fails on noisy
labels, check label agreement) -> PPO/DPO
(clipped/margin, fails on drift, check KL
trace) -> eval (held-out win-rate, fails on
self-grading, check held-out) -> serve
(quantize, cache, fails on OOM, check the
byte budget).

## Implementation and debug task

Reference implementations as in
compute_run5b.py. Bug 1: ascending sort
keeps the tail: symptom is incoherent
samples at any p (the head is dropped),
fix: sort descending. Bug 2: max instead
of min: the clip binds the wrong way
(optimism), symptom is unstable training
with large policy jumps, fix: min.

## Changed-constraint scenarios

S1. Levers: smaller model (1B INT4 = 0.5
GB weights), distillation (keep quality),
KV offload/CPU (slow). Config: 1B INT4:
0.5 GB + KV (12 layers? state the shape,
e.g. toy cache 12 MiB at n=512 fp16 ->
6 MiB int8). Total well under 2 GB.
Justify: weights dominate, the cache is
second.
S2. Changed: BT loss rises (mislabeled
pairs), reward accuracy falls, both
methods' numbers move. Degrades faster:
PPO explores the noise (rollouts chase
mislabeled rewards), DPO fits the noise
but stays within its pairs. Justify: the
C06 extension predicts accuracy falls
with noise, and exploration amplifies
the damage.

## Research-critique question

Strong answer: the loss measures the
training margin, which can rise while
both log-ratios fall (C08 pathology) and
while quality falls (C10 hacking).
Decide with: held-out BT win-rate +
human eval plotted against the loss
across checkpoints, predict divergence
(the loss keeps falling, quality turns).
Red flags: "lower is better" without a
held-out metric, no absolute log-ratio
trace. Remediation: track log pi_c,
log pi_r, KL, and win-rate, not just
the loss.
