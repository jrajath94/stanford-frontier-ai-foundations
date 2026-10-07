# Transfer sets key

## Set 1

(a) N = 1e23/(6*3e11) = 5.6e10, D/N = 5.4. (b) The isoFLOP
minimum is unreachable, the constrained optimum sits at the data
boundary. (c) A data-scaling run: does loss keep falling past
the cap with repeats?

## Set 2

(a) 131072*64*8192 = 68.7 GB. +14 GB weights = 82.7 GB > 80.
(b) 1. Fewer KV heads (GQA-4): halves cache, no quality change
expected. 2. KV quantization: small quality risk, needs eval.
3. Smaller batch: throughput cost. (c) Fixes 1-2 touch quality.
check per-slice evals (U12-C10).

## Set 3

(a) SE = sqrt(0.92*0.08/25) = 0.054. CI [0.81, 1.03] -> [0.81,
1.0]. (b) No: the interval spans 19 points. (c) Grow the slice
to n=200 (half-width 0.038) plus a red-team probe set.

## Set 4

(a) 64e9/1e8 = 640 bits/item, k = 640*ln2 = 444. (b) FPR =
0.5^444 ~ 0: effectively exact at this budget. (c) 1e8 hashes at
~32 bytes = 3.2 GB plus overhead: feasible, the Bloom is
unnecessary here.

## Set 5

(a) Loss on assistant tokens incl. call JSON. 0 on tool results
(observations, not behavior). (b) The model reads PII it must
not learn to emit, worse, the mask-0 text still sits in context.
(c) Redact tool outputs before training (U13-C07), never train
on raw tool text.

## Set 6

(a) Advantages ~0 on 90% of groups: gradients vanish, training
stalls. (b) Mean |advantage| near 0, reward variance near 0 per
group. (c) Filter prompts to the learnable band (10%-90%
success), resample the trivial ones out.

## Set 7

(a) Net +100/min: 10k/100 = 100 min. (b) Queue time grows.
staleness KL blows past budget. (c) Shed oldest first past the
bound, alert, the shed policy is version-aware (drop stale
versions first).

## Set 8

(a) 12*256+2000 = 5072 > 4096: no. (b) 1. Fewer images (drop
least informative). 2. Perceiver-style compression (needs
training). 3. Lower resolution (quality cost on detail). (c)
Test per-task accuracy vs image count on a held-out multimodal
set, usually (1) wins when images are redundant.

## Set 9

(a) G1: slides inspected partial (pp. 1-40), recording missing.
(b) Slide content with page anchors, definitions and figures as
shown. (c) Spoken explanations, demos, Q&A stay unknown.

## Set 10

(a) N = sqrt(0.05*6e21/6) = 7.1e9, D = 1.4e11, D/N = 20. (b)
HYP: decode 2.4 s/req at 300 tokens, 1.7 rps/GPU: 40/1.7*1.3 =
32 GPUs (label HYP). (c) 1. SFT eval gate: instruction up, base
not regressed. 2. Data manifest + lineage complete. 3. Safety
probe suite passes.
