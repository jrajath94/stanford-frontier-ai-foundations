# U17 answer key

All numeric claims from `../visuals/compute_u17.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

Optimize backward first: max saving = its 38.5% share = 55 ms of the
143 ms step.

## A1

(a) 16x16 patches, linear projection, 256 tokens. (b) 768 tokens,
33.3% image. (c) 30*10 = 300 frames, at 256 tokens each = 76,800
tokens: needs temporal compression, raw patches are infeasible.

## A2

(a) One causal stream with boundary tokens. (b) 756 tokens.
(c) Order: audio tokens, text, image patches, text, boundary
tokens per modality.

## A3

(a) Model: tokens/sec/GPU. Data: GB/sec. Interconnect: collective
bytes/step. (b) 0.010 GB/s. (c) The interconnect/all-reduce pipe:
compute its bytes per step first.

## A4

(a) Different optimal setups, sync on weight broadcast. (b)
Broadcast every 10 updates, KL < 0.01. (c) 10:1: the broadcast
moves 10x the weights per cycle, pipeline it or broadcast less
often within the staleness budget.

## A5

(a) util = min/max, backlog = (r-c)*t. (b) 83.3%, 1200/hr.
(c) Likely not: 30 min of queue aging blows most staleness
budgets, balance rates or bound the queue.

## A6

(a) One pinned template version per stage, hash-checked. (b) v2
ab12 vs v3 cd34: mismatch fires. (c) Three stages, two versions:
map each stage to its version, re-run any stage whose version is
wrong.

## A7

(a) Rollout filters, reward monitoring, human sampling, kill
switch. (b) 10k reviews/day, kill at -2 safety points. (c)
Postmortem both: find which control failed to fire, then tighten
it, two fires suggest the threshold or the filter is wrong.

## A8

(a) id_n = sha256(id_{n-1} + stage_n). (b) Six stages ending
ea98188a5dad105d. (c) Recompute the mix stage's inputs, the break
means mix-v7's recorded inputs differ from what ran.

## A9

(a) Full state, interval trades write vs redo cost. (b) 8.4 s per
checkpoint, 500 steps expected waste. (c) Failures every 6h =
10,800 steps at 2 s: interval ~1000 steps keeps waste under 10%
of the MTBF.

## A10

(a) Lineage, template, manifest-reproducibility, eval direction.
(b) 5/6 pass, template fails: reject the run. (c) Add the stage's
invariants to the suite and version them together.

## A11

(a) Gain <= the stage's share. (b) 143 ms, 38.5%, 116 ms, 19.2%.
(c) Overlap first: it attacks the 40% without new hardware.
measure, then decide on links.

## A12

(a) Toy results are hypotheses, not conclusions about large
scale. (b) Sign flip at 64x. (c) Almost nothing about 70B: the
toys validate methods and arithmetic, not scale behavior, every
claim needs staged evidence.
