# keys-transfer.md: transfer set answers

Date: 2026-10-06. Full keys for `transfer-sets.md`.
Keep separate.

## T1

Q1. Scores: (1e6)^2 * 2 bytes = 2e12 bytes = 2 TB
per head fp16 (4 TB fp32). SSM state: d^2 =
16,384 floats = 64 KiB fp32.
Q2. Exact recall: passkey/exact-quote retrieval
(U07 C10). The blur misses sharp lookups.
Q3. Hybrid: SSM layers plus a few softmax layers
(or retrieval, U10). Cost: the softmax layers
pay T^2 at their positions. Keep them few and
late.
Q4. Only via the exact mechanism (softmax
layers or retrieved quotes). The SSM alone
cannot promise verbatim quotes.

## T2

Q1. Queueing (burst exceeds service rate) and
expert imbalance (burst skews the gate).
Q2. Continuous batching removes formation wait,
not queue wait: arrivals still exceed capacity,
so the queue builds.
Q3. Drops spike at the 2 experts. Step time is
set by the fullest expert (up to 4x the
balanced cost in the toy).
Q4. Queue: bounded admission with fast reject
plus autoscale. Gate: raise the balance-loss
weight (watch the specialization tradeoff).

## T3

Q1. Data parallel (gradient sync) or tensor
parallel cross the link. Ring: 2*15/16*20/50 =
0.75 s per sync (hypothetical BW).
Q2. Lost: work since the last checkpoint (up to
30 min). Resumes: model, optimizer, and
data-loader state from the checkpoint.
Q3. Keep TP inside each region. Use DP across
regions (one sync per step, not per matmul).
Q4. Spot wins when discount > expected rework
fraction (U09 C09): here rework ~ (30+5)min
per preemption vs the run length.

## T4

Q1. Per-channel (or per-group): expert weight
ranges skew like activations. Per-tensor dies
on the quiet experts.
Q2. Gate precision: int8 logits blur the top-k
boundary. Routing mistakes rise. Keep the gate
in higher precision.
Q3. 8 * 2 GB = 16 GB fp16 -> 8 GB int8. Saves
8 GB.
Q4. Gate: task delta >= -0.5% (U10 C11). First
experiment: per-channel int8 vs fp16 on the
eval set, 3 seeds.

## T5

Q1. Threshold artifact (U06 C03): a step metric
on a smooth skill ramp. Test: replot with the
continuous score.
Q2. The continuous scores and the eval protocol
(seeds, prompts, metric definition).
Q3. Verdict: no emergence. The jump is a
measurement artifact. The skill was always a
ramp.
Q4. Log-scale x (model scale), y with both the
0/1 metric (steps) and the continuous score
(ramp) on twin axes.

## T6

Q1. Healthy: (4-1)/(16+4-1) = 3/19 = 15.8%. The
slow stage stretches every microbatch: idle
grows beyond the formula.
Q2. Both: a straggler inside a pipeline. The
bubble formula assumed equal stages.
Q3. Schedule: rebalance stages (move layers off
the slow node). Placement: replace or isolate
the node.
Q4. Stage 1 gates the fill: every microbatch
waits for it first, so its slowness serializes
the whole pipeline start.

## T7

Q1. Build 7200 s over 10k queries = 0.72 s per
query amortized, plus 5 ms query = 725 ms >>
50 ms brute force. The index does not pay off
at this volume.
Q2. Incremental updates (not full rebuilds). 
smaller index (fewer docs or PQ).
Q3. 405 ms / 4000 ms = 10.1% of the SLO.
Q4. Brute-force or lightweight ANN per query
(no heavy index), cache hot queries, rerank
only the top 10.

## T8

Q1. 16 bytes * 70e9 / 8 = 140 GB per GPU >
40 GB. Does not fit: needs more GPUs or
offload/CPU states.
Q2. LoRA: 16 * 8192 = 131,072 vs 16,777,216:
128x fewer.
Q3. Risk: rank 16 cannot carry a large manifold
move (U06 C07). Detect: loss stalls while full
fine-tuning (or higher rank) moves.
Q4. Zero after merging: W = W0 + BA folds into
the weights at load.

## T9

Q1. Verification accepts a variable number of
tokens per request: the batch's sequences grow
unevenly, complicating the iteration schedule.
Q2. ~3x fewer target passes per token: the
target does one verify pass per ~3 accepted
tokens (minus draft cost).
Q3. The prefill stalls the verify (U08 C03):
chunk it or queue it behind the verify.
Q4. When acceptance is low (draft too weak) or
batches are huge (the target is already
saturated): speculation adds draft cost for no
gain.

## T10

Q1. No. the rubric grades falsifiability: a
clean falsification with evidence scores on
question, hypothesis, and failure criteria.
Q2. The control: the identity held to 1e-14
(H1 replicated), so the implementation is
sound. Only the timing hypothesis failed.
Q3. It rules out "FFT conv-view always wins"
for per-call kernel builds: valuable to anyone
porting the view naively.
Q4. "We replicated the SSM triple identity to
7e-15 on random SSMs, then falsified the claim
that the FFT convolution view beats the
sequential loop: kernel materialization
dominates per-call (127 ms vs 67 ms at
T=16384). With precomputed kernels the FFT
path wins from T=256. Materialize once, or pay
per call."
