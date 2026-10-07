# U08 interview key

## B1-B6

B1. Mean of shard grads = grad of the concatenated batch (algebraic
identity for mean losses).
B2. Large messages: ring is bandwidth-optimal and N-independent.
Small messages: tree wins on latency.
B3. Stage 1: optimizer states. Stage 2: +grads. Stage 3: +params.
B4. Multiply the loss by S so fp16 grads stay representable,
unscale before stepping. Unnecessary with bf16 (wider range).
B5. All-gather params, compute the layer, release, reduce-scatter
grads in backward.
B6. Checkpoint every k steps, restart from the latest, lost work <=
k steps, record N in the metadata.

## L1

L1.1 Params 14, grads 14, Adam states 56 GB.
L1.2 84.0, 35.0, 22.8, 10.5 GB/GPU.
L1.3 Two all-gathers + one reduce-scatter per layer (~50 percent
more comm than DP).
L1.4 The formula, check stage 3 = total/N and monotonic decrease.
L1.5 Tensor parallelism splits the layer itself (for layers that
exceed one GPU). Debug: wrap coarsely. Critique: uneven layers
unbalance shards. Experiment: the 13B table.

## L2

L2.1 max(0, comm - compute).
L2.2 96 ms sequential, 36 ms overlapped.
L2.3 Size buckets so all-reduce time <= layer compute time.
L2.4 The two-timeline simulator, check overlapped <= sequential.
L2.5 Faster links help only the exposed part. Debug: stream
dependencies. Critique: async needs correct sync. Experiment: sweep
comm/compute.

## E1

(a) 26+26+104 = 156 GB. (b) ZeRO-1: 65.0, ZeRO-2: 42.25, ZeRO-3:
19.5 GB/GPU. (c) All fit in 80 GB. Catch: stage 3 adds per-layer
all-gathers, tiny layers go latency-bound. Rubric: (a) 1 pt, (b) 2
pts, (c) 1 pt. Red flag: "stage 3 always".

## E2

(a) 30 percent. (b) Step 8.5 s, 1.18x. (c) Compress/reduce the
bytes, grow the compute per step, faster links. Red flag: "more
buckets".

## D1

Bug: the summed grads are applied without dividing by N, so the
update is N times too large. At the tuned lr this diverges, the
"1/N batch" story is the misdiagnosis. Fix: apply_update(total /
len(shard_grads)). The check that catches it: invariant 1 (DP update
must equal the big-batch update). Rubric: find 2 pts, fix 1 pt,
name the catching invariant 1 pt.

## S1

Reshard on every change: concatenate shards and re-split for the new
N, re-seed the data loader, and verify the invariant checks before
resuming. Automate it, never hand-edit.

## S2

Every collective waits for the slowest rank: the step stretches ~40
percent. Responses: (1) find and fix the thermal issue (cooling,
power cap, remap), (2) elastic reconfiguration to drop/replace the
rank. Do not "tune around it".

## R1

Gaps: (1) no efficiency: tokens/s alone hides the overhead, need
T1/(N*TN). (2) no per-rank timings: stragglers invisible, need them.
(3) no baseline: "linear" relative to what, need the single-GPU
reference and the setup.
