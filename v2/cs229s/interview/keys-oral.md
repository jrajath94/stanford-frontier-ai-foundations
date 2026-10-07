# keys-oral.md: oral defense answers

Date: 2026-10-06. Full keys for `oral-defenses.md`.
Keep separate.

## O1

a. Attention materializes a T x T score matrix:
O(T^2) memory and traffic dominate.
b. 17.2 TFLOPs scores, 256 MiB fp32 per head.
c. Arithmetic intensity: 2 T^2 d FLOPs over T^2
bytes ~ O(d): below the ridge for large T, so
bandwidth binds.
d. Running max and normalizer per block. O(T d)
memory, never the T^2 matrix.
e. Exactness wins when sharp selection decides
quality (copy, retrieval). Approximation when
the task is smooth.
f. Tiling overhead dominates at small T: the
naive kernel fits in SRAM anyway.
g. Softmax decomposes over blocks with the
running max correction: exact only with the
rescaling algebra.
h. Time naive vs tiled at T = 1k..32k on the
target GPU. Expect 4x scaling for naive and a
crossover where tiling wins.

## O2

a. Loss as a power law of scale: straight line
in log-log space.
b. B = -0.0748/decade, L(1e11) = 1.70.
c. Slope = (log L2 - log L1)/(log N2 - log N1). 
two points fix the line.
d. Fit_law/predict as in the lab. Each point
costs a full training run.
e. Floor form when large-N points bend flat. 
two-point line with exactly two runs.
f. Recipe changed (data mix), or seed luck on
the single big run.
g. Double data quality at fixed N: loss drops,
the N-only law says impossible.
h. Demand continuous scores. Replot vs scale. 
the jump should flatten to a ramp.

## O3

a. The batch re-forms every decode iteration.
b. 1.0 s vs 1.5 s mean latency.
c. Last completion ties at 4.0 s at equal token
rate: only formation wait differs.
d. Arithmetic on arrival times. O(batch) per
iteration to pick the set.
e. Bursty: continuous wins. Simultaneous:
static can win (no wait to remove).
f. Missing chunked prefill: whole prefills
stall decodes.
g. Insertion is free. A huge prefill stalls
every running decode for its duration.
h. Simulate the trace: find GPUs where p99 <=
SLO with bounded admission. Add headroom for
the burst factor.

## O4

a. Score experts, keep top-k, renormalize,
weight the outputs.
b. Experts 0 and 2. Weights 0.646, 0.354.
c. Capacity = 64*2/8*1.25 = 20. Expert 0 drops
20 of 40.
d. Capacity() as in the lab. Buffer memory
scales with the factor.
e. Token-choice drops overflow (needs balance
loss). Expert-choice has zero drops by
construction but may starve unpopular tokens.
f. Heavy drops at that expert: biased training
subset. Check per-expert drop rates.
g. Over-weighted: uniform routing kills
specialization. Under-weighted: collapse.
h. K=2, factor ~1.25, drop-with-residual for
throughput (or no-drop for quality). Verify
against the p99 SLO.

## O5

a. Recurrence (streaming), convolution view
(parallel), scan (tree).
b. States [1.0, 0.9, 1.81, 1.629]. K = [1, 0.9,
0.81, 0.729]. Tree total 1.9333 = loop final.
c. Induction on y_t = sum_{k<=t} C A^k B
x_{t-k}: the taps are the impulse response.
d. (a2,b2)o(a1,b1) = (a2a1, a2b1+b2). O(n)
work, O(log n) depth.
e. Conv view needs fixed A. Scan needs only
associativity.
f. Associativity failed: tree order matters.
g. Per-call kernel materialization dominates:
127 ms FFT vs 67 ms loop at T=16384 (capstone
H2a).
h. Time loop/conv/FFT/scan across T with and
without precomputed K. Report both
crossovers.

## O6

a. Stages: layer groups. Microbatches: batch
splits. Bubble: idle stage time.
b. 3/11 = 27.3%.
c. Fill p-1 steps + m useful + drain p-1 =
m+p-1. Bubble = (p-1)/(m+p-1).
d. Bubble() as in the lab. 1F1B keeps the
bubble with less activation memory.
e. More microbatches when memory allows. 
interleaving when it does not.
f. Uneven stages (e.g. embeddings at 2x).
g. Equal stage times. A heavy stage makes the
formula understate idle time.
h. Profile per-layer cost, balance stages by
time not layer count, verify with measured
idle vs the formula.

## O7

a. ZeRO-1: optimizer states. -2: +gradients. 
-3: +params.
b. ZeRO-3: 20.0 GB/GPU. DP: 160 GB/GPU.
c. Fp16 params 2B + fp16 grads 2B + fp32 master
copy 4B + m 4B + v 4B = 16 bytes.
d. Per_gpu = 16*P/n. Stage 3 adds per-layer
param all-gather.
e. ZeRO-3: 8x less memory, more comm. DP:
simpler, OOMs.
f. Gather latency dominates for tiny layers.
g. Even shards. Hurts with uneven layer sizes
or heterogeneous GPUs.
h. ZeRO-3/FSDP: 14*70e9/n per GPU. Pick n so
per-GPU < 80% of HBM. Keep TP intra-node.

## O8

a. S = (max-min)/(2^b-1), z = round(-min/s). 
x_q = clamp(round(x/s)+z).
b. S = 0.1333, z = 8, x_q = 12, x_hat =
0.5333.
c. Nearest mark is at most s/2 away inside
[min, max].
d. Quantize/dequantize as in the lab. O(N)
one-time.
e. Per-tensor crushes the quiet channel. 
per-channel gives each its own ruler (218x
on the quiet channel in the U05 toy).
f. Outlier-set range. First fix: per-channel
(or clip the range).
g. Range covers the data. One weight at 100
with the rest in [-1,1] destroys the small
values.
h. 3-seed eval: task delta >= -0.5%, latency
win confirmed, then ship.

## O9

a. ANN -> filter -> cross-encoder rerank.
b. 5 + 400 = 405 ms.
c. 400/405 of the cost sits in the reranker.
d. Cost = sum count*cost. Narrowing risks the
gold doc (ANN recall miss).
e. Cascade: better quality at 405 ms. ANN-only:
5 ms with a lower ceiling.
f. The ANN's top-k missed it. The funnel
failed, not the reranker.
g. The funnel keeps the gold. ANN recall 0.95
means 1 in 20 golds never reach the reranker.
h. Funnel 100->20->5, quality gate -0.5%,
p99 budget with chunked prefills, rollback to
keyword search on red.

## O10

a. Mean time between failures for the fleet:
per-GPU MTBF / n.
b. 26,280/256 = 102.7 h = 4.28 days.
c. Any-of-n fails the run (series system). 
assumes independent failures.
d. Overhead = (N/k)*c: 300 s (3%) at k=100.
e. Checkpoint-restart (standard) vs spares
(idle cost, instant resume).
f. Independence broke: correlated failures
(rack, power, network).
g. Independent failures. A rack event takes 32
GPUs at once and the math understates it.
h. Frequent sharded checkpoints, elastic
restart, correlated-failure drills, on-call
rotation: at 6.4 h MTBF, recovery is routine
operations, not exceptional.
