# keys-u01.md: interview answer keys, U01

Date: 2026-10-06. Each answer gives the minimum sufficient
explanation, a strong answer, red flags, a rubric, and
remediation.

## B1

Minimum: path length is the longest sequential dependency
chain. RNN: 64. Attention: 1 per layer.
Strong: adds that total work differs (O(T) vs O(T^2)) and
path length measures parallelism, not work.
Red flags: "attention is faster" with no mention of memory.
Rubric: 2 points for both numbers, 1 for the work-vs-depth
distinction. Remediation: C01 toy.

## B2

Minimum: minimize mean negative log-likelihood of the next
token. Labels come from the text itself (each position's
next token).
Strong: notes every position is one free example, D tokens
give ~D examples.
Red flags: "humans label the data."
Rubric: 1 point objective, 1 point label source.
Remediation: C02.

## B3

Minimum: Q (1, 4, 8, 4), scores (1, 4, 8, 8), merged output
(1, 8, 16).
Strong: derives d = 4 first and checks n divisible by h.
Red flags: missing batch dim or wrong softmax axis.
Rubric: 3 points, one per shape. Remediation: C03.

## B4

Minimum: 6P per token: 2P forward, 4P backward (2P
activation grads, 2P weight grads).
Strong: states the dense-layer assumption and names the
attention term as extra.
Red flags: "6P is exact for all models."
Rubric: 2 points. Remediation: C04.

## B5

Minimum: 4n^2 attention (Q,K,V,O) + 8n^2 MLP (n->4n,
4n->n) = 12n^2.
Strong: notes d_ff = 4n assumption and variant caveats.
Red flags: forgetting embeddings or double counting.
Rubric: 2 points. Remediation: C05.

## B6

Minimum: bytes = 2 * T * L * n * 2 (keys+values, fp16).
Value: 2 * 2048 * 32 * 4096 * 2 = 1073741824 = 1.0 GiB.
Strong: explains the two factors of 2 separately.
Red flags: bytes vs bits confusion.
Rubric: 1 formula, 1 value. Remediation: C07.

## B7

Minimum: quadratic term is the T^2 score matrix. Linear
terms: MLP FLOPs and KV cache (T vectors per layer).
Strong: gives byte formulas for both.
Red flags: "everything is quadratic."
Rubric: 2 points. Remediation: C10.

## B8

Minimum: (1) loss/eval mismatch: the task needs something
loss does not measure, (2) eval contamination or a broken
eval.
Strong: proposes correlating per-checkpoint loss vs task
score.
Red flags: "the model is broken, retrain."
Rubric: 2 points. Remediation: C11.

## Deep ladder 1

L1a. One step: input (B,1,n), read cache (B,h,t,d) x2 per
layer, compute one K/V, attend over t+1 positions, emit
(B,1,n) and logits for one token.
L1b. After 5 tokens: keys 5*8=40 numbers, values 40,
total 80 numbers, 160 bytes fp16.
L1c. Step t reads t cached pairs per layer: traffic =
2*t*n*2 bytes per layer, linear in t.
L1d. Append: K = concat(K_old, k_new). Per step: 2P
FLOPs (weights read once), bytes = 2P*2 + cache traffic.
L1e. Cached wins whenever the cache fits. Recompute wins
only if cache memory is unavailable: then cost is O(T^2)
recompute vs O(T) extra memory.
L1f. FLOP-only view misses memory traffic: the step is
bandwidth-bound, so time tracks bytes (growing in T), not
FLOPs (flat).
L1g. Assumption: batch 1, weights read once per token.
Breaks at large batch (compute-bound) or tiny models
(launch overhead).
L1h. Sweep batch at fixed tokens, plot tokens/s. Control:
same prompt lengths, same hardware, warm cache. Knee
batch is the flip point.
Scoring: 1 point per rung. Red flag: confusing cache
bytes with activation bytes.

## Deep ladder 2

L2a. Per token per layer: MLP 16n^2, projections 8n^2,
scores 4Tn.
L2b. N=8, T=4: 1024, 512, 128. T=64: 1024, 512, 2048.
L2c. 4Tn = 24n^2 gives T = 6n.
L2d. O(1) in T for MLP and projections, O(T) per token
for scores (O(T^2) per sequence).
L2e. N=4096, T=2048: MLP 2.68e8, scores 3.36e7. MLP
dominates 8 to 1.
L2f. Amdahl: scores were ~11 percent of the layer, so 2x
there gives ~5 percent end-to-end.
L2g. SwiGLU uses 3 MLP matrices: MLP term becomes
3*n*d_ff*2 per token, recompute the crossover with the
new constant.
L2h. Profile per-layer MLP vs attention time over a T
sweep. Confounders: memory allocation, kernel launch
overhead at small T, thermal throttling.
Scoring: 1 point per rung. Red flag: "attention always
dominates."

## A1

FLOPs: 6 * 1.3e9 * 3e11 = 2.34e21. Aggregate rate:
512 * 1.5e14 = 7.68e16 FLOP/s. Seconds: 30469. Days:
0.35. Strong answer notes this is a roofline roof, not a
schedule.

## A2

Residual: 2*4*4096*4096*32 = 4294967296 = 4.0 GiB.
Scores: 2*4*32*4096^2 = 4294967296 = 4.0 GiB. Equal at
these settings, the ratio flips with B, h, and T.

## D1

Bug: no causal mask. The query attends to future keys
(positions > t), leaking the future. Fix: mask scores
for positions > t to -inf before softmax, or slice
K[:t+1], V[:t+1]. Invariant: attention weights on
positions after t are exactly 0. Test: feed a sequence
where the future token is distinctive and check the
output does not change when it is perturbed.

## S1

Step i (1-indexed) processes 10+i tokens at 2P FLOPs
each: total 2P * sum_{i=1}^{100}(10+i) = 2P * 6050.
Scales quadratically in output length (triangular sum).

## S2

Decode latency: kernel launch overhead and the sequential
token dependency, design change: graph capture / CUDA
graphs and speculative decoding. Training throughput:
compute roof, design change: larger batch until the
statistical efficiency knee, then parallelism.

## R1

Audit questions: (1) What is the baseline, and is it the
strongest simple method? Invalid: untuned naive baseline.
(2) Are budgets matched (FLOPs, data, tuning)? Invalid:
bigger budget for the new method. (3) Is quality equal?
Invalid: faster but worse loss. (4) What T regime?
Invalid: claim at T=512 generalized to long context.
(5) Seeds and variance? Invalid: one seed, overlapping
error bars.
