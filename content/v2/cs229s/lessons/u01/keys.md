# keys.md, U01 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Path length is the longest chain of sequential dependencies
from input to output. RNN for `T = 32`: 32 steps. Attention:
1 hop per layer.

## E02

Scores: `T^2 = 512^2 = 262144` per head. fp16 bytes:
`262144 x 2 = 524288` bytes, 0.5 MiB per head.

## E03

Diagnosis: the `T^2` score matrix exceeded device memory.
The dominating term is `2 B h T^2` bytes. Fixes: tile the
attention (FlashAttention), shorten `T`, or shard across
devices. The RNN baseline runs because its state is `O(1)`
in `T`.

## E04

The RNN trains faster in wall-clock on one serial core only
if the transformer cannot parallelize. With one core, the
transformer still does `O(T^2)` work versus the RNN's
`O(T)`. The RNN wins. Parallelism, not total work, is the
transformer's edge.

## E05

Fairness control: equal parameter count, equal training
tokens, equal optimizer budget, same hardware. Report `T*`
where wall-clock to target loss crosses, with three seeds.

## E06

Minimize mean negative log-likelihood of the next token.
The labels come from the text itself: each position's label
is the following token.

## E07

Uniform loss: `log(32000) = 10.37` nats. Perplexity 32000.
Loss 2.0 nats: perplexity `e^2 = 7.39`.

## E08

Failure: catastrophic forgetting (overfitting to the small
task set). Fixes: lower the learning rate, early-stop on a
held-out general set, or mix pretraining data into the
fine-tuning batches.

## E09

Pretraining still helps as an initialization prior, but its
edge shrinks. For: it encodes language structure the task
data may not cover. Against: infinite task data with a good
optimizer reaches the same point, the pretraining cost is
then pure overhead.

## E10

Hypothesis: full fine-tuning beats a linear probe by a
margin that grows with task distance from the corpus.
Distance proxy: mean token-level perplexity of the base
model on the task data. Measure the accuracy gap versus
the proxy.

## E11

`n = 12`, `h = 3`, so `d = 4`. `Q`: `(1, 3, 5, 4)` at batch
1. `A`: `(1, 3, 5, 5)`. Merged output: `(1, 5, 12)`.

## E12

Numbers: `2 x 8 x 1024^2 = 16777216`. fp16 bytes: 33554432,
32 MiB.

## E13

Check: (1) softmax over the wrong axis (rows do not sum to
1), (2) division by zero in a mask (all keys masked),
(3) `d` mismatch after the head split (silent broadcast).

## E14

`10` is not divisible by `3`. The head split is undefined.
Minimal fix: change `n` to 9 or 12, or change `h` to 2 or 5.

## E15

Hypothesis: accuracy on a copying task falls when `d`
drops below the token separation need. Metric: minimum
pairwise key distance versus `d`. Measure accuracy versus
`d` at fixed `n`.

## E16

Training per token is `6P`: `2P` forward, `4P` backward.
Backward splits into `2P` for activation gradients and
`2P` for weight gradients.

## E17

FLOPs: `6 x 1.3e9 x 3e11 = 2.34e21`. Device-seconds at
`1e3 x 2e14 = 2e17` FLOP/s: `11700` s, about 3.25 hours.
(Sustained rates are lower in practice, this is the roof.)

## E18

Causes: (1) activation checkpointing is on, so backward
recomputes forward (ratio toward 3x), (2) the backward
pass is memory-bound while forward is compute-bound, so
the FLOP ratio does not map to time.

## E19

The attention term `12 L n T` per token grows. The `6P`
rule covers dense parameters, the attention term is added
separately and doubles with it.

## E20

Experiment: fix model and data, toggle checkpointing,
measure forward/backward time ratio. Control: identical
batch, sequence, and hardware, only the checkpoint policy
changes.

## E21

Per layer `12n^2`: attention room `4n^2` (Q, K, V, O),
MLP room `8n^2` (up and down at `d_ff = 4n`).

## E22

Per layer: `12 x 2048^2 = 50331648`. Times 24:
`1207959552`. Embeddings: `32000 x 2048 = 65536000`.
Total about `1.27B`. fp16 bytes: about 2.55 GB.

## E23

Variant causes: (1) SwiGLU MLP with three matrices and a
larger `d_ff`, (2) untied output embeddings adding a
second `V n` term, (3) MoE layers or a different `d_ff`
multiple.

## E24

Tied embeddings: the output head reuses the embedding
matrix. Total drops by one `V n` term (here `131M`
parameters at the 6.57B scale).

## E25

Hypothesis: the formula predicts public card counts
within 1 percent once the MLP variant and tying choice
are identified from the card. Tolerance: 1 percent.

## E26

Residual-stream bytes: `2 x B x T x n x L` in fp16
(bytes-per-number times batch, tokens, width, layers).

## E27

Bytes: `2 x 4 x 4096 x 4096 x 32 = 4294967296`, exactly
4.0 GiB for the residual stream alone.

## E28

Suspect: optimizer states. Adam keeps fp32 master
weights plus two fp32 moments per parameter, about
16 bytes per parameter total. A tiny model with a large
optimizer footprint OOMs at step 0 before any batch runs.

## E29

Checkpoint every other layer: activation bytes roughly
halve (only stored layers kept). FLOPs rise: the
backward pass recomputes the dropped layers, pushing the
backward/forward ratio from 2x toward 3x.

## E30

Hypothesis: peak allocated memory grows linearly in `T`
with tiled attention (slope from the linear terms) and
quadratically with naive attention. Test: fit
`memory = aT + b` versus `memory = aT^2 + bT + c` and
compare residuals.

## E31

After `t` steps, keys and values each have shape
`(B, h, t, d)`, per layer.

## E32

Bytes: `2 x 4096 x 32 x 4096 x 2 = 2147483648`, exactly
2.0 GiB.

## E33

The KV-cache read per step grows with `T`. Per-token
time is linear in prompt length because each step reads
the full cache. The term is the `O(T)` cache traffic per
layer.

## E34

No cache: step `t` recomputes `t` tokens. Total tokens
processed: sum over 100 steps of roughly `10 + t`,
about `10 x 100 + 100 x 101 / 2 = 6050` token-equivalents
times `2P` FLOPs each. (Prompt 10, then steps see 11, 12,
..., 110 tokens.)

## E35

Hypothesis: ms per token grows linearly in `T` with
slope equal to cache bytes per token divided by measured
memory bandwidth. Measure the slope, divide by the
bandwidth spec, expect a ratio near 1.

## E36

Training: parallel over `B x T`, `6P` FLOPs/token,
compute-bound, activations `O(BTL)`. Inference decode:
parallel over `B`, `2P` FLOPs/token, bandwidth-bound,
KV cache `O(BTL)` with a smaller constant.

## E37

Bytes per token: `13e9 x 2 = 26e9`. Floor:
`26e9 / 3e12 = 0.00867` s, about 115 tokens/s max.

## E38

New bottleneck: the batch made decode compute-bound, or
the KV cache traffic saturates a different limit (cache
capacity forcing small batches, kernel launch overhead
at small per-request work). Profile per-request time.

## E39

With infinite bandwidth, decode is limited by compute
latency: kernel launch overhead and the sequential
dependency of one token per step. Latency floor, not
throughput.

## E40

Sweep batch size at fixed total tokens, plot tokens/s.
Hypothesis: a knee batch `B*` where the curve bends from
bandwidth-bound (rising) to compute-bound (flat).
Measurement: tokens/s and per-token latency at each
batch, report `B*`.

## E41

MLP per token `16n^2`, attention projections `8n^2`,
scores `4Tn`. Set `4Tn = 24n^2`: `T = 6n`.

## E42

`n = 2048`: crossover `T = 12288`. At `T = 4096`: MLP
`16 x 2048^2 = 6.7e7`, scores `4 x 4096 x 2048 =
3.36e7`. MLP dominates by 2 to 1 (plus projections).

## E43

Attention is a small slice at this `T` (C09 numbers: at
`T = 2048`, `n = 4096`, scores are ~1/9 of the layer).
Amdahl's law: 2x on 11 percent of the work gives about
5-6 percent end-to-end.

## E44

`d_ff = 8n`: MLP per token `32n^2`. Crossover:
`4Tn = 40n^2`, so `T = 10n`.

## E45

Hypothesis: the FLOP crossover sits at `6n`, but the
measured time crossover shifts (memory effects favor the
MLP or attention differently). Report both numbers and
the shift direction.

## E46

Quadratic: attention scores (`T^2`). Linear: MLP FLOPs
and KV cache (`T`).

## E47

Bytes: `2 x 1 x 32 x 32768^2 = 68719476736`, 64 GiB.
Naive attention cannot fit this.

## E48

Blame the `T^2` score term first: doubling `T`
quadruples it. Check score bytes before anything else.

## E49

The KV cache still grows linearly with a large constant
(`4BLn` bytes), and per-step cache reads grow with `T`.
FLOPs are tamed, memory traffic is not.

## E50

Hypothesis: max batch `B_max(T) = C / T` in the
cache-bound regime. Fit `B_max` versus `1/T`, report
the constant `C` (cache bytes budget) and the `R^2`.

## E51

Perplexity is the exponential of mean negative
log-likelihood: the model's effective vocabulary
uncertainty per token.

## E52

Perplexities: `e^2.0 = 7.39`, `e^2.3 = 9.97`. Ratio:
`e^0.3 = 1.35`.

## E53

Problem: test contamination (eval text in training).
The score measures memorization, not capability. Fix:
deduplicate eval from training and re-evaluate.

## E54

Trust the quiz for the deployment call: it measures the
task. Use the loss only for training health. If they
must pick one number to ship on, it is task accuracy.

## E55

Hypothesis: perplexity-task correlation is strong on
knowledge tasks and breaks on multi-step reasoning
tasks. Split tasks by reasoning depth, report
per-task correlations.

## E56

Baseline: the reference method run under identical
conditions. Ablation: your method with one part removed.

## E57

Versus naive: `120/40 = 3x`. Versus strong:
`35/40 = 0.875x` (a slowdown).

## E58

Ask for: the baseline's budget (compute, data, tuning
effort), the metric definition, and whether the
baseline is the strongest known simple method.

## E59

Honest claim: report your number with the hardware and
budget named, and mark the comparison as
against-published-numbers, not a controlled experiment.
Do not claim a speedup.

## E60

Hypothesis: re-running with a strong baseline erases at
least half the claimed gain. Audit list: baseline
identity, budget match, hyperparameter tuning effort,
metric definition, seed count.
