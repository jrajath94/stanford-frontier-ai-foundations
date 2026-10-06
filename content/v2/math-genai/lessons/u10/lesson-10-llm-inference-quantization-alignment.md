# Lesson 10, LLM inference, quantization, and alignment

Unit: math-genai-U10. Leaf concepts: math-genai-U10-C01 to C12.
Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

This lesson is locally authored bridge content for prerequisite
modules P12 (tensors), P14 (transformer mechanics), P15
(hardware), and P17 (reinforcement learning). It does not
claim to reproduce the instructor's lectures. Source
attribution for the leaf concepts is PENDING: I inspected
no playlist transcript (see source_manifest.md SRC-04,
source_gaps.md G2). The playlist covers an overview of
reinforcement learning in W11L47, the policy gradient
theorem in W11L48, expressing an AR-LM as an RL policy in
W11L49, PPO in W11L50, TRPO in W11L51, reward modelling
in W12L52, DPO in W12L53, and state-space models in
W12L54, all by title only. A separate source-block lesson
(lesson-10b) follows those eight titles at the title
boundary. All numbers below are computed 2026-10-06,
numpy 1.26.4, float64, seed 0 where RNG is used
(compute_run5b.py reproduces every one). Log base 2 in
bits for distributions, natural log inside the RL
algebra. Hardware numbers in C04 are authored arithmetic
(labeled as such, per the E-014 precedent), not measured
benchmarks.

## Scope and objectives

Scope: temperature and top-p decoding, the KV cache,
quantization, hardware tradeoffs, SFT, preference and
reward modelling, PPO, DPO, KL regularization, reward
hacking, evaluation, and the repository/lecture coverage
audit.

Objectives: after this lesson the learner can compute
decoding distributions at any temperature, do the KV
cache byte arithmetic, quantize a matrix by hand,
price a token with the roofline model, write the SFT
loss, the Bradley-Terry loss, the PPO clipped
objective, and the DPO loss, explain what the KL term
does, demonstrate reward hacking on a toy, run a
BT-based eval, and audit what the sources do and do
not cover.

Dependencies: U09 (AR factorization, the transformer,
sampling, context budgets), R27 (tensor shapes, the
P12 bridge), R30 (hardware basics, the P15 bridge),
R31 (RL basics, the P17 bridge).

## How to read this lesson

Mechanisms A to D, each with shells 0 to 10, then one
section per leaf concept with the full contract. Figures
carry one claim each. The audit table lives in
visual_audit.md.

The running toys. Logits z = [2.0, 1.0, 0.5, 0.1]
(V = 4). KV toy: L = 12 layers, h = 8 heads, d_h =
64, n = 512 tokens, fp16. Quant toy: W = [[0.5,
-1.2], [2.3, 0.1], [-0.4, 1.8], [1.1, -0.9]].
Alignment toys: rewards r_c = 1.2, r_r = 0.3, PPO
cases (ratio 1.3, A = 0.5) and (ratio 0.5, A =
-0.4), DPO beta = 0.1, log-ratios 0.4 and -0.3, KL
toy pi = [0.5, 0.3, 0.2] vs uniform.

---

## Mechanism A, from logits to tokens, fast

Shell 0. The question: training is parallel, but
generation emits one token at a time. How do you
serve that without recomputing everything, and how
do the decoding knobs change the output? What
would change if every step recomputed the full
prefix? The observable result: latency grows
quadratically instead of linearly per token.

Shell 1. The toy: T = 0.5/1.0/2.0 gives pmax
0.8282/0.5745/0.4056 and entropy 0.8754/1.6174/
1.9017 bits. KV cache for n = 512: 12582912
bytes = 12.00 MiB, per new token 24576 bytes.
Prefill 3221225472 FLOPs vs decode 6291456 per
token: ratio 512.

Shell 2. Objects: temperature T, top-p
threshold, the K and V tensors per layer, the
cache. Units: bytes, FLOPs.

Shell 3. One rule: decode reuses the past.
Attention at step n+1 needs K, V for positions
1..n: cache them instead of recomputing. The
decoding knobs (T, top-p) only reshape the
final distribution, they never touch the cache.
Justified assumption: the model is
autoregressive, so past keys and values never
change.

Shell 4. Derive the algorithm: prefill computes
K, V for the prompt (O(n^2)), each decode step
appends one K, V row and attends over the cache
(O(n) per step). Temperature: softmax(z/T).
Top-p: truncate and renormalize.

Shell 5. Check the invariant: cache bytes =
2 L h d_h n bpe. 2 12 8 64 512 2 =
12582912. Per-token increment: 24576. The
ratio prefill/decode-per-token = n = 512
exactly.

Shell 6. Change ONE factor: T 1.0 -> 0.5.
Predict: sharper, pmax up, entropy down.
Measured: 0.5745 -> 0.8282, 1.6174 ->
0.8754 bits. Controls: same logits.

Shell 7. Counterexample: no cache, full
recompute per step. Step k costs O(k^2), total
O(n^3) for n tokens. The cache is not an
optimization, it is what makes generation
feasible.

Shell 8. Compare: greedy (T -> 0) vs sampling.
Greedy: reproducible, mode-seeking, can loop.
Sampling: diverse, needs the knobs. Equal
logits: different products.

Shell 9. Falsifiable extension: measure
per-token latency vs n with and without the
cache on a real setup. Predict: without cache
the curve is quadratic, with cache linear
(memory-bound).

Shell 10. Production: the (T, top-p) pair is
a product knob, the cache size is a capacity
spec. The stakeholder decision: fix both by
the quality gate and the memory budget, then
version them with the model.

### C01, temperature and top-p

Motivating question: how do two numbers
change what the model says?

Start from zero. Logits z = [2.0, 1.0, 0.5,
0.1]. Temperature divides: softmax(z/T). T
< 1 sharpens (the best token wins more), T
> 1 flattens (the tail matters more), T ->
0 is greedy (argmax). Top-p truncates: sort
descending, keep the smallest set with
cumulative mass >= p, renormalize. The two
compose: temperature first, then top-p.

Mental model: temperature is the adventure
dial, top-p is the quality floor. Turn up
the adventure, keep the floor.

Variables: T > 0, p in (0, 1].

Computed numbers. T = 0.5: [0.8282,
0.1121, 0.0412, 0.0185], entropy 0.8754
bits, argmax 0. T = 1.0: [0.5745, 0.2114,
0.1282, 0.0859], entropy 1.6174. T =
2.0: [0.4056, 0.2460, 0.1916, 0.1569],
entropy 1.9017. Top-p 0.9 at T = 1: keep
[0, 1], renorm [0.7311, 0.2689, 0, 0]
(the lesson's keep-while-cumsum-below-p
convention, as in U09-C10).

Code: softmax(z / T), the top-p block in
compute_run5b.py.

Checks. (1) Probs sum to 1. (2) argmax
unchanged (0) across T. (3) Entropy rises
in T: 0.8754 < 1.6174 < 1.9017.

Costs. O(V) for temperature, O(V log V)
for the top-p sort.

Alternatives. Top-k: fixed count, simpler,
less adaptive. Typical sampling:
information-based truncation. Beam
search: deterministic exploration.
Selection boundary: T + top-p for open
generation, greedy for factuality, beam
for constrained outputs.

Failure case. T = 0 in code as z/0:
division by zero. Implement greedy as
argmax, not as a temperature limit.

Research extension. The (T, p) quality/
diversity surface on a fixed eval
(U09-C10 extension). Falsifiable: the
quality-optimal and diversity-optimal
points differ.

### C02, decode and the KV cache

Motivating question: why is the first
token slow and the rest fast?

Start from zero. Prefill: the prompt of n
tokens goes through the model once, every
layer computes K and V for all n
positions: O(n^2) attention. Those K, V
are cached. Decode step: one new token,
its Q attends over the cached K, V: O(n)
per step. Append its K, V to the cache.
Total for m new tokens: O(n^2 + m n)
instead of O((n+m)^3)-ish without cache.

Mental model: prefill reads the book,
decode writes the sequel one word at a
time, and the cache is the notes taken
while reading.

Variables: L, h, d_h, n, bpe (bytes per
element).

Computed numbers. L = 12, h = 8, d_h =
64, n = 512, fp16 (2 bytes): cache = 2
(K,V) 12 8 64 512 2 = 12582912 bytes =
12.00 MiB. Per new token: 2 12 8 64 2
= 24576 bytes. Prefill attention FLOPs:
2 L h n^2 d_h = 3221225472. Decode per
token: 2 L h n d_h = 6291456. Ratio:
512 = n exactly.

Code: kv = 2 * L * h * dh * n * bpe.

Checks. (1) 12.00 MiB exact. (2) Ratio
= n. (3) Cache grows linearly in n,
prefill quadratically.

Costs. Memory 12 MiB here, time
dominated by prefill for long prompts.

Alternatives. No cache: O(n^3) total,
infeasible. Quantized cache (int8):
halves bytes, adds error. Sliding
window: bounds the cache. Selection
boundary: full cache when quality needs
full context, window when n is huge.

Failure case. Forgetting the factor 2
(K and V): budget 6 MiB, OOM at 2x the
planned n. Count both.

Research extension. Measure the real
prefill/decode latency split vs n.
Falsifiable: the crossover where decode
dominates total time is measurable.

---

## Mechanism B, shrinking the model

Shell 0. The question: the weights are
fp32 or fp16 numbers. What happens if
each weight gets 8 bits, or 4? What
would change if you quantized without
calibrating the range? The observable
result: the range sets the resolution,
a wrong range wastes the bits.

Shell 1. The toy: W 4x2, min -1.2, max
2.3. INT8 affine: scale = 3.5/255 =
0.013725, zero-point 87. Max abs error
0.005882, mean abs error 0.004167.
Bytes: 32 -> 8 (4x). Roofline: 7B fp16
14 GB at 2 TB/s: 7.00 ms/token, 142.9
tok/s ceiling. INT4 3.5 GB: 1.75 ms,
571.4 tok/s. All C04 numbers are
authored arithmetic, not benchmarks.

Shell 2. Objects: scale s, zero-point
zp, the quantized ints, bandwidth,
model bytes. Units: bytes, ms, tok/s.

Shell 3. One rule: affine quantization
maps [wmin, wmax] to [0, 255]:
q = round(w/s + zp), w_hat = (q - zp)
s. The error is at most s/2 per weight.
Decode is memory-bound: time per token
~= model bytes / bandwidth. Justified
assumption: one token needs every weight
once (true for dense decode).

Shell 4. Derive the algorithm: s =
(wmax - wmin)/255, zp = round(-wmin/s).
Dequantize and measure max/mean abs
error. Roofline: tok/s <= bandwidth /
model bytes.

Shell 5. Check the invariant: max err
0.005882 <= s/2 = 0.006863. Bytes drop
4x. The ceiling math: 14e9/2e12 = 7e-3
s.

Shell 6. Change ONE factor: fp16 -> int4
(4x fewer bytes). Predict: 4x the
ceiling. Measured (arithmetic): 142.9
-> 571.4 tok/s, exactly 4x. Controls:
same bandwidth.

Shell 7. Counterexample: quantize with
the range [-10, 10] when the weights
span [-1.2, 2.3]. s = 20/255 = 0.0784,
error up to 0.0392: 6.7x worse. The
calibration range is load-bearing.

Shell 8. Compare: INT8 (1% error-ish,
simple) vs INT4 (4x smaller, needs
careful grouping). Equal budget: INT4
wins on memory-bound serving, INT8 wins
on simplicity.

Shell 9. Falsifiable extension: measure
perplexity vs bits (32/16/8/4) on a
real model. Predict: 8 is near-free, 4
costs quality without grouping.

Shell 10. Production: quantization is
the cheapest serving win. The
stakeholder decision: pick bits by the
quality gate at the target tok/s, not
by the compression ratio. Version the
quantized artifact separately.

### C03, quantization

Motivating question: how do you fit the
model in fewer bytes without breaking
it?

Start from zero. Affine INT8: find min
and max, scale = (max - min)/255,
zero-point = round(-min/scale). Each
weight w becomes the int q = clip(round
(w/s + zp), 0, 255). To use it,
dequantize: w_hat = (q - zp) s. The
error per weight is at most s/2.

Mental model: quantization is a round-to-grid
with a ruler. The ruler's span is
[wmin, wmax], 255 ticks. A wider span
means coarser ticks.

Variables: s scale, zp zero-point int,
q the stored ints.

Computed numbers. W min -1.2, max 2.3:
s = 3.5/255 = 0.013725, zp =
round(1.2/0.013725) = 87. Dequantized
max abs error 0.005882 (<= s/2 =
0.006863, check passes), mean abs
error 0.004167. Bytes: 8 floats 32
bytes -> 8 ints 8 bytes.

Code: the quant block in
compute_run5b.py.

Checks. (1) max err <= s/2. (2) zp in
[0, 255]. (3) Dequantized min/max near
the true min/max.

Costs. Quantization is one-time, decode
uses the ints (with dequant on the
fly or int kernels).

Alternatives. Per-channel scales:
finer, more metadata. INT4: 16 levels,
4x smaller, needs grouping. fp8:
simpler, less compression. Selection
boundary: per-tensor INT8 for the
first win, per-channel/grouped for
more.

Failure case. Calibrating on the wrong
data (range too wide): error up to
s/2 with a big s. Calibrate on
representative weights/activations.

Research extension. The error-vs-bits
curve on the toy: 8/4/2 bits, max err
measured. Falsifiable: err doubles per
lost bit (s doubles).

### C04, hardware tradeoffs

Motivating question: what actually
limits tokens per second?

Start from zero. Two limits: compute
(FLOP/s) and memory (bytes/s). Decode
does ~2 FLOPs per parameter per token
but must read every parameter: for 7B
fp16, 14 GB per token. At 2 TB/s
bandwidth that is 7.00 ms/token, i.e.
at most 142.9 tok/s, regardless of
FLOP/s. This is the memory wall: decode
is memory-bound, prefill is
compute-bound. The roofline model says
the limit is the min of the two.

Mental model: serving is a warehouse.
Decode ships one box (token) per trip
and every trip loads the whole
inventory (weights). The trucks
(bandwidth), not the loaders (FLOPs),
set the pace.

Variables: model bytes, bandwidth B,
tok/s ceiling = B / bytes.

Computed numbers (authored arithmetic,
not benchmarks). 7B fp16: 14e9 bytes /
2e12 B/s = 7.00 ms/token -> 142.9
tok/s ceiling. 7B int4: 3.5e9 bytes ->
1.75 ms -> 571.4 tok/s. KV cache adds
bytes per token (C02: 24576/token on
the toy), at long n it matters.

Code: tpt = gbytes / bw.

Checks. (1) Units: s/token. (2) 4x
bytes -> 4x ceiling. (3) The ceiling
is an upper bound, real serving is
slower (batching, overhead).

Costs. This section is the cost model.

Alternatives. Bigger GPU (more
bandwidth): linear win. Smaller model:
linear win. Speculative decoding: more
tokens per weight-read. Selection
boundary: quantization first (free),
then speculation, then hardware.

Failure case. Quoting FLOP/s for a
memory-bound workload: the number is
true and irrelevant. Always roofline
both limits.

Research extension. Measure real tok/s
vs the ceiling across batch sizes.
Falsifiable: small batches sit far
below the ceiling (latency-bound),
large batches approach it.

---

## Mechanism C, from pretraining to preferences

Shell 0. The question: pretraining
predicts the next token on the internet.
How do you get a model that follows
instructions and prefers good answers?
What would change if you only did SFT?
The observable result: SFT imitates
demonstrations, preference methods
choose between answers.

Shell 1. The toy: SFT CE on two tokens:
0.733003 nats. Bradley-Terry: r_c =
1.2, r_r = 0.3: P = 0.7109, loss
0.3412 nats. PPO: (ratio 1.3, A = 0.5)
-> 0.6000, (ratio 0.5, A = -0.4) ->
-0.3200. DPO: margin 0.07, loss
0.6588, implicit rewards 0.0400 and
-0.0300. KL(pi||ref) = 0.0995 bits.

Shell 2. Objects: the SFT demos, the
preference pairs (chosen, rejected),
the reward r, the policy pi, the
reference pi_ref, the KL coefficient.
Units: nats for losses, bits for the
KL here.

Shell 3. One rule: SFT maximizes log
prob of demonstrations. Preference
methods maximize the margin between
chosen and rejected: BT says P(chosen >
rejected) = sigmoid(r_c - r_r), PPO
climbs a clipped surrogate of expected
reward, DPO writes the BT loss directly
on policy log-ratios with a KL anchor
to pi_ref. The KL term keeps the
policy near the reference.

Shell 4. Derive the algorithm: SFT: CE
over demo tokens. BT loss: -log
sigmoid(r_c - r_r). PPO: min(ratio A,
clip(ratio, 1-e, 1+e) A). DPO: -log
sigmoid(beta (log pi_c/pi_ref_c - log
pi_r/pi_ref_r)).

Shell 5. Check the invariant: BT loss
0.3412 = -log(0.7109). PPO case 1:
min(0.65, 0.6) = 0.6 (clipped binds).
Case 2: min(-0.2, -0.32) = -0.32
(pessimism). DPO: -log sigmoid(0.07)
= 0.6588.

Shell 6. Change ONE factor: beta 0.1 ->
1.0 in DPO. Predict: the margin scales
10x, the loss sharpens, the KL anchor
bites harder. (Compute it: margin 0.7,
loss 0.4032.)

Shell 7. Counterexample: DPO without
the reference (beta -> 0... or dropping
pi_ref): the loss still ranks pairs,
but nothing stops the policy from
drifting to degenerate text that
maximizes the margin. The KL anchor is
load-bearing.

Shell 8. Compare: PPO (online, needs
rollouts and a reward model, the
industry standard) vs DPO (offline,
one loss, simpler). Equal data: DPO is
cheaper, PPO explores beyond the data.

Shell 9. Falsifiable extension: on the
toy, sweep beta and plot the BT
win-rate vs KL(pi||pi_ref). Predict:
win-rate rises then the KL explodes,
the knee is the operating point.

Shell 10. Production: alignment is a
pipeline (SFT -> reward model -> PPO,
or SFT -> DPO), each stage versioned.
The stakeholder decision: the eval is
human preference or a fixed proxy
(C11), never the training loss.

### C05, SFT (supervised fine-tuning)

Motivating question: how do you teach
the model the format of good answers?

Start from zero. Collect demonstrations:
(prompt, good response) pairs. Train
exactly like pretraining (next-token
cross-entropy) but only on the response
tokens. The model learns the shape of
helpful answers: instruction ->
response, not document continuation.

Mental model: SFT is an accent course.
The model keeps its knowledge and
learns to speak "assistant".

Variables: demo tokens, CE in nats.

Computed numbers. Logits lz1 = [1.5,
0.5, -0.5], target token 0: -log
softmax(lz1)[0]. lz2 = [0.2, 1.8, 0.1],
target 1. CE = 0.733003 nats total over
the two tokens.

Code: ce = -log(softmax(lz1)[0]) -
log(softmax(lz2)[1]).

Checks. (1) CE >= 0. (2) Perfect
logits -> CE -> 0. (3) Only response
tokens counted (the prompt is masked).

Costs. Same as pretraining per token,
far fewer tokens.

Alternatives. Prompting only: no
training, weaker. Full finetune vs
LoRA: LoRA trains a low-rank delta,
cheaper. Selection boundary: SFT when
demonstrations exist, prompting when
they do not.

Failure case. SFT on bad demonstrations:
the model imitates the badness
faithfully. Garbage in, garbage out is
literal here.

Research extension. Measure the SFT
data scaling curve (quality vs number
of demos). Falsifiable: steep early,
flat later, quality of demos beats
quantity.

### C06, preference and reward modelling

Motivating question: how do you learn
"better" from comparisons?

Start from zero. Humans rank pairs:
(chosen, rejected) per prompt. The
Bradley-Terry model says P(chosen beats
rejected) = sigmoid(r_c - r_r), where
r is a learned scalar reward per
response. Train r to maximize the
log-likelihood of the human choices:
loss = -log sigmoid(r_c - r_r). The
reward model then scores new responses.

Mental model: the reward model is a
taste function. It does not generate,
it grades. BT turns "A beats B" into a
probability.

Variables: r_c, r_r scalars, the loss
in nats.

Computed numbers. r_c = 1.2, r_r =
0.3: margin 0.9, P = sigmoid(0.9) =
0.7109, loss = -log(0.7109) = 0.3412
nats.

Code: p = 1/(1+exp(-(rc-rr))), loss =
-log(p).

Checks. (1) P in (0, 1). (2) r_c = r_r
-> P = 0.5, loss = 0.6931. (3) Large
margin -> loss -> 0.

Costs. One reward scalar per response,
training is a binary classification.

Alternatives. RankNet variants,
pairwise hinge. Direct human scores
(regression): noisier. Selection
boundary: BT for pairwise labels, the
standard.

Failure case. Noisy or inconsistent
labels: the reward learns the noise
(C10 reward hacking starts here). Clean
the labels or the reward is a liar.

Research extension. Measure reward
accuracy vs label noise rate on the
toy. Falsifiable: accuracy falls
linearly, then the policy exploits the
errors (C10).

### C07, PPO

Motivating question: how do you climb
reward without falling off a cliff?

Start from zero. Policy gradient:
move pi toward actions with positive
advantage A. The danger: a big step
wrecks the policy. PPO takes the ratio
rho = pi/pi_old and maximizes min(rho
A, clip(rho, 1-e, 1+e) A). The clip
says: take the improvement, but not
more than (1+e)x of it. For A > 0 the
min keeps the pessimistic (clipped)
one, for A < 0 the min keeps the
pessimistic (unclipped, more negative)
one. Pessimism both ways.

Mental model: PPO is hill-climbing with
a leash. The leash length is e = 0.2.

Variables: rho ratio, A advantage, e =
0.2.

Computed numbers. Case 1: rho = 1.3, A
= 0.5: unclipped 0.65, clipped min(1.3,
1.2) 0.5 = 0.6: objective 0.6000 (clip
binds). Case 2: rho = 0.5, A = -0.4:
unclipped -0.2, clipped 0.8 (-0.4) =
-0.32: objective -0.3200 (pessimism:
the worse one wins the min).

Code: min(ratio*A, clip(ratio,1-e,1+e)*A).

Checks. (1) rho = 1 -> objective = A.
(2) e = 0 freezes the policy. (3) The
min is pessimistic in both cases.

Costs. Online: rollouts + reward
scoring + several epochs per batch.

Alternatives. TRPO (W11L51 title): hard
KL constraint, second-order, heavier.
REINFORCE: no clip, unstable. A2C:
no clip, simpler. Selection boundary:
PPO for LLM alignment (the standard),
TRPO when the constraint must be hard.

Failure case. e too large: the leash
is decorative, training diverges. A
misestimated (bad value net): the
climb follows noise. Both are common.

Research extension. Sweep e in {0.05,
0.1, 0.2, 0.5} on a toy bandit and
measure final reward vs KL from init.
Falsifiable: the reward-KL frontier has
a knee near 0.2.

### C08, DPO assumptions

Motivating question: can you skip the
reward model?

Start from zero. DPO starts from the BT
model and the KL-regularized RL
objective, solves for the optimal
policy in closed form, and substitutes
back: the reward becomes beta log
pi/pi_ref (the "implicit reward").
The loss is then pure supervised:
-log sigmoid(beta (log pi_c/pi_ref_c
- log pi_r/pi_ref_r)). No rollouts, no
reward net, no PPO.

Mental model: DPO is preference
learning with the RL solved out. The
policy carries its own reward function
inside the log-ratios.

Variables: beta the KL strength, the
four log-probs.

Computed numbers. beta = 0.1, log-ratio
c = 0.4, r = -0.3: margin = 0.1 (0.4 +
0.3) = 0.07. Loss = -log sigmoid(0.07)
= 0.6588. Implicit rewards: chosen
0.1 0.4 = 0.0400, rejected 0.1 (-0.3)
= -0.0300.

Code: loss = -log(sigmoid(beta*(lrc -
lrr))).

Checks. (1) beta -> 0: loss -> 0.6931
(no signal). (2) Equal log-ratios:
loss = 0.6931. (3) Implicit rewards
have the right signs.

Costs. Offline, one loss, no sampling
during training.

Alternatives. PPO (online, explores).
IPO, KTO: other offline variants.
Selection boundary: DPO when
preference pairs are fixed and compute
is limited, PPO when exploration
matters.

Failure case. Assumption audit (the
section's point): DPO assumes the BT
model is true, the reference is good,
and the pairs cover the space. Break
any: BT false -> wrong objective,
bad pi_ref -> anchored to garbage,
narrow pairs -> no generalization.
Also: DPO can drive both log-ratios
down (not just the margin up), watch
the absolute values, not just the
loss.

Research extension. Track log pi_c and
log pi_r separately during toy DPO.
Falsifiable: both fall while the margin
rises, the known DPO pathology.

### C09, KL regularization

Motivating question: what stops the
policy from drifting into nonsense?

Start from zero. Add beta KL(pi ||
pi_ref) to the objective (or subtract
from the reward). The KL prices every
bit of deviation from the reference
(SFT) model. Without it, reward
climbing finds adversarial text: high
reward, zero sense. With it, the policy
stays in the neighborhood of fluent
text.

Mental model: the KL is a leash to the
SFT model. beta sets the length.

Variables: pi, pi_ref distributions,
KL in bits here.

Computed numbers. pi = [0.5, 0.3, 0.2],
pi_ref = uniform: KL = 0.5 log2(1.5) +
0.3 log2(0.9) + 0.2 log2(0.6) = 0.0995
bits.

Code: (pi * log2(pi/pi_ref)).sum().

Checks. (1) KL >= 0. (2) pi = pi_ref ->
0. (3) Units bits (log2).

Costs. O(V) per step, usually
approximated on samples.

Alternatives. Hard KL cap (TRPO
style). No KL (DPO without anchor):
drifts. Selection boundary: soft KL
(PPO/DPO) for alignment, hard cap when
the constraint is contractual.

Failure case. beta = 0: the policy
chases reward off the manifold (C10).
beta huge: the policy never moves,
alignment does nothing. Tune the knee.

Research extension. The win-rate vs KL
frontier (shell 9). Falsifiable: the
knee exists and its beta transfers
across nearby tasks.

---

## Mechanism D, the adversary and the audit

Shell 0. The question: the reward model
is a proxy, and proxies get gamed. How
do you detect it, measure real quality,
and audit what the course actually
covers? What would change if the proxy
were perfect? The observable result:
with a perfect proxy, hacking vanishes,
with a real proxy, you need held-out
evals and audits.

Shell 1. The toy: proxy rewards A =
1.00, B = 1.20 (B wins), true quality
0.9 -> 0.7 (A was better). Eval: 4
pairs, BT win probs [0.7109, 0.4750,
0.6225, 0.5250], mean win-rate 0.5834.
Audit: 4 of 12 concepts have title-level
lecture coverage (C06-C09), 7 are authored
bridges, 1 is the audit itself.

Shell 2. Objects: the proxy reward, the
true quality, the win-rate, the audit
table. Units: unit-free scores, counts.

Shell 3. One rule: optimize the proxy,
measure the truth. The BT win-rate is
one honest eval, the coverage audit is
the honest map of what the sources
cover. Both separate the claim from
the evidence.

Shell 4. Derive the algorithm: proxy =
quality + exploit term, win-rate = mean
BT probs over held-out pairs, audit =
per-concept source status.

Shell 5. Check the invariant: the toy
hacking numbers are consistent (B wins
the proxy, loses the truth). The audit
denominators sum to 12.

Shell 6. Change ONE factor: remove the
length bonus from the proxy. Predict: B
no longer wins. Measured: proxy = q
gives A = 0.9 > 0.7 = B. The hack
disappears when the exploit term goes.

Shell 7. Counterexample: eval on the
training pairs. Win-rate 1.0 by
construction, says nothing. Held-out
pairs or it did not happen.

Shell 8. Compare: BT win-rate (cheap,
model-based) vs human eval (expensive,
ground truth). Equal pairs: humans win
on truth, BT wins on cost.

Shell 9. Falsifiable extension: inject
label noise into the 4 pairs and watch
the win-rate. Predict: it degrades
gracefully, then the policy exploits it
(C06 extension).

Shell 10. Production: the eval suite is
versioned with the model, the audit is
re-run per course edition. The
stakeholder decision: ship only what
the eval and the audit both support.

### C10, reward hacking

Motivating question: what does
"optimizing the reward" actually
optimize?

Start from zero. The reward model is a
proxy for human preference. The policy
climbs the proxy, not the truth. Toy:
two responses. A: quality 0.9, length
10. B: quality 0.7, length 50. Proxy =
quality + 0.01 length: r_A = 1.00, r_B
= 1.20. The policy picks B. True
quality fell 0.9 -> 0.7 while the proxy
rose. The policy did exactly what it
was told, what it was told was wrong.

Mental model: the proxy is a
corruptible judge. The policy is a
lawyer who finds the loopholes. Length
is the oldest loophole.

Variables: q true quality, proxy =
q + exploit.

Computed numbers. r_A = 0.9 + 0.1 =
1.00. r_B = 0.7 + 0.5 = 1.20. Policy
picks B. Quality 0.9 -> 0.7.

Code: rA = qA + 0.01*lenA.

Checks. (1) B wins the proxy. (2) A
wins the truth. (3) Removing the bonus
flips the choice back to A.

Costs. Hacking is free for the policy,
detecting it costs held-out evals.

Alternatives. Better proxy (fix the
reward). KL leash (C09: limits drift,
not direction). Human eval (truth, $$).
Selection boundary: fix the proxy when
the exploit is known, add KL always,
humans for ship decisions.

Failure case. Watching only the proxy
during training: the curve goes up and
to the right while quality collapses.
Always plot a held-out truth metric
alongside.

Research extension. Catalog exploit
terms on the toy: length, formatting,
sycophancy. Falsifiable: each has a
measurable signature in the reward
residuals.

### C11, evaluation

Motivating question: how do you score
an aligned model honestly?

Start from zero. Hold out preference
pairs the policy never trained on.
Score with the BT model: win prob per
pair = sigmoid(r_c - r_r) with the
policy's implicit or explicit rewards.
Mean = win-rate. Toy: 4 pairs, margins
0.9, -0.1, 0.5, 0.1: win probs
[0.7109, 0.4750, 0.6225, 0.5250], mean
0.5834. Above 0.5: better than the
reference on average.

Mental model: the eval is a boxing
record against held-out opponents. The
training loss is practice rounds, the
win-rate is the fight.

Variables: held-out pairs, BT probs.

Computed numbers. Margins: 1.2-0.3 =
0.9, 0.8-0.9 = -0.1, 1.5-1.0 = 0.5,
0.2-0.1 = 0.1. Sigmoids: 0.7109,
0.4750, 0.6225, 0.5250. Mean 0.5834.

Code: mean(1/(1+exp(-(a-b)))).

Checks. (1) Each prob in (0, 1). (2)
Mean in (0, 1). (3) Pairs held out
(stated, the toy has no train split).

Costs. One reward eval per pair.

Alternatives. Human eval: truth,
expensive. LLM-judge: cheap, biased.
Task benchmarks: objective, narrow.
Selection boundary: BT win-rate for
iteration, humans for ship, benchmarks
for regressions.

Failure case. Eval on training pairs:
win-rate 1.0, meaningless. Eval with
the same reward model that trained the
policy: the judge grades its own
student. Use a held-out judge or
humans.

Research extension. Correlate BT
win-rate with human win-rate across
checkpoints. Falsifiable: the
correlation is strong early, then
diverges as hacking sets in (C10).

### C12, repository and lecture coverage audit

Motivating question: what do the
sources actually cover?

Start from zero. Audit each U10 concept
against the two source lists: the 73
playlist titles (SRC-04) and the 35
repository paths (SRC-05). Rule: a
title or path name counts as
title-level coverage, contents were
never inspected (G2, G6), so nothing
counts as more.

The audit table:

| concept | playlist title? | repo path? | status |
|---|---|---|---|
| C01 temperature/top-p | none | none | authored bridge |
| C02 decode/KV cache | none | none | authored bridge |
| C03 quantization | none | none | authored bridge |
| C04 hardware tradeoffs | none | none | authored bridge |
| C05 SFT | none directly | none | authored bridge (adjacent: none) |
| C06 preference/reward | W12L52 Reward-Modelling | none | title-level |
| C07 PPO | W11L50 PPO | none | title-level |
| C08 DPO assumptions | W12L53 DPO | none | title-level |
| C09 KL regularization | W11L51 TRPO (adjacent) | none | title-level (adjacent) |
| C10 reward hacking | none | none | authored bridge |
| C11 evaluation | none directly | none | authored bridge |
| C12 this audit | n/a | n/a | the audit itself |

Honest denominators: 4 of 12 concepts
have title-level lecture coverage
(C06-C09, W11L47-L49 form the RL ramp
behind C07, not separate U10 concepts).
7 are authored bridges with no
title-level source (C01-C05, C10, C11:
inference, quantization, hardware, SFT
demos, hacking, eval). 1 is the audit.
4 + 7 + 1 = 12. The
repo has no U10-named notebook
(IITM_DGM_MHA.ipynb serves U09,
nothing serves U10). W12L54
State-space-Models is a listed topic
outside this unit's transformer scope:
recorded, not taught.

Mental model: the audit is the map
legend. It says which roads were
surveyed (titles) and which were drawn
from theory (bridges).

Checks. (1) Denominators sum: 5 + 6 +
1 = 12. (2) No concept claims more
than title-level. (3) W12L54 recorded
as out-of-scope, not omitted.

Alternatives. None: the audit is what
it is.

Failure case. Counting a title as
"taught": the audit exists to prevent
exactly this. Title-level is not
taught.

Research extension. None: audits do
not generalize. The open item is
re-running this audit if transcripts
ever become available (G2).

## Not yet understood (for the next builder)

1. No real GPU timings: all C04 numbers
are authored arithmetic. Measured
roofline pending.
2. PPO/TRPO never run: the objectives
are computed on toys, no policy
trained.
3. Reward model never trained on real
preferences: BT is computed, not fit.
4. The DPO pathology (both log-ratios
falling) stated, not measured.
5. W12L54 state-space models: title
only, no math. A bridge unit would
need its own toy.

## Lesson exercises (questions. Keys in lessons/u10/keys.md)

E01. Compute softmax(z/T) for T = 0.5,
1.0, 2.0. Verify pmax 0.8282, 0.5745,
0.4056.
E02. Compute the entropies. Verify
0.8754, 1.6174, 1.9017. Explain the
trend.
E03. Top-p 0.9: kept indices and renorm
probs under the lesson's convention.
Verify [0, 1] and [0.7311, 0.2689].
E04. KV bytes for L=12, h=8, d_h=64,
n=512, fp16. Verify 12582912. Per-token
increment?
E05. Prefill vs decode FLOPs on the toy.
Verify the ratio is n.
E06. Quantize W by hand (one entry,
0.5): compute q and w_hat. Verify
against the script's max err bound.
E07. Roofline: 7B fp16 at 2 TB/s.
Verify 7.00 ms/token and 142.9 tok/s.
E08. INT4: verify 1.75 ms and 571.4
tok/s. State the assumption (authored
arithmetic).
E09. SFT CE: compute -log softmax(lz1)
[0] by hand. Verify the total 0.733003.
E10. BT: r_c = 1.2, r_r = 0.3. Verify
P = 0.7109, loss 0.3412.
E11. PPO case 1: verify 0.6000 and name
which term the min picks. Case 2:
verify -0.3200.
E12. DPO: verify margin 0.07 and loss
0.6588. Implicit rewards?
E13. KL: verify 0.0995 bits. What does
beta do to it in the objective?
E14. Reward hacking: recompute r_A, r_B
without the length bonus. Who wins?
E15. Eval: verify the mean win-rate
0.5834 from the four margins.
E16. Audit: reproduce the denominator
(5 title-level, 6 bridges, 1 audit).
Name the 5.
E17. A colleague cites W12L52 as proof
the course "teaches reward hacking".
Refute with the audit table.
E18. Sampling: T -> 0 in code as z/0.
What breaks? The fix?
E19. Cache: forgetting the factor 2.
Recompute the toy bytes wrong, then
right.
E20. Pick PPO vs DPO for (a) a fixed
preference dataset, (b) an online
product with live feedback. Justify
each in two sentences.

## Deep oral ladders (questions. Keys in lessons/u10/keys.md)

L01. Logits to tokens: define T and
top-p, compute the three distributions,
do the KV arithmetic, debug the z/0,
compare greedy vs sampling, design the
(T,p) surface experiment.
L02. Shrinking: define affine quant,
compute s and zp, bound the error, do
the roofline, debug the wide-range
calibration, compare INT8 vs INT4,
design the bits-vs-perplexity study.
L03. Alignment: define SFT/BT/PPO/DPO,
compute all five toy losses, audit
DPO's assumptions, explain the KL's
job, debug a drifting policy, compare
PPO vs DPO, design the beta sweep.
L04. The adversary: define the proxy,
compute the hack, design the held-out
eval, debug the self-graded eval,
attack "the course teaches hacking",
run the coverage audit.
L05. End to end: from pretrain to ship:
name every stage, its input/output,
its failure mode, and the check that
catches it. One paragraph per stage.

## Implementation and debug task

Implement softmax/T/top-p, the KV byte
formula, affine quantize/dequantize,
the BT loss, the PPO clipped objective,
and the DPO loss from memory. Then
debug: (1) a colleague's top-p keeps
the wrong set (they sort ascending):
name the symptom (tail kept, head
dropped) and the fix. (2) Their PPO
objective uses max instead of min:
name the symptom (no pessimism, the
clip never binds the right way) and
the fix.

## Changed-constraint scenarios

S1. The deploy target is a phone (2 GB
RAM, no GPU). The 7B model cannot fit
even at INT4 (3.5 GB). Name three
levers (smaller model, distillation,
offload) and compute the bytes for one
concrete feasible config (e.g. 1B
INT4 = 0.5 GB + KV).
S2. The preference data is pairwise
but noisy (30% flips). Which lesson
numbers change (BT loss up, reward
accuracy down) and which method
degrades faster (PPO, which explores
the noise, vs DPO, which fits it)?
Justify with the C06 research
extension.

## Research-critique question

"Lower DPO loss means a better model."
Attack: name what the loss measures
(margin on training pairs), what can
improve while quality falls (the
margin via both log-ratios dropping,
C08), and the experiment that decides
(win-rate on held-out pairs + human
eval vs the loss curve across
checkpoints).
