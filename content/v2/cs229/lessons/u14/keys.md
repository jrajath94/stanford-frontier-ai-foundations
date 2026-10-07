# Keys: Lesson 14, LLM architecture and training

## Breadth recall

E01: A token is one symbol the model reads:
a word, word piece, punctuation, byte, or
special marker. BPE starts from bytes or
characters and repeatedly merges the most
frequent adjacent pair, giving a subword
vocabulary between character and word
level.

E02: p(x_1, ..., x_T) = p(x_1) p(x_2 | x_1)
... p(x_T | x_1, ..., x_{T-1}).

E03: loss = (1/T) sum_t l_ce(f_theta(x_0,
..., x_{t-1}), x_t): mean cross-entropy of
the next-token logits, averaged over
positions and sequences.

E04: Scores p_{t,j} = softmax(q_t k_j^T /
sqrt(d_h)), output h^out_t = sum_j p_{t,j}
v_j.

E05: At initialization the q-k dot product
has variance d_h, without scaling the logits
grow with head dimension, the softmax
saturates, and gradients vanish. Dividing by
sqrt(d_h) keeps logit variance near 1.

E06: M_{t,j} = 0 for j <= t, -infinity for
j > t, added before the softmax. It
guarantees position t sees only the current
and past tokens: the autoregressive
property.

## Deep oral ladders

L01: (1) (17.14)-(17.18). (2) Rows: [1, 0,
0], [0.3302, 0.6698, 0], [0.4011, 0.4011,
0.1978]. (3) Stack the per-position rules:
rows of Q, K, V are the q_t, k_t, v_t, so
the t-th row of softmax_row(Q K^T / c) V is
exactly sum_j p_{t,j} v_j. (4) Check row
sums = 1 and exact zeros above the diagonal.
(5) Causal: for generation. Bidirectional:
for encoders, leaks the future, unusable for
autoregressive sampling. (6) The mask is
missing or applied after the softmax. (7)
The independence assumption is false in
real transformers, it is a heuristic that
works at initialization. (8) Perturb future
tokens, assert u_t unchanged, check exact
zero triangle on random inputs.

L02: (1) O(T^2 d_h) compute, O(T d_h)
cache per head. (2) T = 8192, fp16, per
layer: MHA 128.0 MiB, GQA (n_g = 8) 32.0
MiB, MQA 4.0 MiB. (3) Cache holds n_g KV
heads, not n_h: 2 * T * n_g * d_h * bytes.
(4) The calculator with the table above.
(5) MHA: best quality, biggest cache. GQA:
the middle. MQA: smallest cache, shared
bottleneck. Sliding window: bounded cache,
loses long range. (6) Provisioned with MHA
numbers on a GQA model: 4x overprovision,
the fix is the n_g-aware formula. (7) MQA
shares one KV head: it can bottleneck
distinct query behaviors on quality-
sensitive long-context tasks. (8) For 128k
serving: GQA or MQA with a measured quality
check, sliding window if the task is local,
plus cache-per-sequence accounting at real
lengths.

## Analytical exercises

E07: The t-th row of Q K^T / c holds q_t
k_j^T / c for j = 1..T. Adding the mask row
and applying softmax_row gives exactly the
score vector of (17.17), with the causal
mask the entries j > t are -infinity, hence
0 after softmax, matching (17.23).

E08: tau = 1: [0.5745, 0.2114, 0.1282,
0.0859]. tau = 0.5: [0.8282, 0.1121,
0.0412, 0.0185]. tau = 2.0: [0.4056,
0.2460, 0.1916, 0.1569]. Top-p 0.9 at tau
= 1 keeps indices [0, 1, 2] (cumulative
0.9141), renormalized [0.6285, 0.2312,
0.1402].

## Failure diagnosis

E09: Near-zero training loss after one epoch
means the model sees the future: the causal
mask is absent or misplaced (or the data
leaks labels into inputs). At generation
there is no future to read, so text is
incoherent. Fix: run the mask test (perturb
x_{t+1}, check u_t) and the exact-zero
triangle check.

## Counterfactual comparison

E10: Zero-shot wins when there are no
labels, the task is well described in words,
and iteration must be instant. SFT wins when
5,000 pairs exist and the output format or
style needs to be reliable. The loss mask
buys B answer-only training: capacity goes
to p(y | x), not to prediction of the prompt.

## Research question

E11: Falsifiable claim: as n_g falls from
n_h to 1 at fixed n_h, long-context quality
is flat down to some threshold n_g* and then
drops, n_g* grows with the task's
dependence on distant context.

## Implementation task

E12: Verified by the numbers: the attention
toy matches [1,0,0] / [0.3302,0.6698,0] /
[0.4011,0.4011,0.1978], the cache table
matches 128.0 / 32.0 / 4.0 MiB per layer at
T = 8192, the temperature and top-p values
match E08. See lab-08 keys.
