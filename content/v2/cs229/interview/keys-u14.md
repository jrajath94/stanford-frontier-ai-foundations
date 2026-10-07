# Interview keys, U14

Date: 2026-10-06. Each answer gives the minimum sufficient
explanation, a strong answer, common red flags, a rubric, and
remediation. Computed values: numpy 1.26.x, float64.

## B1

Minimum: a token is one symbol the model reads (word, piece,
punctuation, byte, marker). BPE starts from bytes/characters
and repeatedly merges the most frequent adjacent pair.
Strong: adds the subword compromise (between char and word
level) and that the tokenizer is fixed before pretraining.
Red flags: "tokens are words". Missing the fixed-tokenizer
requirement.
Rubric: 1 pt token, 1 pt BPE, 1 pt fixed before training.
Remediation: lesson C01, SL-01.

## B2

Minimum: p(x_1, ..., x_T) = p(x_1) p(x_2 | x_1) ... p(x_T |
x_1, ..., x_{T-1}).
Strong: adds why (support |V|^T is astronomical, each
conditional has support |V|).
Red flags: dropping the conditioning. Writing the product
without the chain rule.
Rubric: 1 pt formula, 1 pt support argument.
Remediation: lesson C02, SL-02.

## B3

Minimum: (1/T) sum_t l_ce(f_theta(x_0, ..., x_{t-1}), x_t),
averaged over sequences, AdamW on mini-batches.
Strong: notes the per-position |V|-way classification reading
and that labels are free (the next token).
Red flags: including x_t in the inputs. Forgetting the 1/T
normalization.
Rubric: 1 pt loss, 1 pt free-label reading, 1 pt optimizer.
Remediation: lesson C03, SL-02.

## B4

Minimum: p_{t,j} = softmax(q_t k_j^T / sqrt(d_h)),
h^out_t = sum_j p_{t,j} v_j.
Strong: gives the Q/K/V projections with shapes and the
query/key/value intuition.
Red flags: missing the scaling. Summing over t instead of j.
Rubric: 1 pt scores, 1 pt output, 1 pt scaling.
Remediation: lesson C04, C05, SL-04.

## B5

Minimum: the q-k dot product has variance ~d_h at
initialization, unscaled logits saturate the softmax and kill
gradients. Scaling keeps logit variance near 1.
Strong: flags it as a heuristic (real q, k are not
independent) and names the saturated-regime symptom.
Red flags: "it is exact". Confusing with layer norm.
Rubric: 1 pt variance argument, 1 pt saturation consequence.
Remediation: lesson C05, SL-04.

## B6

Minimum: M_{t,j} = 0 for j <= t, -infinity for j > t, added
before the softmax. Guarantees position t depends only on
current and past tokens (autoregressive property).
Strong: adds parallel training (all positions in one pass)
and other mask uses (padding, blocks).
Red flags: applying the mask after softmax. Saying it "makes
training faster" as the main point.
Rubric: 1 pt mask, 1 pt guarantee, 1 pt before-softmax.
Remediation: lesson C05, SL-04.

## D1

D1.1: q_t = h^in_t W^Q etc., W in R^{d x d_h}. H^in in
R^{T x d}, H^out in R^{T x d_h}, scores softmax(q_t k_j^T /
sqrt(d_h)), output sum_j p_{t,j} v_j.
D1.2: rows [1, 0, 0], [0.3302, 0.6698, 0], [0.4011, 0.4011,
0.1978].
D1.3: stacking the per-position rules gives Q, K, V with rows
q_t, k_t, v_t, the t-th row of softmax_row(Q K^T / c) V is
exactly the per-position output.
D1.4: the causal mask is absent or applied after the softmax.
Checks: (1) perturb x_{t+1}, assert u_t unchanged, (2) assert
the score matrix has exact zeros above the diagonal.
D1.5: the mask becomes all-zeros (bidirectional), the
autoregressive factorization and the next-token loss no
longer apply (masked-LM loss instead, per the notes' BERT
flag).
Rubric: 1 pt each, 2 pts for D1.3.
Remediation: SL-03, SL-04.

## D2

D2.1: per head, training/prefill O(T^2 d_h) compute, naive
T^2 memory per head (FlashAttention: O(T)), decode KV cache
O(T d_h) per head.
D2.2: per layer at T = 8192 fp16: MHA 128.0 MiB, GQA (n_g =
8) 32.0 MiB, MQA 4.0 MiB.
D2.3: the cache stores one K/V head per group, and there are
n_g groups, query heads share within a group, so the stored
tensors scale with n_g.
D2.4: 4x overprovision (128 vs 32 MiB per layer at 8k, 512 vs
128 at 32k). Fix: provision with 2 * T * n_g * d_h * bytes.
D2.5: hidden assumption: quality is equal at all n_g. Test:
fix n_h, sweep n_g in {1, 2, 4, 8, 16, 32} on long-context
tasks, and find where quality first drops.
Rubric: 1 pt each, 2 pts for D2.5.
Remediation: SL-05, SL-06.

## Q1

Row t of Q K^T / c holds q_t k_j^T / c. Adding the causal mask
row sets entries j > t to -infinity, softmax_row then yields
probabilities that are exactly 0 for j > t and proportional
to exp(q_t k_j^T / c) for j <= t: the vector of (17.23).

## Q2

tau = 0.5: [0.8282, 0.1121, 0.0412, 0.0185]. tau = 1.0:
[0.5745, 0.2114, 0.1282, 0.0859]. tau = 2.0: [0.4056, 0.2460,
0.1916, 0.1569]. Top-p 0.9: kept [0, 1, 2] (cumulative
0.9141), renormalized [0.6285, 0.2312, 0.1402]. Top-k k = 2:
kept [0, 1], renormalized [0.7311, 0.2689].

## T1

Bug 1: the new key/value are appended after the attention
computation, so each token never attends to its own position,
every step is off by one and the error compounds down the
sequence. Fix: append k_new/v_new to the cache before
computing scores (equivalently, compute against the extended
cache).
Bug 2: the code pairs one (dh,) query with a (t, dh) cache,
assuming one key head. Under GQA the query has n_h heads and
the cache has n_g < n_h key heads, the heads must be matched
by the group index g(j) of (17.32), or shapes/broadcasting
silently pair the wrong heads.
Check: prefill the full sequence without a cache and compare
the last-position output to the cached incremental step,
they must agree to numerical precision.
Rubric: 1 pt per bug, 1 pt per fix, 1 pt check.
Remediation: lesson C06, SL-05, SL-06.

## S1

Architectural: switch MHA to GQA/MQA (cuts cache with n_g) or
add sliding-window attention (bounds the active cache), each
sacrifices some long-range quality. Systems: per-sequence
cache accounting with eviction/offload instead of max-context
provisioning, sacrifices worst-case latency guarantees.
Measure quality at 128k before shipping either.

## S2

Symptoms: (1) the loss mainly predicts prompt tokens, so it
falls while answer quality does not improve, (2) generations
degrade or the model parrots prompt-like text. Fix: flip the
loss mask (1 on answer positions, 0 on prompt positions).

## R1

Confound: zero-shot scores depend on the exact prompt wording,
so the gap may be prompt engineering, not pretraining.
Protocol: fix a set of prompts (or a prompt distribution)
before seeing results, evaluate both models on the same
prompts, and report the mean and spread across prompt
variants, the pretraining claim needs a win across the
distribution, not on one tuned prompt.
