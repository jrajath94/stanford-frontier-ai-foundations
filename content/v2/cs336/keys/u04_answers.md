# U04 answer key , lesson assessments

## R1 (remediation)

MHA cache 50.3 MB at g=8. MQA has g=1: 50.3/8 = 6.3 MB. Score FLOPs
unchanged: queries still number h=8. Rubric: cache 2 pts, FLOPs 1 pt.

## A1

(a) 2*B*h*T*w*dh versus 2*B*h*T*T*dh.
(b) Ladder. Window: keys in [t-w, t]. 8x: T/w = 1024/128. Banded mask:
ones on the diagonal band plus causal. Full: w=T. Debug: check the
mask triangle, a leak shows as nonzero future weights. Critique:
locality. Transfer: T=100k still needs 100k*128 scores per query,
the next lever is linear attention or SSM.

## A2

(a) Cache per token per layer: 2*g*dh bytes, g = h (MHA), 1 (MQA),
between (GQA).
(b) Ladder. KV head: one key/value projection head. MQA by hand: 50.3/8
= 6.3 MB. FLOPs: scores stay (B,h,T,T). Repeat: repeat_interleave KV
heads to h. Debug: uptrain instead of averaging. Critique: dh fixed.
Transfer: the cache-bound variant's advantage grows: decode traffic is
proportional to g, and larger batches multiply it.

## A3

(a) Cache: 2*dc bytes per token per layer (plus small RoPE key).
(b) Ladder. Latent: the compressed KV vector. 12.6 MB: 2*128*2 bytes *
1024 * 12 * 2 / 1e6. Extra FLOPs: the per-head up-projections at
decode. Down/up: c = xW_down, k_h = cW_up. Debug: raise dc, flat
weights indict too-small latent. Critique: joint training. Transfer:
MLA helps decode (memory-bound) and hurts prefill slightly
(compute-bound): net win for serving.

## A4

(a) S_t = S_{t-1} + phi(k_t)v_t^T, out_t = phi(q_t)^T S_t / phi(q_t)^T z_t.
(b) Ladder. State: the (dh,dh) accumulator. Two steps by hand on the
toy. Associativity: (QK^T)V = Q(K^T V). Softmax: sharper selection.
Debug: use elu+1 or relu+eps. Critique: the kernel choice. Transfer:
state = 128^2 * 4 bytes = 64 KB per head, the quadratic form needs
T^2 = 1e12 scores: impossible.

## A5

(a) h_t = A h_{t-1} + B x_t, O(T*N) time, O(N) memory.
(b) Ladder. State: the N-vector summary. Unroll: y = convolution with
(CB, CAB, ...). Scan: the recurrence. Linear attention: matrix state.
Debug: check A's eigenvalues, project or reparameterize. Critique:
summaries lose exact recall. Transfer: a hybrid: SSM layers for the
bulk plus a few full-attention layers for recall.

## A6

(a) Gated: S_t = alpha_t S_{t-1} + beta_t outer(phi(k_t),v_t). Delta:
also subtract the old key content.
(b) Ladder. Valves: learned forget/write scales. One gated step by
hand. Delta: the projection removes stale content. Plain linear:
simpler. Debug: make gates input-dependent. Critique: selectivity.
Transfer: the gated delta variant: repeated keys need overwriting,
not just decay.

## A7

(a) Total E*expert params, active k*expert per token.
(b) Ladder. Expert: one SwiGLU FFN. 4.0x: 8/2. Conditional compute:
pay per token, own in parameters. Dense: simpler. Debug: add the aux
loss, check utilization. Critique: k << E. Transfer: memory-bound
inference favors smaller E (all experts resident), the E/k ratio must
also provision HBM.

## A8

(a) p = softmax(xW_r), output = sum of top-k weighted experts.
(b) Ladder. Scores: expert selection weights. 4 tokens, 3 experts:
score, pick top-k, weight. Softmax: competition, sigmoid:
independence. Hash: fixed, balanced. Debug: entropy collapse indicts
the router, add balancing. Critique: joint training. Transfer: k=1
makes selection discontinuous, the standard fix is the aux loss plus
occasional exploration (or k=2).

## A9

(a) y = sum of top-k weighted expert outputs.
(b) Ladder. top-k: the k highest scores. Hand combine: weight and sum
two expert vectors. k=2: smoothing at 2x compute. Soft: no saving.
Debug: match the renormalization convention to the checkpoint.
Critique: k <= E. Transfer: top-1 risks picking the wrong expert on a
near tie, top-2 hedges.

## A10

(a) C = factor * T * k / E.
(b) Ladder. Dropped token: routed but unprocessed. 5: 1.25*16*2/8.
Factor: memory versus drops. Expert-choice: experts pick tokens.
Debug: drops concentrate on popular experts, raise the factor or add
balancing. Critique: drops are not free. Transfer: variable expert
loads cause the spikes, fix with no-drop inference batching or
expert-choice.

## A11

(a) L_aux = alpha * E * sum(frac_e * pbar_e), uniform gives k.
(b) Ladder. frac_e: assigned fraction. Uniform: (k/E)(1/E) per expert,
sum k/E, times E = k. E factor: normalizes across E. Expert-choice:
balance by construction. Debug: lower alpha. Critique:
differentiability via straight-through. Transfer: tune alpha down and
check specialization (per-expert token clusters), also try k=2.

## A12

(a) Total, active per token, active FLOPs.
(b) Ladder. Active params: k experts' params. 4.0x: 8/2. Memory: total
rules, bill: active rules. Dense: one number. Debug: provision from
total params. Critique: router negligible. Transfer: MoE usually wins
in steps-to-quality but the all-to-all can dominate wall time, measure
both.
