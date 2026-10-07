# U04 interview key

## B1-B6

B1. Cache = 2 * kv_heads * dh * bytes per token per layer. MHA: h heads.
GQA: g groups. MQA: 1 head.
B2. 2*B*h*T*w*dh versus 2*B*h*T*T*dh: a factor T/w cut.
B3. A kernel feature map lets (QK^T)V regroup as Q(K^T V), the bracket
is a fixed-size state updated per token: O(T) time.
B4. Total: all experts (memory, capacity). Active: k experts per token
(compute, latency).
B5. A routed token skipped because its expert's inbox is full. The
capacity factor controls the drop rate.
B6. The correlation between assigned-token fraction and mean router
probability per expert, it pushes routing toward uniform.

## L1

L1.1 2 * g * dh * bytes_per_element.
L1.2 dh=64. MHA: 2*8*64*2*1024*12*2/1e6 = 50.3 MB. MQA: /8 = 6.3 MB.
L1.3 Scores stay (B,h,T,T): query count unchanged, only the K/V
projection width and cache shrink with g.
L1.4 repeat_interleave KV heads to h, then standard attention, with
g=h the repeat is identity and outputs equal MHA exactly.
L1.5 At equal bytes: MQA (g=1) vs GQA-2 vs MLA-128 (all ~12.6 MB here
except MQA at 6.3). Debug: uptrain, do not average heads. Critique:
dh fixed. Experiment: fixed training FLOPs, compare loss.

## L2

L2.1 Scores: softmax(xW_r). top-k: selected experts. Capacity:
factor*T*k/E. Dropped: overflow tokens.
L2.2 Capacity = 1.25*16*2/8 = 5. Tokens beyond 5 per expert drop,
3 dropped in the compute run.
L2.3 L_aux = alpha*E*sum(frac_e*pbar_e), uniform gives k.
L2.4 Per-expert queues with overflow counting, dropped + processed =
T*k.
L2.5 Expert-choice: experts pick tokens, balance by construction.
Debug: collapse shows as one hot expert, add/raise the aux weight.
Critique: drops are not free. Experiment: sweep alpha, plot task loss
versus max load.

## E1

dh = 64. Per token per layer: 2*8*64*2 = 2048 bytes. Total:
2048*8192*32*16 = 8,589,934,592 bytes = 8.59 GB. MQA: /8 = 1.07 GB.
Savings: 7.52 GB. Rubric: per-token 1 pt, total 2 pts, MQA + savings
1 pt. Red flag: forgetting the factor 2 for K and V.

## E2

Expert: 3*4096*14336 = 176,160,768. Total: 64x = 11.27B. Active:
2x = 352.3M. Ratio 32. Collapsed to 4 experts: the model behaves as a
4-expert MoE, effective total 704.7M, active per token unchanged at
352.3M. Red flag: "the model now has 352M params".

## D1

Bug: np.tile interleaves KV heads ([kv0,kv1,kv0,kv1,...]) instead of
grouping them ([kv0,kv0,...,kv1,kv1,...]). Query heads then read the
wrong KV heads. When g=h the repeat count is 1 and both are identity,
so the bug hides. Fix: np.repeat(K, h//g, axis=1). Rubric: find 2 pts,
fix 1 pt, explain the g=h masking 1 pt.

## S1

Assume h=32, dh=64, bf16. Bytes per token per layer: MHA (g=32): 8192.
GQA-8: 2048. MLA-128: 512. MQA: 256. Rank largest to smallest: MHA,
GQA-8, MLA-128, MQA. Quality risk of MQA: one shared KV head blurs
head specialization, the next lever is MLA-128 or GQA-8. Red flag:
ranking without stating h. Rubric: formula 1 pt, ranking 2 pts, risk
1 pt.

## S2

Fix 1: raise the capacity factor (fewer drops, more memory). Fix 2:
add or raise the load-balancing aux loss (fixes the cause). Try the
aux loss first: drops are a symptom of imbalance. Red flag: "raise
the factor to 10" without the memory cost.
