# U04 lab , attention alternatives and MoE

Environment: CPython 3 with numpy. Run: `python3 u04_lab_run.py`.
Expected outputs in `u04_lab_key.md` (executed 2026-10-06).

## Task 1 , local attention

Implement `local_mask(T, w)`. Assert row t allows exactly min(t+1, w)
keys, rows sum to 1 after softmax, and w=T recovers full attention.
Report the FLOP ratio for T=1024, w=128.

## Task 2 , GQA

Implement GQA attention via KV-head repeat. Assert GQA with g=h equals
MHA exactly and output shapes match. Print the cache table for g in
{8, 2, 1} at B=2, T=1024, d=512, L=12, bf16.

## Task 3 , MLA

Implement down/up projections. Assert the reconstruction check: with
dc=d and identity projections, MLA attention equals MHA. Report cache
bytes for dc=128.

## Task 4 , linear attention

Implement the recurrence. Assert equivalence with the causal quadratic
form (max dev < 1e-12) and state size independent of T.

## Task 5 , SSM scan

Implement `ssm_scan` for a scalar system. Assert recurrence equals the
unrolled convolution. Run the hand example (A=0.9, x=[1,0,1]) and
report h.

## Task 6 , gated recurrence

Implement the gated update. Assert reduction to plain linear attention
at alpha=beta=1 and causality (output at t unchanged when future
inputs change).

## Task 7 , MoE layer

Implement the MoE layer with a loop over experts. Assert E=1, k=1
equals the dense FFN. Report total, active params and the ratio for
E=8, k=2, d=512, dff=2048.

## Task 8 , router

Implement `route` (softmax, top-k). On 16 seeded tokens with E=8,
report the load histogram and assert each token gets k experts.

## Task 9 , top-k combine

Implement `topk_combine`. Assert k=E equals the full weighted sum and
k=1 picks the argmax expert.

## Task 10 , dispatch with capacity

Implement `dispatch` with capacity factor 1.25. Assert no expert
exceeds capacity and dropped + processed = T*k. Report drops for the
16-token toy.

## Task 11 , aux loss

Implement `aux_loss`. Assert uniform inputs give exactly k=2. Report
the toy value versus uniform.

## Task 12 , accounting

Implement `moe_accounting`. Assert the k=E and E=1 edge cases. Report
the three numbers for the toy config.
