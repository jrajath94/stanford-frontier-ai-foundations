# Lab 08: Attention by hand, decoding, KV-cache accounting

Date: 2026-10-06. Unit: cs229-U14.
Work the tasks, then check keys-lab-08.md. Run all code.

Setup: numpy only. No other installs.

## Task 1: causal attention by hand

Q rows: [1, 0], [0, 1], [1, 1]. K rows: [1, 0],
[0, 1], [1, -1]. V rows: [2, 0], [0, 2],
[1, 1]. Compute the causal masked attention
output H^out by hand (scores q_t k_j^T /
sqrt(2), causal mask, softmax, times V).
Report the full 3x2 output. Then verify with
code. State the two structural checks the
score matrix must pass.

## Task 2: decoding knobs

Logits [2.0, 1.0, 0.5, 0.1]. Compute the
sampling distribution at tau = 0.5, 1.0, 2.0.
At tau = 1.0, apply top-p with p = 0.9:
report the kept indices and the renormalized
distribution. Also apply top-k with k = 2 and
report the renormalized distribution. State
in one sentence what temperature changes and
what it does not change.

## Task 3: KV-cache ledger

Model: d = 4096, n_h = 32, d_h = 128,
fp16, 32 layers. Compute the per-layer and
total KV-cache memory for T in {2048, 8192,
32768} under MHA (n_g = 32), GQA (n_g = 8),
and MQA (n_g = 1). Report the full table in
MiB and the MHA-to-MQA ratio at each T.
State which term of the ledger each variant
changes and which term none of them changes.

## Deliverable

A short log: the three result blocks with numbers. No
essay. The numbers must match keys-lab-08.md within the
stated tolerance.
