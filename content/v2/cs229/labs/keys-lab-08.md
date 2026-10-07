# keys-lab-08.md

Date: 2026-10-06. Reference outputs for lab 08.

Computed with numpy 1.26.x, float64, on this machine.
Tolerances below absorb platform float differences.

## Task 1

Attention scores (rows): [1, 0, 0],
[0.3302, 0.6698, 0], [0.4011, 0.4011,
0.1978]. H^out = A V:
row 1: [2.0000, 0.0000].
row 2: 0.3302 * [2, 0] + 0.6698 * [0, 2] =
[0.6605, 1.3395].
row 3: 0.4011 * [2, 0] + 0.4011 * [0, 2] +
0.1978 * [1, 1] = [1.0000, 1.0000].
Tolerance: 0.001. Structural checks: each
row sums to 1, and every entry above the
diagonal is exactly 0 (the causal mask).

## Task 2

tau = 0.5: [0.8282, 0.1121, 0.0412,
0.0185]. tau = 1.0: [0.5745, 0.2114,
0.1282, 0.0859]. tau = 2.0: [0.4056,
0.2460, 0.1916, 0.1569]. Top-p 0.9 at tau
= 1.0: kept [0, 1, 2], renormalized
[0.6285, 0.2312, 0.1402]. Top-k k = 2:
kept [0, 1], renormalized [0.7311,
0.2689]. Tolerance: 0.001. Temperature
changes the sharpness of the sampling
distribution, it does not change the model
weights or the training loss.

## Task 3

Per-layer KV cache (MiB), fp16:
T=2048: MHA 32.0, GQA 8.0, MQA 1.0.
T=8192: MHA 128.0, GQA 32.0, MQA 4.0.
T=32768: MHA 512.0, GQA 128.0, MQA 16.0.
Total over 32 layers (GiB): T=2048: 1.00 /
0.25 / 0.03. T=8192: 4.00 / 1.00 / 0.12.
T=32768: 16.00 / 4.00 / 0.50. MHA-to-MQA
ratio: 32x at every T. Tolerance: exact.
The variants change the KV-cache memory
term (scales with n_g), none changes the
O(T^2 d) attention compute term.
