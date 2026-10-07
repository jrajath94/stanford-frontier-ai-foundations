# U02 lab , tensor programming and resource accounting

Environment: CPython 3 with numpy. No GPU needed, every task is a
correctness task on CPU. Run: `python3 u02_lab_run.py`. Expected outputs
are in `u02_lab_key.md` (executed 2026-10-06).

## Task 1 , shape contracts

Implement `check_shapes(t, expected, op)` with None as wildcard. Write
contract tests for matmul (4,256,512)@(512,2048), embedding lookup,
softmax rows, and layer norm. Assert a wrong contract raises with the op
name in the message.

## Task 2 , attention einsums

Implement `attention_scores(Q, K)` ("bhid,bhjd->bhij") and
`attention_mix(P, V)` ("bhij,bhjd->bhid") with output shape asserts.
Verify against explicit transpose-based code with allclose on seeded
random inputs (B=2, h=4, T=16, d_h=8).

## Task 3 , strides, views, copies

On a (4,6) float32 array: predict the transpose strides, assert
shares_memory for the view, assert no sharing after ascontiguousarray,
and demonstrate aliasing (write through the view, read the original).
Implement `is_c_contig(shape, strides, itemsize)` and check both.

## Task 4 , activation estimator

Implement `activation_estimate` per U02-C04. For B=4, T=256, d=512,
dff=2048, h=8, L=8, report total fp16 bytes and the per-term breakdown.
Assert doubling B doubles the total and doubling T more than triples it.

## Task 5 , optimizer bytes

Implement `optimizer_bytes(params, mode)` for sgd, adam_fp32,
adam_mixed. For 41.55M params assert the 12/16/16 bytes-per-param ratios
and print totals in MB.

## Task 6 , FLOP counter

Implement `layer_flops(B, T, d, dff, h)` with the per-op split. For the
toy layer assert total 6.711 GFLOP within 1 percent and FFN share above
60 percent. Find the T where the attention score term passes the FFN
term (d=512, dff=2048 fixed).

## Task 7 , tiny autograd op count

Build a minimal scalar autograd (Value class, ~40 lines) or hand-count
a two-layer MLP: count multiply-adds in forward and backward on paper,
then assert the backward/forward ratio lands in [1.8, 2.4].

## Task 8 , intensity

Implement `intensity(M, N, K)`. Assert the three computed values
(170.7, 204.8, 1.0 within 1 percent) and the square-scaling and M=1
properties from U02-C08.

## Task 9 , MFU

Implement `mfu(model_flops, seconds, peak, ai, bw)` per U02-C09. For a
step doing 26.6 GFLOP per layer-equivalent at 40 TFLOP/s on the toy
device with intensity 50, report the roofline cap and MFU against the
cap. Assert MFU above 1 raises.

## Task 10 , ledger

Implement `ledger` per U02-C10 for the toy model. Assert the total and
the verdict for 1 GB and 512 MB budgets.

## Task 11 , dtype demo

Accumulate 1e-6 ten thousand times in fp16 and fp32, report both sums
versus 0.01. Report activation bytes in fp16 versus fp32 from the task 4
estimator.

## Task 12 , evidence classification

Run the full U02 suite. For each task print whether its checks are
correctness evidence or performance evidence, with one line of
justification each.
