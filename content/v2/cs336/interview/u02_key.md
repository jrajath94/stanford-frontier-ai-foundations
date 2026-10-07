# U02 interview key

## B1-B6

B1. (...,M,K)@(...,K,N)->(...,M,N). Read right to left: the last two
axes multiply, leading axes batch and broadcast.
B2. Attention scores per head: queries (b,h,i,d) against keys (b,h,j,d),
output (b,h,i,j).
B3. A view shares storage (new metadata only), a copy owns new storage.
Strides reveal it: a transpose swaps strides and shares, a copy has
standard strides and no sharing.
B4. Gradients, optimizer state (16 bytes/param for Adam), and activations
(B,T,L scaling) add to the parameters.
B5. 2MNK per matmul (MNK mults + MNK adds). One toy layer: 8.858 GFLOP
forward (qkv 1.611, scores 0.268, out 0.537, ffn 6.443).
B6. FLOPs per byte. 1.0 FLOP/byte is deeply memory-bound: buy bandwidth
or cut traffic, not FLOPs.

## L1

L1.1 Rows: parameters, gradients, optimizer state, activations,
workspace/input batch.
L1.2 41.55M * 16 = 664.8 MB.
L1.3 Mixed keeps the fp32 master and moments (4+4+4) and only shrinks
compute tensors (2+2), the sum equals fp32 Adam. Savings land in
activations and compute, not optimizer state.
L1.4 Ledger: sum of C04+C05+workspace, verdict fits if total * 1.15 is
below device memory.
L1.5 Snapshot: allocator ground truth after a run. Debug: add NCCL
buffers and loader workers, the single-device ledger undercounts them.
Critique: one device, no fragmentation. Experiment: extend with NCCL
rows, validate on 2 GPUs within margin.

## L2

L2.1 MFU = achieved model FLOP/s over peak FLOP/s, with the honest model
count as denominator.
L2.2 min(100, 50*1) = 50 TFLOP/s, memory-bound.
L2.3 Cap = min(peak, intensity * bandwidth), balance = peak/bandwidth =
100 FLOP/byte here, below is memory-bound, above compute-bound.
L2.4 mfu returns rate, cap, rate/cap, MFU > 1 means the FLOP count is
inflated (recompute double-counted).
L2.5 Counters: per-kernel truth for debugging, MFU: whole-job tracking.
Debug: 90 percent of peak on a memory-bound shape is impossible, the
denominator is wrong. Critique: the count must be the model count.
Experiment: MFU versus batch size, expect rise then plateau at the
roofline.

## E1

(a) Per layer 12*d^2 = 12*1024^2 = 12,582,912. x24 = 301,989,888.
Embeddings 32000*1024 = 32,768,000. Total 334,757,888 ≈ 334.8M params.
(b) 16 bytes/param * 334.76M = 5.36 GB.
(c) qkv = 3*2*8*512*1024^2 = 12.88 GFLOP. scores =
2*8*16*512^2*64 = 4.29 GFLOP. out = 2*8*512*1024^2 = 8.59 GFLOP.
ffn = 3*2*8*512*1024*4096 = 103.08 GFLOP. Total ≈ 128.85 GFLOP.
Rubric: 2 pts per part, the method carries partial credit.
Red flag: counting 2 FFN maps instead of 3.

## E2

One token: (1,4096)@(4096,12288). FLOPs = 2*12288*4096 ≈ 1.01e8. Bytes
fp16 ≈ 2*(4096 + 12288*4096 + 12288) ≈ 1.01e8. Intensity ≈ 1.0
FLOP/byte: memory-bound. First lever: shrink weight traffic
(quantization) or raise batch size. Red flag: "optimize the matmul
kernel".

## D1

Bug: SwiGLU FFN has three linear maps (gate, up, down), not two, the
line must read `3 * 2 * B * T * d * dff`. Changed claim: per-layer
forward is 8.858 GFLOP with 72.7 percent FFN share (not 6.711/64
percent). Rubric: find 2 pts, fix 1 pt, claim update 1 pt.

## S1

(1) Optimizer state: ZeRO sharding, cost is communication. (2)
Activations: checkpointing, cost is ~33 percent throughput. (3) Batch:
smaller microbatch, cost is throughput. Order by cost: shard, then
checkpoint, then shrink. Red flag: "use fp16" as if it shrinks Adam
state (it does not).

## S2

Activations row explodes (T^2 attention term, 64x). The score term
crosses the FFN term (crossover near T = 3*dff). Mitigations in order:
(1) activation checkpointing for the attention block, (2) a
subquadratic attention variant (U04) or sequence parallelism (U09).

## R1

Gaps: (1) no stated FLOP count, close by publishing the counted ops.
(2) no intensity or roofline position, close with the roofline plot.
(3) no baseline, close with the unfused baseline on the same hardware.
Red flag: accepting "percent of peak" without the denominator.
