# U02 answer key , lesson assessments

## R1 (remediation)

(1,512) by (512,512), fp16. FLOPs = 2*1*512*512 = 524288. Bytes =
2*(1*512 + 512*512 + 1*512) = 2*263168 = 526336. Intensity =
524288/526336 = 1.0 FLOP/byte (approx). Memory-bound on any realistic
device. Rubric: FLOPs 1 pt, bytes 1 pt, intensity and verdict 1 pt.
Red flag: forgetting the factor 2 in FLOPs.

## A1

(a) Matmul: (...,M,K)@(...,K,N)->(...,M,N). Embedding: (B,T)+(V,d)->
(B,T,d). Softmax over -1: shape kept, rows sum to 1. Norm over last
axis: shape kept.
(b) Ladder. Rank: number of axes. (4,256,512)@(512,2048) -> (4,256,2048).
Right-to-left because the last two axes do the multiply, leading axes
batch. check_shapes: compare each axis, None is wildcard, raise naming
the op. einops: the contract is in the expression. Debug: unsqueeze the
(B,d) bias to (B,1,d). Critique: dynamic shapes need runtime asserts.
Transfer: it applied a per-head merge or projection dropping the head
axis, the contract is (B,h,T,d)->(B,T,d), likely a reshape or mean.
Answers graded on the contract statement, not the guess.

## A2

(a) Named axes multiply and sum when repeated, labels absent from the
output are summed out, the output lists kept labels in order.
(b) Ladder. One head scores: "td,sd->ts" on (T,d),(S,d). Output order
sets the result layout, wrong order transposes silently. attention_mix:
"bhij,bhjd->bhid". Transpose code: equivalent but hides the contract.
Debug: compare against @ on fixed data, the shapes match but values do
not. Critique: sizes must agree, the check is runtime. Transfer:
softmax over keys: the einsum is unchanged ("bhid,bhjd->bhij"), the op
is `softmax(scores, dim=-1)`.

## A3

(a) Stride: bytes between consecutive elements along an axis. View:
shared storage, new metadata. Contiguous: walk order matches storage.
(b) Ladder. (4,6) float32 transpose: shape (6,4), strides (4,24).
Transpose is free because only metadata changes. Contiguity check:
stride[i] == itemsize * product(shape[i+1:]). View: edits alias, copy:
independent. Debug: clone the activation before in-place ops, or avoid
in-place ops on saved tensors. Critique: padded/strided storage breaks
the simple formula. Transfer: place the contiguous call right after the
transpose chain, before the kernel that needs it, once.

## A4

(a) Attention input, QKV projections, attention weights (B,h,T,T), FFN
intermediates.
(b) Ladder. Saved tensor: kept for backward. Weights term: B*h*T*T
elements, here 4*8*256*256 = 2097152, times 2 bytes = 4.2 MB per layer.
T^2 because every query meets every key. Estimator: sum the terms.
Checkpointing: recompute instead of save. Debug: the ledger's
activation row versus the OOM size, cut batch or checkpoint. Critique:
fragmentation is unmodeled, keep margin. Transfer: the T^2 term
dominates, first mitigation is attention checkpointing or a
linear-attention variant (U04).

## A5

(a) SGD 12, Adam 16, mixed 16 bytes per param.
(b) Ladder. Master weights: the fp32 copy the optimizer updates.
Mixed by hand: 2+2+4+4+4 = 16. Mixed equals fp32 because the master and
moments stay fp32, only compute tensors shrink. Counter: dict of modes.
8-bit: less memory, quantization risk on moments. Debug: loss stalls
while gradients look healthy, check update magnitudes versus bf16
rounding. Critique: sharding (U08) changes the per-device math.
Transfer: (1) shard optimizer state (ZeRO), sacrifices nothing but adds
communication. (2) checkpoint activations, sacrifices ~33 percent
throughput. Or (3) smaller batch, sacrifices throughput.

## A6

(a) MNK multiplications plus MNK additions = 2MNK.
(b) Ladder. MAC: one multiply-add = 2 FLOP. QKV by hand: 3 * 2 * 4 * 256
* 512 * 512 = 1.61 GFLOP. SwiGLU FFN: gate, up, down = 3 maps.
MAC vs FLOP: factor 2, same information. Debug: add the 2*B*T^2*d
score term. Critique: masking/sparsity change the count. Transfer: d
2x and T 0.5x: d^2 terms go 4x, T^2 terms go 0.25x, total depends on the
mix, compute per term.

## A7

(a) Backward about 2x forward, a training step about 3x forward.
(b) Ladder. dout: gradient of loss wrt the op output. dx = dout W^T and
dW = x^T dout, each 2MNK. The 2x is a band because norms and activations
cost less. Counter: forward + 2x forward. Checkpointing: +1 forward for
recompute. Debug: the budget missed backward entirely, multiply the
forward estimate by 3. Critique: recompute breaks 2x. Transfer: half the
layers checkpointed: total = forward + backward + 0.5 * forward =
3.5x forward.

## A8

(a) FLOPs per byte moved, FLOP/byte.
(b) Ladder. Byte count: 2*(MK+KN+MN) for fp16. (1,512,512): 524288
FLOPs / 526336 bytes = 1.0. N/3 scaling for square: FLOPs grow as N^3,
bytes as N^2. Function: three lines. Roofline: plot the measured points.
Debug: count the output tensor once, not twice. Critique: state the
boundary (HBM vs SRAM). Transfer: intensity rises ~8x toward the
compute-bound side, the plan shifts from traffic reduction to FLOP
efficiency.

## A9

(a) MFU = achieved model FLOP/s divided by peak FLOP/s, using the honest
model FLOP count.
(b) Ladder. Cap = min(peak, intensity * bandwidth) = min(100, 50) = 50
TFLOP/s. Cap before peak because physics binds first. mfu: achieved,
cap, ratio. Counters: per-kernel truth. Debug: MFU > 1 means the FLOP
count is inflated (recompute double-counted). Critique: the count must
be the model count. Transfer: (1) low occupancy from small shapes,
(2) unfused elementwise ops adding traffic. Profile before guessing.

## A10

(a) Params, grads, optimizer state, activations, workspace/input.
(b) Ladder. Verdict: fits or not with margin. Toy total: 664.8 + 134.2
= 799.0 MB, margin 15 percent -> 918.9 MB. Margin covers fragmentation
and unmodeled buffers. Ledger: sum of C04+C05+workspace. Snapshot: the
framework's allocator report. Debug: add NCCL and loader rows, the
single-device ledger undercounts multi-GPU. Critique: one device only.
Transfer: cheapest is usually shrinking the microbatch (costs
throughput) or enabling activation checkpointing (costs ~33 percent
throughput), sharding costs communication but no throughput if the
network is fast.

## A11

(a) fp16: 2 bytes, 10 mantissa bits, max 65504. bf16: 2 bytes, 7
mantissa bits, fp32-range exponents.
(b) Ladder. Loss scaling: multiply the loss to keep gradients in fp16
range, unscale before the update. Halving: 268.4 -> 134.2 MB. Master
copy absorbs bf16 rounding of small updates. Overflow demo: accumulate
1e-5 x 1000 in fp16 (stalls) vs fp32 (0.01). bf16 vs fp16: range vs
precision. Debug: early layers stop learning, switch to bf16 or add
loss scaling. Critique: needs low-precision matmul hardware. Transfer:
fall back to fp32 training or fp16 with loss scaling, expect lower
throughput and higher memory.

## A12

(a) CPU proves correctness (shapes, values, invariants), it does not
prove performance (rates, speedups).
(b) Ladder. Correctness: exact numerics on seeds. Performance: measured
rates on target. Port order: numpy, verify, torch, measure. CPU-first:
cheap iteration. GPU-first: measures early but slowly. Debug: the
"optimization" changed cache behavior, not the bottleneck, re-profile
on target. Critique: rounding differs across devices, use tolerances.
Transfer: ask for the measurement device and the baseline, a CPU
speedup is a hypothesis, not a result.
