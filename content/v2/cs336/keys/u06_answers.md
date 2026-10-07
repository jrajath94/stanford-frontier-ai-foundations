# U06 answer key

Reference device (lesson assumptions, not measurements): peak bf16
dense 312 TFLOP/s, HBM 2.0 TB/s, 80 GB, 108 SMs. All numbers from
`../visuals/compute_u06.py` or the lab run (executed 2026-10-06).
Claim class: REQUESTED-BRANCH.

## R1 (remediation)

Switch at intensity = peak/bandwidth = 312e12/2.0e12 = 156 FLOP/byte.

## A1

(a) SM -> thread block -> warp (32 threads, lockstep). (b) 256*8*108
= 221,184 threads, 2.0K per SM, 256 = 8 warps. (c) 10 blocks occupy at
most 10 SMs: 98 idle.

## A2

(a) Registers (KB/thread, working set), shared memory (tens of
KB/block, scratch), HBM (80 GB, model+cache). (b) 65,536 bytes/token,
80e9/65536 = 1,220,703 tokens. (c) Halve the KV bytes (MQA: 8x
tokens) or quantize the cache.

## A3

(a) attainable = min(peak, intensity*bandwidth), knee 156 FLOP/byte.
(b) Scores: min(312, 2.0) = 2.0 TFLOP/s, memory-bound. GEMM: 312,
compute-bound. (c) Bandwidth doubling helps only below-knee kernels:
scores speed up 2x, GEMM unchanged.

## A4

(a) Warp switching covers load latency, occupancy = resident/max
warps. (b) Toy: 32 regs + 48KB smem -> 2 blocks/SM, 8KB smem -> 8
blocks/SM. (c) Raise the arithmetic intensity (tiling, fusion) or
reduce traffic, occupancy is already sufficient.

## A5

(a) Sync, warm up, median of N>=30. (b) Median of [3.1,3.0,3.2,3.1]
= 3.1 ms (12.0 excluded as warmup). (c) Not necessarily: need the
spread, if the 0.2 ms is within run-to-run noise, rerun with more
samples.

## A6

(a) Intensity = 2MNK/((MN+NK+MK)*bytes). (b) M=1: 1.00 FLOP/byte,
memory-bound, M=4096: 1365, compute-bound. (c) Batch 64: M=64,
intensity ~= 64x the M=1 case = 64 FLOP/byte, still below the 156
knee: memory-bound, but 64x better.

## A7

(a) MFU = 2*P*tok_s/peak. (b) 2*7e9*1800/312e12 = 8.1 percent. (c)
Suspects: tiny batch (memory-bound), communication dominating, bad
kernel shapes, then check throttling.

## A8

(a) t ~= 2*bytes/bandwidth. (b) 2*7e9/600e9 = 0.02 s, 2*7e9/200e9 =
0.07 s. (c) Fixes: overlap communication with compute, reduce bytes
(quantized/fused collectives), topology-aware placement.

## A9

(a) Quote claim with conditions, reproduce or state deviation,
measure per the timing rules, report achieved vs claimed with the gap
explained. (b) Conditions: dense fp8 GEMM, large shapes. (c) Ask for:
the exact workload, shapes, dtypes, measurement method, and variance.

## A10

(a) bf16 312, fp8 624 dense TFLOP/s, fp8 halves traffic. (b) M=1
intensity doubles to 2.0 FLOP/byte: still memory-bound, ~2x faster.
(c) Fallback: bf16 with kernel tuning, int8 for inference, wait for
hardware.

## A11

(a) Clocks drop under power/thermal caps. (b) 1.0/1.4 = 0.71: 29
percent hit. (c) Stop tuning, check clocks/temperature, fix cooling
or power cap, rerun the baseline.

## A12

(a) throughput <= roofline, counts within 2x of analytic, 10 percent
reproducibility. (b) The violation traces to unsynced timing. (c)
Re-measure with sync and warmup before any other conclusion.
