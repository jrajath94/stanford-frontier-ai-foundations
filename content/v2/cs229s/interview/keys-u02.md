# keys-u02.md: interview answer keys, U02

Date: 2026-10-06. Minimum sufficient explanation, strong
answer, red flags, rubric, remediation per item.

## B1

Minimum: registers, shared/L1, L2, HBM, with sizes ~256
KB/SM, ~100-200 KB/SM, tens of MB, tens of GB.
Strong: adds latency orders (~1, ~25, ~200, ~500
cycles).
Red flags: putting HBM above L2.
Rubric: 2 points. Remediation: C01.

## B2

Minimum: global = blockIdx.x*blockDim.x + threadIdx.x.
Blocks: ceil(100000/512) = 196. Idle: 352.
Strong: notes the guard for the tail.
Red flags: integer division without ceil.
Rubric: 1 formula, 2 numbers. Remediation: C02.

## B3

Minimum: fusion merges kernels so intermediates stay on
chip. It saves bytes moved, not FLOPs.
Strong: quantifies with the toy (5120 vs 2048).
Red flags: "fusion reduces FLOPs."
Rubric: 2 points. Remediation: C03.

## B4

Minimum: I = FLOPs/bytes, FLOP/byte. 64x64 fp16:
524288/24576 = 21.3.
Strong: contrasts with the matvec 0.97.
Red flags: unit confusion (FLOP/s).
Rubric: 2 points. Remediation: C04.

## B5

Minimum: min(pi, beta*I). Ridge = pi/beta = 150
FLOP/byte.
Strong: reads a bound off the plot.
Red flags: "ridge is where speed is highest."
Rubric: 2 points. Remediation: C05.

## B6

Minimum: strides (16, 2) bytes. Column read touches 4
far-apart lines, wasting 64x bytes.
Strong: names the 128-byte line.
Red flags: "same data, same speed."
Rubric: 2 points. Remediation: C06.

## B7

Minimum: reduced dtype, dims multiple of 8/16, aligned
memory.
Strong: gives the fallback consequence.
Red flags: "fp16 is enough alone."
Rubric: 2 points (3 items, 1 point each, cap 2... Award
all 3). Remediation: C10.

## B8

Minimum: fp16: fine precision, short range (65504 max).
bf16: fp32 range, coarse precision.
Strong: maps to the train vs inference split.
Red flags: "bf16 is strictly better."
Rubric: 2 points. Remediation: C11.

## Deep ladder 1

L1a. I = FLOPs / bytes moved. Needs the operation count
and the byte traffic.
L1b. FLOPs 524288, bytes 24576, I = 21.3.
L1c. Bytes/s needed = S/I <= beta gives S <= beta*I,
also S <= pi. Min of the two.
L1d. O(1) arithmetic. The triage costs nothing next to
profiling.
L1e. Batch 1: memory-bound, ceiling 2e12. Batch 512:
compute-bound, ceiling 3e14.
L1f. Low occupancy (tiny grid) or a non-roof stall
(bank conflicts, divergence), nameplate vs sustained
roof.
L1g. Assumes the kernel can sustain the roof. Misleads
on latency-bound tiny kernels where neither roof binds.
L1h. Batch sweep, precision halving, intensity check.
Confounder: low occupancy fakes a compute-bound read,
check achieved occupancy first.
Scoring: 1 point per rung.

## Deep ladder 2

L2a. Fused: one kernel, intermediates in registers/
shared. Unfused: each op writes to and reads from HBM.
L2b. Unfused 5120 bytes, fused 2048 bytes (toy).
L2c. Ratio = unfused passes / fused passes over the
intermediate, here 5/2 = 2.5x.
L2d. Fusion does not change FLOPs, only bytes.
L2e. 536 MB: fusion saves ~1 GB traffic per pass,
worthwhile. 2 KB: the saving is noise vs launch
overhead, pointless.
L2f. Check for graph breaks (data-dependent control
flow). Fix the breaks before trusting compile.
L2g. Assumes the intermediate fits on chip. At 4 GB it
cannot, the compiler keeps separate kernels.
L2h. Measure HBM traffic (profiler counters) eager vs
compiled. Expect ~2x drop matching the intermediate
size in bytes.
Scoring: 1 point per rung.

## A1

50 kernels: overhead 250 us, work 2000 us, total 2250
us, share 11.1 percent. 4 kernels: overhead 20 us,
total 2020 us, share ~1 percent.

## A2

1 GB: 500 us transfer + 0.5 us latency = 500.5 us,
latency share 0.1 percent. 1 KB: 0.0005 us + 0.5 us =
0.5005 us, latency share ~100 percent. Small kernels
are latency/overhead-bound, fuse them.

## D1

Bug: `n // block` floors: 1000//256 = 3 blocks = 768
threads, leaving 232 elements uncovered. Fix:
`math.ceil(n / block)` = 4 blocks. Invariant:
blocks*block >= n, with a bounds guard for idle
threads.

## S1

Tiling and fusion break first (no on-chip staging for
intermediates). Replacement: register blocking only
(tiny tiles), deeper software pipelining from L2, and
smaller kernels that fit the register file.

## S2

64x64 matmul: limited by parallelism (too little work
to fill fixed SMs) and launch overhead, the new
bottleneck is occupancy/latency. 7B decode: with
infinite bandwidth, limited by compute latency per
token (kernel launches, sequential dependency),
optimize with graphs and speculative decoding.

## R1

(1) Which shapes? Invalid: only large multiples of 8.
(2) Sustained or nameplate roof? Invalid: nameplate pi
never achieved. (3) Measured how (which counters)?
Invalid: FLOP count inflated. (4) Batch sizes? Invalid:
batch 1024 only. (5) Independent reproduction? Invalid:
no script or hardware named.
