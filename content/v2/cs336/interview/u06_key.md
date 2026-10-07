# U06 interview key

## B1-B6

B1. 32 threads executing in lockstep. Divergence serializes the warp,
block sizes should be multiples of 32.
B2. min(peak, intensity*bandwidth), knee 156 FLOP/byte.
B3. Achieved model FLOPs / peak. Excludes rematerialization and
communication (HFU includes them).
B4. Approximately 2*B/bandwidth.
B5. Device sync, warmup, median of many runs.
B6. Clock drops under power/thermal caps. Detect with a fixed
microbenchmark baseline before and after, flag >5 percent drift.

## L1

L1.1 FLOP per byte moved.
L1.2 Scores: 2.0 TFLOP/s, memory-bound. GEMM: 312 TFLOP/s,
compute-bound.
L1.3 Knee = 312e12/2.0e12 = 156.
L1.4 Three lines, check monotonic below the knee and flat above.
L1.5 Simulation is accurate but expensive. Debug: look for occupancy
or instruction overhead. Critique: only two limits. Experiment:
classify all U02 intensities.

## L2

L2.1 Numerator: 2*params*tok/s. Denominator: device peak.
L2.2 2*7e9*1800/312e12 = 8.1 percent.
L2.3 Remat adds FLOPs the model needs twice, HFU counts them, MFU
does not.
L2.4 Three lines, assert result <= 1.0.
L2.5 Debug: restarts waste wall clock outside the compute phase.
Critique: 2*params/token is a model, not a count. Experiment: sweep
remat fraction, plot MFU vs HFU.

## E1

(a) 2*64*4096*4096/((64*4096+4096*4096+64*4096)*2) = 62.1 FLOP/byte.
(b) Below the 156 knee: memory-bound. (c) Target min(312, 62.1*2.0)
= 124 TFLOP/s: the roofline, not the vendor peak. Rubric: (a) 2 pts,
(b) 1 pt, (c) 1 pt. Red flag: "312, it is a GEMM".

## E2

(a) 2*140e9/200e9 = 1.4 s. (b) 14 percent of the step. (c) Overlap
with backward compute, larger gradient buckets / fewer syncs,
quantized collectives. Red flag: "buy faster links" as the first
answer.

## D1

Bug: no device synchronization. `f()` returns after launching, the
timer measures launch overhead (~5 us), not the kernel. Also no
warmup and min instead of median. Fix: warm up, synchronize before
and after each timed region, take the median. Rubric: find 2 pts,
fix 1 pt, explain the 5 us 1 pt.

## S1

Lever 1: MQA (g=8->1): 65,536 -> 8,192 bytes/token: 8x tokens,
risk: head specialization. Lever 2: KV quantization to fp8/int8:
halves bytes, risk: quality loss on long context. State the bytes
math for each. Red flag: "increase batch size".

## S2

Pin the methodology to the invariant checks: baseline microbenchmark
every run, discard throttled windows, compare only within stable
clock periods, and report clocks alongside timings. Tuning against a
moving target is worse than no tuning.

## R1

Gaps: (1) no shapes/dtype: the roofline is undefined, need them.
(2) no variance: one number, need repeats. (3) "of peak" without
the peak's conditions: need the vendor claim's conditions and the
achieved-vs-roofline gap.
