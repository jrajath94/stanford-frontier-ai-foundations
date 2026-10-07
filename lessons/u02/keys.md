# keys.md, U02 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Registers (per thread, ~256 KB/SM), shared memory/L1 (per
block, ~100-200 KB/SM), L2 (tens of MB), HBM (tens of GB).
Order: registers, shared/L1, L2, HBM.

## E02

Transfer: 2^20 / 2e12 s = 0.5 us. Plus 0.5 us latency.
Total ~1.0 us.

## E03

Target shared memory/registers first: tile the working set
on chip (U04 C03) and raise occupancy so latency hides.

## E04

Tiling and fusion break first: every intermediate would
round-trip to HBM, and the bandwidth-bound regime would
widen to most kernels.

## E05

Hypothesis: effective copy bandwidth rises with size and
plateaus at the spec, sizes below ~1 MB are latency-bound.
Knee prediction: the size where transfer time equals
latency (~1 MB at these toy numbers).

## E06

global = blockIdx.x * blockDim.x + threadIdx.x.

## E07

Blocks: ceil(100000/512) = 196. Launched: 100352. Idle:
352.

## E08

The missing line is the bounds guard `if (i < N)`. Tail
threads wrote out of range or the tail was never covered.

## E09

4 SMs x 1 block x 1024 threads = 4096 elements resident at
once. The rest queues.

## E10

Hypothesis: time falls with block count until blocks cover
the SMs, then flattens. Plateau at the SM count, tiny
blocks add scheduling overhead before the plateau.

## E11

Fusion merges several kernels into one so intermediate
tensors stay on chip instead of round-tripping to HBM.

## E12

One write plus one reread of (8, 2048, 16384) fp16:
2 * 8*2048*16384*2 = 1073741824 bytes, 1.0 GiB saved per
layer per pass.

## E13

Check for graph breaks first: data-dependent Python
control flow that splits the compiled graph.

## E14

No. The 4 GB intermediate cannot stay on chip, so the
fused kernel cannot form. The compiler keeps separate
kernels (or the program must tile first).

## E15

Hypothesis: compiled MLP traffic drops ~2x versus eager,
and the drop in bytes matches the intermediate tensor
size. Report measured traffic both ways.

## E16

I = FLOPs / bytes moved, in FLOP per byte.

## E17

FLOPs: 2*128*256*512 = 33554432. Bytes: 2*(128*256 +
256*512 + 128*512) = 458752. I = 73.1 FLOP/byte.

## E18

Causes: (1) poor parallelism/occupancy (tiny grid cannot
fill the machine), (2) the roof values are nameplate, not
sustained, or another stall (bank conflicts, divergence).

## E19

Decode-step intensity rises with batch: each weight is
reused across B requests, so I ~ B (fp16 matvec ~1 per
request).

## E20

Hypothesis: measured FLOP/s rises with computed intensity
then plateaus at the roof. Report the plateau intensity
and compare to the ridge.

## E21

Attainable = min(pi, beta * I). Ridge I* = pi / beta,
where the bandwidth line meets the compute ceiling.

## E22

I=0.97: 1.94e12. I=21: 4.2e13. I=150: 3e14 (ridge).
I=2000: 3e14 (compute-bound).

## E23

Check the measurement first: FLOP count, byte count, or
timer is wrong. A point above the roof falsifies the
measurement, not the model.

## E24

New ridge: 3e14 / 4e12 = 75. Bound at I=21: 4e12*21 =
8.4e13, still memory-bound.

## E25

Hypothesis: all measured kernels sit on or below the
roof, none above. Report each kernel's (I, achieved)
point.

## E26

C-order (4, 8) fp16 strides in bytes: (16, 2). Row stride
16 bytes, element stride 2 bytes.

## E27

Column read touches 4 addresses, each pulling a 128-byte
line: 512 bytes moved for 8 bytes wanted (64x waste).

## E28

Fix: change the access to the fast axis (transpose the
tensor or restructure the loop) so threads in a warp
touch adjacent addresses.

## E29

With 16-byte lines, each column element pulls 16 bytes:
64 bytes moved for 8 wanted, 8x waste instead of 64x.

## E30

Hypothesis: each memory order wins on its fast axis by a
factor near the cache-line ratio. Report both ratios
(C-order row vs column, Fortran-order row vs column).

## E31

A vectorized load moves several elements (e.g. 4 floats,
16 bytes) in one instruction instead of one element per
instruction.

## E32

1M fp32: 1000000 scalar loads vs 250000 float4 loads.

## E33

Check 16-byte alignment of the addresses. Misaligned
wide loads split into multiple transactions and lose.

## E34

8-byte alignment allows float2 (8 bytes) as the widest
safe vector for fp32 pairs, float4 needs 16-byte
alignment.

## E35

Hypothesis: float4 beats scalar ~2x at large sizes and
~1x at tiny sizes where overhead dominates. Report the
crossover size.

## E36

Total = launches * overhead + total work.

## E37

Overhead: 50*5 = 250 us. Work: 50*20 = 1000 us. Total
1250 us. Overhead share: 20 percent.

## E38

Suspect: kernel launch overhead. Many small kernels at
small batch leave the GPU idle between launches.

## E39

With zero overhead, fine-grained kernels are free: more,
smaller kernels with no fusion pressure. Design would
favor clarity over kernel count.

## E40

Hypothesis: total time vs N is linear with slope equal
to launch overhead (~5 us toy). Report the fitted slope
and intercept.

## E41

A stream is an ordered queue of GPU work, work in one
stream runs in order, streams run independently.

## E42

Best: max(6, 10) = 10 ms. Saving: 16 - 10 = 6 ms.

## E43

Both kernels need the SMs, which are full. Streams share
the SMs, only the copy engine is a truly separate lane.
No SM headroom means no overlap.

## E44

Overlap of two compute kernels is still possible only
with SM headroom (one kernel's idle SMs serve the
other). Without a copy engine, copy/compute overlap is
impossible, kernel/kernel overlap needs spare SMs.

## E45

Hypothesis: two-stream time approaches max(compute,
copy), serial approaches the sum. Report both timings.

## E46

Contract: reduced-precision dtype (fp16/bf16/int8),
contracted dims as multiples of 8/16, aligned memory.

## E47

FLOPs: 2*64^3 = 524288. At 3e14: 1.7 us. At 2e13:
26 us.

## E48

First check: are m, n, k multiples of 8? If not, the
matmul may fall back to CUDA cores.

## E49

Pad to 8: waste = (8-3)*4096*4096*2 FLOPs extra per
matmul, about 1.7e8 FLOPs, versus ~15x slower CUDA-core
math. Pad.

## E50

Hypothesis: time drops in jumps at multiples of 8 where
tensor cores engage. Report the jump sizes at each
multiple.

## E51

fp16 has fine precision but short range (to 65504),
bf16 has fp32 range but coarse precision.

## E52

1e5 overflows in fp16 (max 65504). 1e-5 is fine in
fp16. In bf16, 1e5 is representable but coarse, 1e-5
loses precision (7 mantissa bits).

## E53

Suspects: (1) gradient overflow to inf in fp16 (range),
(2) a genuinely diverging update. Check max gradient
magnitude first.

## E54

Training breaks first in the optimizer: tiny updates
round to zero in int8, and gradient ranges overflow.
Master weights and updates need wider formats.

## E55

Hypothesis: fp16 without loss scaling diverges or
stalls, bf16 tracks fp32 final loss closely. Report all
three final losses.

## E56

Test A: double the batch, speedup means bandwidth-bound.
Test B: halve precision, speedup means bandwidth-bound.
Test C: compute intensity, I < I* means memory-bound.

## E57

0.5: memory. 100: memory (below 150). 300: compute.

## E58

Confounders: low occupancy (batch sweep confounded),
launch overhead at small work, or a non-roof stall
(bank conflicts).

## E59

Use the precision test (halve bytes without growing the
batch) or the roofline computation (intensity vs
ridge). Both avoid batch growth.

## E60

Hypothesis: at least 8 of 10 kernels agree across all
three tests, disagreements trace to occupancy or launch
overhead. Report the agreement table.
