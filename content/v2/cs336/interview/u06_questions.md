# U06 interview bank , questions

Closed-book. Keys in `u06_key.md`. Reference device: 312 TFLOP/s bf16
dense, 2.0 TB/s, 80 GB, 108 SMs (lesson assumptions). Quotas: 6
breadth, 2 deep ladders of 5, 2 analytical, 1 implementation/debug, 2
changed-constraint, 1 research-critique.

## Breadth (6)

B1. What is a warp, and why does its size matter?
B2. State the roofline formula and the knee for the reference device.
B3. What is MFU, and what does it exclude?
B4. How long does a ring all-reduce of B bytes take?
B5. What three rules make a GPU timing trustworthy?
B6. What causes clock throttling, and how do you detect it?

## Deep ladders (2 x 5)

L1. Roofline.
- L1.1 Define arithmetic intensity.
- L1.2 Toy: classify attention scores (1.0) and GEMM (170.7).
- L1.3 Derive the knee at 156 FLOP/byte.
- L1.4 Implement roofline, state the monotonicity check.
- L1.5 Compare roofline with simulation, debug the 50-percent case,
  critique the two-limit assumption, propose the U02 classification.

L2. MFU.
- L2.1 Define the numerator and denominator.
- L2.2 Toy: 7B params, 1800 tok/s. Compute MFU.
- L2.3 Derive why rematerialization raises HFU above MFU.
- L2.4 Implement mfu, state the <=100 percent check.
- L2.5 Compare MFU with HFU, debug the restart case, critique the
  FLOP-model assumption, propose the remat comparison.

## Analytical exercises (2)

E1. A GEMM has M=64, N=4096, K=4096, fp16. (a) Compute its intensity.
(b) Classify it on the reference device. (c) The vendor peak is 312
TFLOP/s. What throughput do you actually target, and why?
E2. A 70B model (140 GB fp16) trains on 8 nodes with 200 GB/s links.
(a) Estimate one gradient all-reduce. (b) The step is 10 s. What
fraction is communication? (c) Name two mitigations.

## Implementation/debug task (1)

D1. This timing function reports that every kernel takes ~5 us. Find
the bug and fix it.

```
def time_bad(f, iters=100):
    ts = []
    for _ in range(iters):
        t0 = time.time()
        f()          # launches a GPU kernel
        ts.append(time.time() - t0)
    return min(ts)
```

## Changed-constraint scenarios (2)

S1. Context must double on fixed 80 GB HBM with GQA-8. Name two
levers, the bytes math for each, and the quality risk.
S2. The cluster throttles every night at 2am. Your tuning
experiments run overnight. What changes in the methodology?

## Research critique (1)

R1. A paper reports "90 percent of peak" on a custom kernel but gives
no shapes, no dtype, no roofline, and no variance. List three gaps
and the measurement for each.
