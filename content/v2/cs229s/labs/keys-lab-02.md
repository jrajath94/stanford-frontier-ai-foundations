# keys-lab-02.md: lab 02 answer keys

Date: 2026-10-06. Observed outputs from
`python3 verify_lab02.py` on this machine. All 31 checks
passed.

## Observed output

```
task1: latency/bandwidth crossover near 1048576 bytes
task2: grid math ok
task3: fusion saves 2.5x traffic on toy
task4: I matmul=21.33, matvec=0.97, real=1024
task5: roofline points placed, ridge continuous
task6: strides ok
task7: row sum 2.73 ms, col sum 1.41 ms (ratio 0.52, numpy may cache, direction only)
task8: 1M vs 250k loads
task9: overhead math ok
task10: overlap = max
task11: shape rules ok
task12: fp16 overflows at 1e5, keeps 1e-5
ALL 31 CHECKS PASSED
```

## Key numbers

- Latency/bandwidth crossover: 2^20 bytes (1 MiB) at the
  toy 2 TB/s + 0.5 us spec.
- Intensity: matmul 21.33, matvec 0.97, real-scale
  (2048,4096)x(4096,4096) 1024 FLOP/byte.
- Roofline toy: ridge at 150 FLOP/byte, decode point
  (0.97) ceiling 1.94e12, toy matmul (21.33) ceiling
  4.27e13, real matmul compute-bound.
- Fusion: 5120 vs 2048 bytes on the toy (2.5x).
- Launch overhead: 1000 small kernels 6000 us vs 1
  fused 1005 us, 50-kernel layer overhead share 20%.
- fp16: 100000.0 overflows to inf (RuntimeWarning is
  expected), 1e-5 stays nonzero.

## Notes

Task 7 is direction-only: numpy's internal blocking can
make column sums faster than row sums on this CPU. The
lesson claim is about GPU cache-line waste, not numpy
timing. Rerun with `python3 verify_lab02.py`.
