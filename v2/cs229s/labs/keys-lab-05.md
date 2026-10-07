# keys-lab-05.md: lab 05 answer keys

Date: 2026-10-06. Observed outputs from
`python3 verify_lab05.py` on this machine. All 23 checks
passed.

## Observed output

```
task1: s=0.1333 z=8, xq(0.5)=12, zero exact
task2: max in-range err=0.0667 <= s/2=0.0667
task3: per-tensor MSE=7.7049, per-channel MSE=7.565803, quiet-channel MSE 0.2795 -> 0.001283
task4: dynamic s=0.0098 z=51
task5: min/max s=3.9255, percentile s=0.00783
task6: 2:4 validation ok
task7: magnitude pruning ok
task8: unstructured 95us, 2:4 55us vs dense 100us
task9: butterfly n=8 full mixing, params=48, n=1024 params=20480
task10: Monarch block-diagonal structure ok
task11: error bound 0.02 / 0.025
task12: ship gates ok
ALL 23 CHECKS PASSED
```

## Key numbers

- (s, z) toy: s=0.1333, z=8, xq(0.5)=12, zero maps
  exactly to zero.
- In-range error bounded by s/2 = 0.0667 on 10000
  random values.
- Per-channel vs per-tensor: quiet-channel MSE
  0.2795 -> 0.001283 (218x). Total MSE is dominated
  by the loud channel in both, the honest comparison
  is per-row.
- Calibration: min/max s=3.9255 vs percentile
  s=0.00783 (501x) with one outlier at 1000.
- 2:4 validation, magnitude pruning mask, and
  realized-speed arithmetic (95 us unstructured, 55
  us 2:4 vs 100 us dense) all check.
- Butterfly: full mixing at n=8, 48 params, 20480
  params at n=1024.
- Error bound: 0.02 (lips 1,1), 0.025 (lips 1,1.5).

## Notes

The task-3 lesson claim was corrected during the run:
per-tensor total MSE is dominated by the loud
channel, so the reported win is the quiet channel's
218x, not a total-MSE ratio. Rerun with
`python3 verify_lab05.py`.
