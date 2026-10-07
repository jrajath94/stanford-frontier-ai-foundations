# keys-lab-04.md: lab 04 answer keys

Date: 2026-10-06. Observed outputs from
`python3 verify_lab04.py` on this machine. All 24 checks
passed.

## Observed output

```
task1: warp mapping ok
task2: stride 1 -> 1 txn, stride 32 -> 32 txn
task3: tiled reads drop as n/tile
task4: barrier contract holds
task5: occupancy 25/50/100%, limiter={'threads': 8, 'shared': 2, 'regs': 8}
task6: tree reduction ok
task7: traffic 134MB/head at T=4096, 4x at 8192
task8: sparse/lowrank 16x less work at T=4096,w=256
task9: online softmax matches ref, extreme err=0.00e+00
task10: flash toy max err=1.39e-16
task11: recomputed dV matches, fd err=1.08e-11
ALL 24 CHECKS PASSED
```

## Key numbers

- Warp of thread 700: (21, 28).
- Coalescing: stride 1 -> 1 transaction, stride 32 ->
  32 transactions (fp32, 128-byte lines).
- Tiling: A-matrix HBM reads drop as n/tile (16x at
  n=1024, tile 64).
- Occupancy toy: 25% at 48 KB/block (shared-limited,
  2 blocks), 50% at 24 KB, 100% at 12 KB.
- Attention traffic: 134 MB/head at T=4096 (4 HBM
  passes, fp16), 4x at T=8192.
- Online softmax: matches two-pass reference exactly
  on the toy, 0.00e+00 relative error on the extreme
  stress (values 1000 apart).
- FlashAttention toy: max error 1.39e-16 vs naive
  attention (T=8, d=4, 2x2 blocks).
- Backward: recomputed dV matches reference,
  finite-difference error 1.08e-11.

## Notes

Task 4 uses OS threads as an analogy for the barrier
contract, not real GPU barriers. Rerun with
`python3 verify_lab04.py`.
