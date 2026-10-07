# U08 lab , distributed training math

Instructions: implement each task as a small numpy function, then run
`python3 u08_lab_run.py`. Your outputs must match `u08_lab_key.md`
exactly. Simulations only, no cluster on this box.

## Task 1 , DP averaging (C01)

1a. Simulate 4 shards, check the mean equals the big-batch grad.
1b. Show the missing /N scales the update by 4.

## Task 2 , collectives (C02)

2a. Implement `ring_time(bytes, bw, N)`.
2b. Check 14 GB over 8 at 200 GB/s: 0.122 s, at N=1024: 0.140 s.

## Task 3 , ZeRO table (C03)

3a. Implement `zero_bytes(stage, N)`.
3b. Check 84.0/35.0/22.8/10.5 GB/GPU for stages 0-3 at N=8.

## Task 4 , bucketing (C04)

4a. Compute sequential vs overlapped for 8 layers (12 ms compute,
3 ms comm): 96 vs 36 ms.
4b. Compute exposed comm for (20,12): 8.0 ms.

## Task 5 , loss scaling (C05)

5a. Check 6e-8 * 1024 = 6.14e-05 is representable, unscale recovers
the value.
5b. State the overflow rule: skip, do not step.

## Task 6 , FSDP cycle (C06)

6a. Simulate gather-compute-release on a 1M-param toy over 4 ranks:
gathered 2.0 MB, resident 0.5 MB.

## Task 7 , checkpoints (C07)

7a. State both layouts for 84 GB over 8 ranks.
7b. Implement reshard 8->4 and check the round-trip.

## Task 8 , overlap (C08)

8a. Implement `exposed(comm, compute)`.
8b. Check (3,12)->0.0, (20,12)->8.0 ms.

## Task 9 , efficiency (C09)

9a. Compute efficiency for T1=100, T8=14: 89 percent.
9b. Model one straggler at 20 s: the step takes 20 s.

## Task 10 , fault model (C10)

10a. Compute expected lost work for k=100, 10 s steps: 500 s.

## Task 11 , data sharding (C11)

11a. Implement `shard_indices` for 100 items over 4 ranks.
11b. Check disjointness and coverage.

## Task 12 , invariants (C12)

12a. Implement the four invariant checks.
12b. Check all pass, the missing-/N bug fails invariant 1.
