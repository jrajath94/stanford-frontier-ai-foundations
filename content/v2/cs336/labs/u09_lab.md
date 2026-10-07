# U09 lab , parallelism planning math

Instructions: implement each task as a small numpy function, then run
`python3 u09_lab_run.py`. Your outputs must match `u09_lab_key.md`
exactly. Simulations only, no cluster on this box.

## Task 1 , TP split (C01)

1a. Simulate a column split of a (4,8)x(8,6) multiply over 2 ranks,
check the concatenated parts equal the unsharded product.
1b. Compute the TP all-reduce per layer: 67.1 MB.

## Task 2 , pipeline layout (C02)

2a. Compute layers/GPU for L=32, p=4: 8.
2b. Compute total stage-times for m=32: 35.

## Task 3 , bubble (C03)

3a. Implement `bubble(p, m)`.
3b. Check 42.9/8.6/46.7 percent and the p=1 edge.

## Task 4 , 1F1B (C04)

4a. State the bubble and in-flight count for p=4, m=32: 9.4
percent, ~4.

## Task 5 , hybrid triples (C05)

5a. Verify (8,2,4), (4,4,4), (2,8,4) cover 64 GPUs with tp<=8.

## Task 6 , comm table (C06)

6a. Compute TP 2.1 GB/step (64 msgs) and DP 14 GB/step (1 msg).

## Task 7 , balance (C07)

7a. Compute max/mean for stage times [10,8,8,9]: 1.14.
7b. State the cost-based fix.

## Task 8 , checkpointing (C08)

8a. State the memory tradeoff: 100 vs 32, compute +33 percent.

## Task 9 , inference (C09)

9a. Compute the toy latencies: TP=4 2.5 ms, PP=4 16.0 ms.

## Task 10 , hybrid bytes (C10)

10a. Compute (8,2,4): params 0.875, optimizer 14.0 GB/GPU.
10b. Compute (4,4,4): same.

## Task 11 , failure signatures (C11)

11a. State the three signatures.

## Task 12 , plan gate (C12)

12a. Implement `check_plan` with the five invariants.
12b. Check (8,2,4) passes and (16,2,2) fails tp<=node.
