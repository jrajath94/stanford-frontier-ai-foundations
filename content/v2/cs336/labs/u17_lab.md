# U17 lab , training-system time models

Instructions: implement each task as a small function, then run
`python3 u17_lab_run.py`. Your outputs must match `u17_lab_key.md`
exactly. Time models only, no cluster.

## Task 1 , modality budget (C01)

1a. 256 image + 512 text = 768 tokens (33.3% image).

## Task 2 , interleaving (C02)

2a. 200 + 256 + 300 = 756 tokens, one stream.

## Task 3 , pipes (C03)

3a. Data pipe 0.010 GB/s at the 2.5 Mtok/s toy.

## Task 4 , separation (C04)

4a. Broadcast every 10 updates. KL under 0.01.

## Task 5 , async (C05)

5a. Utilization 83.3%, backlog 1200/hr.

## Task 6 , templates (C06)

6a. v2 vs v3 hash mismatch fires before GPUs burn.

## Task 7 , safety (C07)

7a. Controls: filter, monitor, 1% sample, kill at -2pts.

## Task 8 , lineage (C08)

8a. Chain to artifact id ea98188a5dad105d.

## Task 9 , resumability (C09)

9a. Checkpoint 8.4 s, expected waste 500 steps at interval 1000.

## Task 10 , e2e (C10)

10a. 5/6 invariants pass, template fails -> reject.

## Task 11 , bottlenecks (C11)

11a. Backward 38.5% of 143 ms. 2x -> 116 ms (19.2%).

## Task 12 , scope (C12)

12a. Effects +0.10/+0.06/+0.02/-0.03: flip at 64x.
