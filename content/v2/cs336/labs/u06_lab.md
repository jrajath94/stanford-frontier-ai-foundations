# U06 lab , profiling arithmetic

Instructions: implement each task as a small numpy function, then run
`python3 u06_lab_run.py`. Your outputs must match `u06_lab_key.md`
exactly. Reference device assumptions are stated in the lesson, no
GPU on this box.

## Task 1 , thread mapping (C01)

1a. Compute active threads for 256 threads/block, 8 blocks/SM,
108 SMs: 221,184.
1b. Compute idle SMs for a 10-block launch: 98.

## Task 2 , KV fit (C02)

2a. Implement `kv_fit(g, dh, L)`: 80e9/(2*g*dh*2*L).
2b. Check GQA-8 (dh=64, L=32): 1,220,703 tokens, MQA: 8x.

## Task 3 , roofline (C03)

3a. Implement `roofline(intensity)`.
3b. Find the knee: 156 FLOP/byte.
3c. Classify scores (1.0) and GEMM (170.7).

## Task 4 , occupancy calculator (C04)

4a. Implement `resident_blocks(regs, smem)` with the three limits.
4b. Check (32 regs, 48KB) -> 2, (32 regs, 8KB) -> 8.

## Task 5 , timing pattern (C05)

5a. Given runs [12.0, 3.1, 3.0, 3.2, 3.1], exclude warmup and take the
median: 3.1 ms.

## Task 6 , GEMM sweep (C06)

6a. Implement `gemm_intensity(M, N, K, bytes)`.
6b. Check M=1 -> 1.00 (memory-bound), M=4096 -> 1365 (compute-bound).

## Task 7 , MFU and HFU (C07)

7a. Compute MFU for 7B params at 1800 tok/s: 8.1 percent.
7b. Compute HFU with 30 percent rematerialization: 10.5 percent.

## Task 8 , collectives (C08)

8a. Implement `allreduce_time(bytes, bw)`.
8b. Check 7B over 600 GB/s -> 0.02 s, over 200 GB/s -> 0.07 s.
8c. Model 70B over the slow tier: 1.40 s.

## Task 10 , formats (C10)

10a. Tabulate bf16 312 and fp8 624 TFLOP/s dense.
10b. Model the M=1 intensity doubling under fp8.

## Task 11 , throttle detection (C11)

11a. Compute the throughput factor for 1.4 -> 1.0 GHz: 0.71 (29
percent drop).
11b. State the detection rule and apply it to an 8 percent drift.

## Task 12 , invariants (C12)

12a. Implement `check_invariants(throughput, roof, repro)`.
12b. Check the sane profile passes and the 120-percent-roofline
profile fails.
