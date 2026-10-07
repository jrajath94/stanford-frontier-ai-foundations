# U11 lab , inference arithmetic and time models

Instructions: implement each task as a small numpy function, then run
`python3 u11_lab_run.py`. Your outputs must match `u11_lab_key.md`
exactly. Time models only, no GPU on this box.

## Task 1 , phases (C01)

1a. State decode AI 1.0 FLOP/byte, prefill(2048) ~2048 FLOP/byte.
1b. Weight bytes per decode token for 7B bf16: 14.0 GB.

## Task 2 , KV bytes (C02)

2a. Per-token bytes: 131072 (128.0 KB).
2b. b=16, T=4096: 8.59 GB.

## Task 3 , sharing (C03)

3a. 10 requests x 2000 shared tokens: unshared 2.62 GB, shared 0.26
GB, saving 2.36 GB.

## Task 4 , continuous batching (C04)

4a. Lengths [100,400,250,800], 2 slots: static 800 steps,
continuous 775 slot-steps.

## Task 5 , paged cache (C05)

5a. Lengths [512,...,1536]: contiguous waste 50.0%, paged 0.0%.

## Task 6 , speculative (C06)

6a. alpha=0.7, k=3: E=2.53, speedup 1.95x.

## Task 7 , exactness (C07)

7a. p=[0.7,0.3], q=[0.5,0.5]: accept probs [1.0, 0.6].

## Task 8 , sampling (C08)

8a. Logits [3,2,1,0]: temp=1.0 top-1 0.644, top-p=0.9 keeps 3.

## Task 9 , prefix reuse (C09)

9a. 1500-token history + 100 new: prefill 1600 -> 100
token-equiv, saves 192 MB.

## Task 10 , quantization (C10)

10a. 7B footprints: fp16 14.0 GB, int8 7.0 GB, int4 3.50 GB.

## Task 11 , tails (C11)

11a. TTFT toy: mean 120 ms, p99 533 ms.

## Task 12 , sizing (C12)

12a. 10 RPS, 2 slots/GPU: 5 GPUs at mean load, plus headroom.
