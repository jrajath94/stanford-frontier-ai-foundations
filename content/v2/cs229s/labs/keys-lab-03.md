# keys-lab-03.md: lab 03 answer keys

Date: 2026-10-06. Observed outputs from
`python3 verify_lab03.py` on this machine. All 27 checks
passed.

## Observed output

```
task1: toy=333312, real=4.52e+22, attn share=0.071
task2: decode 100 tokens=364800, attn add-on ratio=0.096
task3: ratios ok, B=8/T=8192 cache=32.0 GiB
task4: prefill(1) == decode step
task5: prefill compute-bound, decode(B=1) memory-bound
task6: B=1 lat=20ms thr=50/s, B=8 lat=27ms thr=296/s
task7: E[k]=3.689, speedup=2.95x, bad case=0.41x
task8: output freq=[0.199 0.501 0.3  ], target p=[0.2 0.5 0.3]
task9: best gamma=8 at 3.09x
ALL 27 CHECKS PASSED
```

## Key numbers

- Training FLOPs toy: 333312 (dense 309504, attention
  23808). Real scale: 4.52e22, attention share 7.1%.
- Decode 100 tokens (toy): 364800 FLOPs, attention
  add-on 9.6% of 2P.
- KV cache: 320 bytes toy, 32.0 GiB at B=8, T=8192,
  L=32, n=4096, fp16.
- Prefill(T=1) == one decode step: identity holds.
- Batch toy: B=8 gives 296 tok/s at 27 ms/token vs 50
  tok/s at 20 ms/token.
- Speculative: E[k] = 3.689, speedup 2.95x at
  (a=0.8, gamma=5, c=0.05), bad case 0.41x at
  (a=0.3, gamma=5, c=0.5).
- Exactness: 60000-round histogram [0.199, 0.501,
  0.300] matches p = [0.2, 0.5, 0.3] within 0.01.
- Speedup surface peaks at gamma=8 (3.09x).

## Notes

The histogram test is the empirical proof of C09/C10
exactness on the toy. Rerun with
`python3 verify_lab03.py`.
