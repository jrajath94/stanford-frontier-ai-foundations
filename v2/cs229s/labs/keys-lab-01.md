# keys-lab-01.md: lab 01 answer keys

Date: 2026-10-06. Observed outputs from
`python3 verify_lab01.py` on this machine. All 25 checks
passed.

## Observed output

```
task1: rnn steps grow with T, attention path stays 1
task2: shapes ok, softmax rows sum to 1
task3: 6P rule holds, manual dW matches finite differences
task4: toy=1664, real-scale=6573522944 (~6.57B)
task5: T doubling -> resid 2x, scores 4x, real resid=4.0 GiB
task6: delta per token ok, T=2048 cache=1.00 GiB
task7: equality at T=48, strict crossover at T=49 for n=8
task8: scores 4x, cache 2x on T doubling
task9: A ppl=12.18 acc=0.75, B ppl=14.88 acc=1.0
ALL 25 CHECKS PASSED
```

## Key numbers

- Toy parameter count: 1664. Real-scale: 6573522944.
- Real-scale residual activations: 4.0 GiB (fp16,
  B=8, T=2048, n=4096, L=32).
- KV cache at T=2048: 1.00 GiB. Delta per appended
  token: `2Ln` numbers.
- MLP/attention FLOP equality at T=48 for n=8 (strict
  past it).
- Perplexity/accuracy disagreement: lower perplexity
  (12.18) with lower accuracy (0.75).

## Notes

Task 1 checks are construction checks (path length by
definition). Task 3 includes a finite-difference gradient
check with tolerance 1e-6. No GPU was used. Rerun with
`python3 verify_lab01.py`.
