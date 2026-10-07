# keys-lab-06.md: observed lab outputs

Date: 2026-10-06. `python3 verify_lab06.py` ran on this
machine. 31 checks passed.

- task1: b=-0.0748/decade, L(1e11)=1.70 (script prints
  -0.0749 from unrounded log10 inputs, same value within
  the 1e-3 check tolerance)
- task2: deltas=[0.09, 0.07, 0.03]
- task3: metric jumps 0->1, skill moves 0.15
- task4: loss=0.3583, ppl=1.4309
- task5: P(A wins)=0.5987, obj=0.75
- task6: margins 0.12 / 0.18
- task7: merge identity ok, d=k=4096 r=16 ratio=128x
- task8: 2.0s / 2.8s / 2.0s step times
- task9: yield 0.68 -> 340000 toks. Strict 212500
- task10: 3 bins, efficiency 0.781
- task11: 8192 tok/s, $0.81/Mtok. Packed 12779
  (8192*1.56=12779.52. The script prints 12780 with :.0f
  formatting. Lesson, keys, and figure use 12779.)
- task12: mean=2.31 std=0.0158, recipe B ties
- ALL 31 CHECKS PASSED
