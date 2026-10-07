# U15 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u15_lab_run.py`, outputs
must match exactly.

```
T1 stages: pretrain LM / midtrain LM-new-mix / post-train SFT-RL
T2 wavelength 6.28e4 -> 3.14e6 (x50)
T3 template round-trip: turns -> text -> turns
T4 mask [0, 0, 1, 1], supervised 2/4
T5 funnel 1M -> 250k kept
T6 mask 1 on call JSON, 0 on results. 5-call trace = 5 segments
T7 exposure 250k*3=750k. 1% bad = 7500 bad lessons
T8 pad waste 47.6%
T9 LoRA pair 131072 (0.78%), q,v 32L: 8.39M vs 1074M
T10 mean forgetting +0.15 nats
T11 slices: instruction +0.10, base -0.02 (noise), context flat
T12 full state 84.0 GB, weights-only 14.0 GB
```
