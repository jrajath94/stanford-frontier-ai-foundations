# U15 lab , adaptation arithmetic

Instructions: implement each task as a small numpy function, then run
`python3 u15_lab_run.py`. Your outputs must match `u15_lab_key.md`
exactly. No training runs, arithmetic and toy simulations.

## Task 1 , stages (C01)

1a. Classify runs by objective: pretrain/midtrain/post-train.

## Task 2 , context (C02)

2a. Wavelength 6.28e4 -> 3.14e6 (x50).

## Task 3 , templates (C03)

3a. Round-trip turns -> text -> turns.

## Task 4 , masks (C04)

4a. Mask [0,0,1,1] on the toy roles. 2/4 supervised.

## Task 5 , curation (C05)

5a. Funnel 1M -> 250k kept.

## Task 6 , tool masks (C06)

6a. 5-call trace: 5 mask segments, results masked 0.

## Task 7 , exposure (C07)

7a. 250k x 3 epochs = 750k exposures. 1% bad = 7,500.

## Task 8 , packing (C08)

8a. Pad waste 47.6% on the 512-seq toy.

## Task 9 , LoRA (C09)

9a. Pair 131072 (0.78%), q,v 32 layers: 8.39M vs 1074M.

## Task 10 , forgetting (C10)

10a. Mean old-task rise +0.15 nats.

## Task 11 , slices (C11)

11a. Instruction +0.10, base -0.02 (noise), context flat.

## Task 12 , checkpoints (C12)

12a. Full state 84.0 GB, weights-only 14.0 GB.
