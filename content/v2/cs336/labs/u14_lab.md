# U14 lab , filtering and dedup toys

Instructions: implement each task as a small numpy function, then run
`python3 u14_lab_run.py`. Your outputs must match `u14_lab_key.md`
exactly. Synthetic document sets only.

## Task 1 , quality threshold (C01)

1a. At t=0.5: keep good 0.94, keep bad 0.06.

## Task 2 , safety posture (C02)

2a. State the recall-first posture and the >0.99 probe target.

## Task 3 , exact dedup (C03)

3a. 10,000 docs -> 5,840 unique, dup rate 41.6%.

## Task 4 , Bloom k (C04)

4a. Sweep k in {3,6,10}, best k=6, FPR 0.0216.

## Task 5 , LSH (C05)

5a. b=20, r=5: detection prob at s=0.8 is 1.000.

## Task 6 , train-test dedup (C06)

6a. State the 13-gram index and the 80% drop rule.

## Task 7 , reweighting (C07)

7a. Multiplier = tgt/src, code 1.67.

## Task 8 , mixture search (C08)

8a. Grid argmin: w=1.0, L=2.500.

## Task 9 , synthetic filter (C09)

9a. 1M generated, 80% pass, 800k at 10% mix weight.

## Task 10 , bias audit (C10)

10a. Audit keep-rate per domain at fixed t.

## Task 11 , ablations (C11)

11a. Deltas: dedup +0.06, filter +0.03, reweight +0.01.

## Task 12 , repetition (C12)

12a. Recall at 1/2/5/10 reps: 0.18/0.34/0.61/0.78.
