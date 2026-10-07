# U10 lab , scaling-law arithmetic on synthetic toys

Instructions: implement each task as a small numpy function, then run
`python3 u10_lab_run.py`. Your outputs must match `u10_lab_key.md`
exactly. All data is synthetic with fixed seeds. No training runs.

## Task 1 , budgets (C01)

1a. Compute C=6ND for (70B, 1.4T): 5.88e23 FLOPs, D/N=20.0.
1b. Compute C for 7B at 20 tok/param: 5.88e21 FLOPs.

## Task 2 , loss floor (C02)

2a. Grid E over {1.5, 1.69, 1.9} on the synthetic set, pick the best:
1.69.

## Task 3 , power-law fit (C03)

3a. Fit log(L-E) vs log C, report a=0.340.

## Task 4 , isoflop curve (C04)

4a. Build one isoFLOP curve at C=1e22 (16 N values), report N* and
min loss.

## Task 5 , allocation (C05)

5a. Allocate C=5.88e23: N=7.00e10, D=1.40e12, D/N=20.0.

## Task 6 , hp transfer (C06)

6a. Transfer lr 3e-4 from width 256 to 1024: 7.5e-05.

## Task 7 , residuals (C07)

7a. Compute residual RMS: 0.0031, max |resid|: 0.0074.

## Task 8 , bootstrap (C08)

8a. 200 resamples, report the 95% CI for a: [0.336, 0.343].

## Task 9 , inference-optimal (C09)

9a. Compute toy losses A=2.290, B=2.646, gap 0.36 nats.

## Task 10 , preregistration (C10)

10a. Write the band [0.30, 0.38] before fitting, score: HELD at
0.340.

## Task 11 , overtraining (C11)

11a. 7B at 20 vs 200 tok/param: losses 3.003 and 2.646.

## Task 12 , transfer checklist (C12)

12a. Run form/noise/range checks, the range check FLAGS on the toy.
