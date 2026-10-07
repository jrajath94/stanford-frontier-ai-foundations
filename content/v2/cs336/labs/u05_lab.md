# U05 lab , training machinery checks

Instructions: implement each task as a small numpy function, then run
`python3 u05_lab_run.py`. Your outputs must match `u05_lab_key.md`
exactly. Limits: CPU numpy only, no torch on this box. RNG seed 0
where used.

## Task 1 , stable cross-entropy (C01)

1a. Implement `stable_ce(logits, target)` with the max subtraction.
1b. Show naive softmax+log gives nan on [[1000,1001,999]] (guard the
warnings with np.errstate).
1c. Hand-check: stable CE for [[2,1,0]], target 0, equals 0.4076.

## Task 2 , label shift (C02)

2a. Implement `shift(ids)`: return inputs, labels.
2b. Verify [5,6,7,8] gives inputs [5,6,7], labels [6,7,8].

## Task 3 , AdamW hand step (C03, C05)

3a. Implement `adamw_step(w, m, v, g, t, lr, lam)`.
3b. Check one step on w=[1.0,-0.5], g=[0.3,-0.2] with lr=1e-3,
lam=0.01, t=1: w becomes [0.99899,-0.498995].
3c. Assert the t=1 exactness: mhat == g, vhat == g^2.

## Task 4 , decay versus L2 (C04)

4a. Implement both update forms.
4b. Show they differ on w=[0.5], g=[0.1], lam=0.1.

## Task 6 , gradient clipping (C06)

6a. Implement `clip_grads(g, c)` with the global norm.
6b. Check norm 5.0 -> 1.0000, direction cosine 1.000000.
6c. Check the no-op when the norm is below c.

## Task 7 , schedules (C07)

7a. Implement `cosine_lr(t)` with warmup 100, total 1000, peak 3e-4,
floor 3e-5.
7b. Check lr(0)=0, lr(100)=3e-4, lr(1000)=3e-5.
7c. Implement `wsd_lr(t)` with a stable phase to step 800, check
lr(500)=3e-4.

## Task 8 , accumulation equivalence (C08)

8a. Show mean([0.2,0.4],[0.6,-0.2]) = [0.4,0.1].
8b. Show forgetting /a doubles the update.

## Task 9 , SGD+m and Lion hand steps (C09)

9a. Implement `sgd_step` and `lion_step`.
9b. On w=[1.0], g=[0.5], lr=1e-3: SGD+m -> 0.999500, Lion -> 0.999000.

## Task 10 , muP table (C10)

10a. Implement `mup_scales(width)`: std-init 1/sqrt(n), hidden lr
mult 128/n, output init std 1/n.
10b. Print the table for 128/512/2048.

## Task 11 , RNG restore (C11)

11a. Save the RNG state, draw 3 numbers, restore, draw again: exact
match.

## Task 12 , gradient check (C12)

12a. Implement `grad_check` with central differences, e=1e-5, fp64.
12b. Check a (4,3) linear layer: max relative error 1.88e-10.
