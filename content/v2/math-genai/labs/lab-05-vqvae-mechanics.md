# Lab 05, VQ-VAE mechanics on the codebook toy

Unit: math-genai-U06. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-05.md. Test-mode: solve closed-book, then
check. The toy: codebook E = [[1,0],[0,1],[-1,0],[0,-1]],
z_e = [0.9, 0.2], identity decoder, beta = 0.25.
Ground truth: compute_run5a.py.

## Task 1, quantization by hand then in code

(a) By hand: compute the four squared distances for
z_e = [0.9, 0.2]. Name the winner and the margin.
(b) In code: implement quantize with row/col asserts
on the distance matrix. Reproduce k = 0 and the
distance list.
(c) Break it: pass z_e = [0.7, 0.7]. Record the tie
and state what argmin returns.
(d) Predict first: the winner for z_e = [-0.8, 0.1].
Measure it and judge the prediction.

## Task 2, the three losses: predict first, measure second

(a) Predict: recon, codebook, and commitment losses
on the toy. Write all three numbers before computing.
(b) Measure: implement vq_losses as in lesson C03.
Record the three numbers and the total.
(c) Swap the sg[.] placements (freeze e_k in the
codebook loss, freeze z_e in the commitment loss).
State in one sentence what now moves wrongly.
(d) Set beta = 0 and beta = 1.0. Record both totals.
State which term each beta choice endangers.

## Task 3, straight-through: predict first, measure second

(a) Predict: the true gradient and the STE gradient
of ||z_q - target||^2 at the toy, target = [1, 0.5].
Write both before computing.
(b) Measure: finite-difference the true gradient with
h = 1e-6. Implement the STE copy. Record both.
(c) Move z_e to [0.55, 0.45] (near the bisector).
Recompute both gradients. State what changed about
the STE's honesty.
(d) Write one sentence: what would you tell a
teammate who claims the STE is unbiased "because
the forward pass is exact"?

## Task 4, EMA and dead codes

(a) By hand: one EMA update from counts [10, 2, 0, 1]
with batch assigns [3, 2, 0, 1], gamma = 0.99. Give
the new counts.
(b) In code: implement ema_update with the dead-code
guard. Run it. Then remove the guard and record
exactly what happens to the dead code's mean.
(c) Compute usage over the 8 lesson encodings.
Record the vector and name the dead code.
(d) Revive it: reassign e_3 to a live encoding plus
noise 0.05 (seed 0). Recompute usage. Record the new
vector.

## Task 5, rate/distortion ablation

(a) By hand: compute distortion at K = 1 on the 8
encodings (single code at origin). Show that it
equals mean ||z_e||^2.
(b) In code: implement rate_distortion for K = 1, 2,
4. Reproduce (0, 0.91625), (1, 0.19125), (2,
0.06625).
(c) Controlled ablation: fix seed and data, vary
only the codebook seed (3 seeds). Record the
distortion spread at K = 4. State whether K or seed
dominates.
(d) Write one paragraph: your team proposes K = 64
"for quality". Use the diminishing-returns numbers
and the dead-code lesson to argue for a smaller K
with a measurement plan.
