# Answer keys, lab 05 (VQ-VAE mechanics)

Date: 2026-10-06. Ground truth: compute_run5a.py.

## Task 1

(a) [0.05, 1.45, 3.65, 2.25]. Winner e_1, margin
1.40.
(b) quantize returns k = 0, zq = [1, 0], d2 as
above.
(c) d^2 = [0.58, 0.58, 3.38, 3.38]. Tie between e_1
and e_2. argmin returns 0 (first minimum).
(d) Prediction: e_3 = [-1, 0]. d^2 = [3.25, 1.45,
0.05, 2.65]. Winner index 2. The prediction holds.

## Task 2

(a) Recon 0.05, codebook 0.05, commitment 0.0125.
(b) vq_losses returns (0.05, 0.05, 0.0125, 0).
Total 0.1125.
(c) The codebook loss would move z_e (wrong: the
codebook should move) and the commitment loss would
move e_k (wrong: the encoder should move). Each
loss trains the wrong party.
(d) Beta = 0: total 0.10. The encoder drifts. Quantization error unpriced. Beta = 1.0: total
0.15. The encoder is pinned hard to possibly stale
codes.

## Task 3

(a) True [0, 0]. STE [0, -1].
(b) Finite differences give [0, 0] exactly (winner
unchanged). The STE copy gives [0, -1].
(c) At [0.55, 0.45]: d^2 = [0.405, 0.405, 3.605,
2.205]. Still a tie between e_1 and e_2. Winner
index 0. True grad still [0, 0]. The STE still says
[0, -1]. Near the boundary the STE's fantasy is
most dangerous: an infinitesimal nudge flips the
winner and the copied gradient optimizes the wrong
branch.
(d) "Exact forward does not imply unbiased
backward: the toy shows [0,-1] versus [0,0]. The
STE is a biased but useful estimator. Its bias is
small only when the quantization gap is small."

## Task 4

(a) [9.93, 2.0, 0.0, 1.0].
(b) With the guard the dead code keeps [-1, 0].
Without it, num/0 gives NaN (and a RuntimeWarning)
for code 3's mean: the codebook is poisoned.
(c) Usage [4, 3, 0, 1]. Dead: code 3.
(d) Reassign e_3 to [0.9, 0.2] + noise (seed 0:
noise [0.00629, -0.00661] -> [0.90629, 0.19339]).
Recomputed usage over the 8 encodings: [2, 3, 2,
1]. The revived code steals two assignments from
e_1. Code 3 is alive.

## Task 5

(a) Single code at origin: distortion = mean of
||z_e||^2 over the 8 encodings = 0.91625. Each
||z_e||^2: [0.85, 1.22, 0.73, 0.68, 1.45, 0.85,
0.82, 0.73]. Mean 7.33/8 = 0.91625.
(b) rate_distortion returns [(0.0, 0.91625), (1.0,
0.19125), (2.0, 0.06625)].
(c) Seed distortions: 0.3957, 0.1778, 0.5664.
Spread 0.39, larger than the K = 2 -> 4 gain
(0.125). At this toy scale the seed dominates. With trained codebooks the K effect dominates.
(d) Acceptable argument: the second bit bought
only 2.9x on aligned codebooks while random-init
spread is 3x. K = 64 risks many dead codes
(C06). Propose: sweep K in {4, 8, 16} with 3
seeds each, plot effective K and distortion, and
pick the elbow.
