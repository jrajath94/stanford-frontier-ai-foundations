# Lab 07, trees and ensembles

Unit: math-ml-U07. Date: 2026-10-06. Keys in labs/keys-lab-07.md.
All numbers computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## Task 1, split scan audit on a new toy

Data: x = [0.2, 0.5, 0.6, 1.1, 1.4, 1.8],
y = [0, 0, 1, 1, 1, 0]. Scan midpoints with the
Gini gain. Report the best threshold, its gain,
and the weighted child Gini.

Measured reference: best t = 0.55, gain 0.25,
weighted Gini 0.25. Your run must reproduce
these to 1e-9.

Assert: gain <= parent Gini (0.5). the scan
visits exactly 5 candidates.

## Task 2, AdaBoost round 1 by hand

Points x = [0, 1, 2, 3], y = [-1, -1, +1, -1].
Stump: x <= 1.5 -> -1 else +1. Start weights
0.25 each.

(a) Compute eps, alpha, and the normalized
weight vector. Reference: eps 0.25, alpha
0.5493061443, weights [1/6, 1/6, 1/6, 1/2].

(b) A second stump (x <= 1.5 -> +1 else -1)
misses x = 0, 1, 2 on the new weights. Compute
its eps and alpha, and explain in one sentence
what a zero vote means. Reference: eps 0.5,
alpha 0.0.

## Task 3, OOB versus in-bag error

Points x = [0,1,2,3], y = [-1,-1,+1,-1].
Bootstrap sets: rep 0 [2,2,3,3], rep 1 [0,2,3,3],
rep 2 [0,1,1,3]. Fit the best Gini stump to each
replicate (ties: smaller threshold, then -1).

(a) Report each fitted stump and its in-bag
error. Reference: rep 0: t = 2.5, err 0.0. rep 1:
t = 1.0, err 0.25. rep 2: constant -1, err 0.0.
Mean in-bag error 0.0833.

(b) Build the OOB prediction table with majority
vote (ties toward -1) and report the OOB error.
Reference: point 0: reps [0] -> +1, wrong. point
1: reps [0,1] -> tie -> -1, right. point 2: reps
[2] -> -1, wrong. point 3: no OOB reps. OOB
error 2/3 = 0.6667.

(c) Explain in two sentences why (a) is
optimistic and (b) is honest, using the two
numbers 0.0833 and 0.6667.
