# Keys, lab 07 trees and ensembles

Date: 2026-10-06.

## Task 1

Candidates 0.35/0.55/0.85/1.25/1.6 give weighted
Gini 0.4/0.25/0.4444/0.5/0.4 and gains
0.1/0.25/0.0556/0.0/0.1. Best t = 0.55, gain
0.25, weighted Gini 0.25. Gain 0.25 <= parent
Gini 0.5. Five candidates: n - 1 = 5.

## Task 2

(a) eps = 0.25 (only x = 3 missed). alpha = 0.5
ln(0.75/0.25) = 0.5 ln 3 = 0.5493061443.
Weights: correct points 0.25 e^-0.5493 =
0.14435, missed point 0.25 e^0.5493 = 0.43301.
sum 0.86606. normalized [1/6, 1/6, 1/6, 1/2].

(b) eps = 1/6 + 1/6 + 1/6 = 0.5. alpha = 0.5
ln(1) = 0.0. A zero vote means the stump is at
chance level: it contributes nothing to the
ensemble sum.

## Task 3

(a) rep 0 [2,2,3,3]: only boundary 2.5, stump
t = 2.5, in-bag err 0.0. rep 1 [0,2,3,3]:
boundaries 1.0 (err 0.25) and 2.5 (err 0.25).
tie -> t = 1.0, in-bag err 0.25. rep 2
[0,1,1,3]: all labels -1, constant stump,
in-bag err 0.0. Mean 0.0833.

(b) Point 0: OOB reps [0], stump t=2.5 gives
+1, true -1: wrong. Point 1: OOB reps [0,1],
votes +1/-1, tie -> -1, true -1: right. Point
2: OOB reps [2], constant -1, true +1: wrong.
Point 3: no OOB reps. OOB error 2/3 = 0.6667.

(c) In-bag error scores each stump on the data
it was fit to: 0.0833 is optimistic. OOB
scores each point only on trees that never saw
it: 0.6667 is the honest estimate on this tiny
toy.
