# Lab keys 02, sampling and estimation computations

Date: 2026-10-06. Computed values, numpy 1.26.4, float64, seed 7.
Executed by compute_run2_labs.py (fix builder, 2026-10-06). Every
number below is printed by that script. Task 1 draws use
Generator.binomial (the Bernoulli sampler) on the seed-7 stream.

## Task 1

(a) Draws [0 1 1 0 0 1 0 1], mean 0.5.
(b) Predicted typical error sqrt(0.3*0.7/8) = 0.1620185174601965.
Observed |0.5 - 0.3| = 0.2. Same order. The rate predicts the scale,
not the exact miss.
(c) n = 800: mean 0.30125, error 0.00125. The error shrank ~160x,
beating the 10x prediction: a favorable fluctuation. The 1/sqrt(n)
law predicts typical shrinkage, and this run landed on the lucky
side. Report the luck, do not hide it.

## Task 2

(a) Grid argmax p = 0.7, log-likelihood -2.2739976361421332.
(b) Analytic MLE 0.75 gives -2.249340578475233, higher: the grid is
coarse and misses the peak. Finer grid or the derivative finds it.
(c) n = 1 (H only): the grid max is p = 0.9 (boundary of the grid),
log-likelihood log(0.9). The true MLE is p = 1.0. Danger: one draw
certifies total certainty. The estimate has no notion of its own
fragility.

## Task 3

(a) Counts [2, 2, 2, 2], edges [0, 0.625, 1.25, 1.875, 2.5].
(b) Width 0.625, density per bin 2/(8*0.625) = 0.4. Total area:
4 * 0.4 * 0.625 = 1.0. Checks out.
(c) 8 bins give [1, 1, 1, 1, 1, 1, 1, 1]: one point per bin. The
"density" is pure noise shaped like the sample. Lesson: with n = 8,
finer bins do not add information. They add the illusion of detail.
