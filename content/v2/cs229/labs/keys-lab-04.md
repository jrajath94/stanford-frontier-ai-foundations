# keys-lab-04.md

Date: 2026-10-06. Reference outputs for lab 04.

Computed with numpy on this machine. Tolerances below absorb
platform float differences.

## Task 1

Final J per seed (initial -> final):

- seed 0: 4.92 -> 1.19
- seed 1: 2.05 -> 1.19
- seed 2: 2.27 -> 1.19
- seed 3: 1.39 -> 1.19
- seed 4: 12.73 -> 1.19

Tolerance: final J within 0.05. J monotone
non-increasing in every run (asserted) on this
well-separated data all seeds reach the same final J.
on harder data they need not.

## Task 2

Per start (initial ll -> final ll, final means):

- (0.0, 0.5): -989.6 -> -595.8, (-2.09, 1.93)
- (-5.0, 5.0): -1934.6 -> -595.8, (-2.09, 1.93)
- (1.0, 1.5): -1215.4 -> -595.8, (-2.09, 1.93)

Tolerance: final ll within 1.0, means within 0.05.
Likelihood monotone non-decreasing in every run
(asserted). All three starts converge to the same
local optimum here.

## Task 3

k-means label agreement: 0.987. EM can beat hard
assignment on overlapping clusters because soft
responsibilities let boundary points share weight
between components instead of committing to one.
