# Lab 04: k-means and EM

Date: 2026-10-06. Unit: cs229-U10.
Work the tasks, then check keys-lab-04.md. Run all code.

Setup: numpy only. Seeds as stated per task. No other installs.

## Task 1: k-means seed sensitivity

Data: two Gaussians in 2-D, means (-2, 0) and (2, 1),
sd 0.8, 150 points each, seed 42. Implement k-means
(k = 2, centroids initialized to random training
examples). Run 20 iterations for seeds 0..4. Report
the initial and final distortion J (mean squared
distance) per seed. Assert J never increases within a
run. State whether the final J depends on the seed.

## Task 2: EM monotone ascent

Data: 1-D mixture, 150 points from N(-2, 1), 150 from
N(2, 1), seed 42. Implement EM for a 2-component
Gaussian mixture. Run 30 iterations from three starts:
means (0.0, 0.5), (-5.0, 5.0), (1.0, 1.5). Report the
initial and final log likelihood and the final means
for each start. Assert the likelihood never decreases.

## Task 3: hard vs soft on the same data

Using the Task 1 data and seed 0: compute the
k-means label agreement with the true component
labels (best of the two labelings). State the
agreement and one reason EM can beat it on
overlapping clusters.

## Deliverable

A short log: the three result blocks with numbers. No
essay. The numbers must match keys-lab-04.md within the
stated tolerance.
