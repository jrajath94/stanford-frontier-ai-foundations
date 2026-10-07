# Keys: Lesson 10, Clustering and latent mixture models

## Breadth recall

E01: Initialize centroids. Repeat: assign each
point to the nearest centroid. Move each centroid
to the mean of its assigned points.

E02: J(c, mu) = sum_i ||x^{(i)} - mu_{c(i)}||^2.

E03: Each step minimizes J over one block of
variables, so J is monotone non-increasing and
bounded below. It converges to a local minimum
(coordinate descent). The parameters need not
converge to a global optimum.

E04: A latent variable is an unobserved random
variable in the model (e.g., the mixture
component z^{(i)}) that is marginalized out of
the likelihood.

E05: w^{(i)}_j = p(x^{(i)}|z^{(i)}=j) phi_j /
sum_l p(x^{(i)}|z^{(i)}=l) phi_l.

E06: phi_j = mean of w_j. Mu_j = weighted mean
with weights w_j. Sigma_j = weighted covariance
with weights w_j.

## Deep oral ladders

L01: (1) As E01. (2) Two blobs separate cleanly
in a few iterations. (3) Assignment: each point
goes to its nearest center, lowering its term.
Mean: the mean minimizes the within-cluster sum
of squares. (4) Log J per iteration. Assert
non-increase. (5) EM keeps probabilities. K-means keeps 0/1 decisions. (6) A centroid lost
all points. Guard the division. (7) Rings and
moons break the blob assumption. (8) Multiple
seeds, keep the lowest J, report the spread.

L02: (1) l = sum_i log sum_j phi_j N(x^{(i)}. Mu_j, Sigma_j). (2) Means recover to -1.97, 2.03
from a bad start. (3) Posterior by Bayes' rule
with the current parameters. (4) Log the
marginal likelihood each iteration. Assert
monotone. (5) As Sigma -> sigma^2 I with sigma
-> 0, posteriors harden and EM becomes k-means.
(6) A likelihood decrease means the M-step did
not maximize (implementation bug). (7) Local
optima. Restart. (8) Restarts from spread-out
initializations, keep the best likelihood.

## Analytical exercises

E07: Minimize sum_i w_i ||x^{(i)} - mu||^2:
derivative -2 sum_i w_i (x^{(i)} - mu) = 0 gives
mu = sum_i w_i x^{(i)} / sum_i w_i.

E08: With Sigma_j = sigma^2 I, w^{(i)}_j is
proportional to exp(-||x^{(i)} - mu_j||^2 /
(2 sigma^2)). As sigma -> 0, the largest
exponent dominates: the posterior concentrates
on argmin_j ||x^{(i)} - mu_j||, i.e., hard
assignment to the nearest mean. The M-step mean
then equals the k-means centroid update.

## Failure diagnosis

E09: Singular collapse: a Gaussian component
shrunk onto one (or a few) points, so its
covariance determinant -> 0 and the likelihood ->
infinity. The monotone ascent still holds. The
model is degenerate. Fix: regularize
covariances (add epsilon I), constrain to
spherical, or use fewer components.

## Counterfactual comparison

E10: Team B. One k-means run is seed luck. EM
with restarts explores more of the likelihood
surface and the best-likelihood selection is
principled. Neither is guaranteed global.

## Research question

E11: Falsifiable claim: as component overlap
increases, the variance of final likelihood
across seeds grows faster for k-means than for
EM with the same restarts.

## Implementation task

E12: Verified by the monotone likelihood curve
and recovery of the true means within tolerance.
