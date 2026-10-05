---
title: "Lab 3: EM from Scratch"
course: cs229
type: lab
---

Implement Gaussian mixture clustering in NumPy. 45 minutes. No sklearn.

## Problem 1: E-step (15 min)

Write a function `e_step(X, phi, mu, sigma)` that returns the responsibilities \(w_j^{(i)}\):

\[ w_j^{(i)} = \frac{\phi_j \, \mathcal{N}(x^{(i)} \mid \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \phi_l \, \mathcal{N}(x^{(i)} \mid \mu_l, \Sigma_l)}. \]

Use `scipy.stats.multivariate_normal.pdf` for the Gaussian density, or write it yourself. Work in log-space if you hit underflow, but a direct implementation passes for small tests.

**Check:** each row of the responsibility matrix sums to 1. For well-separated clusters, responsibilities should be near 0 or 1.

## Problem 2: M-step (15 min)

Write `m_step(X, w)` that returns updated parameters:

\[ \phi_j = \frac{1}{m}\sum_i w_j^{(i)}, \quad \mu_j = \frac{\sum_i w_j^{(i)} x^{(i)}}{\sum_i w_j^{(i)}}, \quad \Sigma_j = \frac{\sum_i w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_i w_j^{(i)}}. \]

**Check:** run 20 iterations on a synthetic 2-Gaussian dataset (generate with `np.random.multivariate_normal`). The recovered means should be close to the true means, up to permutation of cluster labels.

## Problem 3: K-means as a limit (15 min)

1. Modify your E-step to use hard assignments (0/1 instead of soft responsibilities). Show that the resulting algorithm is k-means.
2. In two sentences, explain when you would prefer soft EM over k-means in practice.

**Check:** hard E-step plus the M-step with 0/1 weights gives exactly the k-means update: assign each point to the nearest centroid, recompute centroids as cluster means. Prefer EM when clusters overlap, have different shapes (full covariances), or you need calibrated cluster probabilities.

**Interview trap:** "Does EM converge to the global optimum?" Answer: no. The likelihood increases monotonically, but EM converges to a local optimum. Initialization matters. In practice use k-means++ or multiple restarts.

## Scoring

- Working implementation plus the k-means connection: strong signal on unsupervised learning.
- If the M-step covariance update was the hard part, reread the L10 derivation, then reimplement from memory.
