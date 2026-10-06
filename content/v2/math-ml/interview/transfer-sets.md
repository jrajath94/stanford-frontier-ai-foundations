# Unfamiliar transfer sets, math-ml RUN 5

Date: 2026-10-06. Questions only. Keys in interview/keys-transfer.md.
Closed-book. These items are new scenarios, not lesson recall. Each
names the units it transfers. Solve without the lesson open.

## Set A, U02 x U06 (geometry into margins)

A1. You are given w = [3, 4] and b = -25 as a classifier. Without
retraining, compute the geometric margin width. Then scale w and b
by 1/5: which quantities change and which stay the same? (U02-C03,
U06-C08.)
A2. A teammate "improves" a linear SVM by doubling all feature
values. Using the C11 numbers (kappa 3.35e6 raw vs 2.618
standardized), predict what happens to the dual alphas and the
test accuracy. Then state the one-line fix. (U06-C09, U06-C11.)
A3. The Gram of a linear kernel is X X^T. For the C01 design
matrix (4 x 2), state the Gram's shape and its rank without
computing it. Then say which U02 concept fixes the rank and what
the fix costs. (U02-C02, U02-C09, U06-C07.)

## Set B, U04 x U06 (probability into classification)

B1. A coin has p = 0.7. You observe 4 flips with 3 heads. The
lesson used Laplace (k+1)/(n+2) = 0.6667. Now a logistic model
with one bias parameter is fit to the same 4 flips. Predict
whether its p_hat is above or below 0.6667, and why. (U04b-SB14,
U06-C03, U05-C04.)
B2. Two Gaussians N(0, 1) and N(2, 1) with equal priors. Write the
Bayes classifier's decision rule in the form w x + b = 0, giving
w and b. Then compute its error using Phi. (U04c-SB18, U06-C08.)
B3. The softmax Jacobian is diag(p) - p p^T. Prove it is PSD
using only the fact that it is a covariance matrix. Then say
which U04 concept this reuses. (U02-C10, U04a-SB04, U06-C04.)

## Set C, U05 x U09 (risk into unsupervised)

C1. k-means train inertia falls at every k (7.6417, 0.3698,
0.3161, 0.1789). A teammate proposes picking k by train
inertia. Map this proposal onto a U05 failure mode by name, and
state the unsupervised analogue of the fix. (U05-C07, U09-C12.)
C2. You run PCA and keep enough components for 99 percent
variance on train. On new data the reconstruction error is 3x
the train error. Name the U05 concept, and say whether keeping
more components helps or hurts. (U05-C12, U09-C04.)
C3. The factor toy: PCA k=1 gives MSE 1.2785, k=2 would give a
lower train MSE. A teammate cites "lower error is better" and
ships k=2. Write the two-sentence falsification using the
eigenvalues [7.33, 3.83, 0.0085]. (U05-C09, U09-C09.)

## Set D, U03 x U06 (optimization into models)

D1. Newton on logistic regression uses the Hessian X^T W X with
W_ii = p_i(1 - p_i). On separable data, what happens to W as
||w|| grows, and what does that do to the Newton step? Predict
before you compute. (U03-C08, U06-C03.)
D2. The hinge subgradient at a kink is a set, not a vector.
Write the subgradient set of max(0, 1 - z) at z = 1. Then say
why SGD with constant eta can oscillate there, in one
sentence. (U03-C07, U06-C10.)
D3. Dual coordinate ascent updates one alpha at a time. Map this
onto a U03 concept by name, and state the per-update cost for
the RBF dual in terms of n. (U03-C07, U06-C09.)

## Set E, U01 x everything (language and checking)

E1. A function signature says f(X: (n, d), w: (d,)) -> (n,).
A caller passes X with shape (d, n). Name the U01 concept that
catches this, and the U02 concept that explains why the
product is still defined but wrong. (U01-C07, U02-C05.)
E2. A log-likelihood function returns -inf on the first call.
List three distinct causes, one from U01 (numerics), one from
U04 (modeling), one from U06 (computation). (U01-C11, U04-C06,
U06-C04.)
E3. Seed 7 is set once at the top of a script that draws train
data, shuffles, and draws validation data. A teammate moves the
seed line below the shuffle. Using U01-C12, state exactly what
changes and what stays the same across reruns. (U01-C12.)

## Set F, U09 x U04 (latent models meet density)

F1. k-means is the hard limit of a Gaussian mixture. State the
limit precisely (which parameter goes to what), and say which
lesson-04c concept becomes trivial in that limit. (U09-C01,
U04c C05/C06.)
F2. The Parzen window (U04c-SB25) and the RBF kernel (U06-C07)
use the same exponential. State one thing they share and one
thing that differs (what is optimized vs what is fixed).
(U04c-SB25, U06-C06.)
F3. Held-out likelihood picks mixture components. Held-out
inertia picks k. State why the likelihood version can use a
likelihood-ratio argument and the inertia version cannot.
(U04-C06, U09-C12.)
