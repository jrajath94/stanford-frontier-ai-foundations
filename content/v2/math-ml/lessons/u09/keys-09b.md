# Keys, lesson 09b unsupervised leaves

Date: 2026-10-06. Test-mode: solve closed-book first. All numbers
computed 2026-10-06, numpy 1.26.4, float64, seed 7.

## E01

Iter 0: assignments [0,0,0,1,1,1], J = 0.1700. Centers become
(0.0333, 0.1) and (5.0333, 5.0). Iter 1: same assignments,
J = 0.133333. Iter 2: unchanged.

## E02

Bad start centers (0,0) and (0.1,0.1). Point (0,0): distances 0
and 0.02, assigns 0. Point (0.2,0.1): 0.05 and 0.01, assigns 1.
Point (-0.1,0.2): 0.05 and 0.05, tie. Argmin picks 0.

## E03

Assignment: each point takes its nearest center, so the sum can
only fall. Update: the mean minimizes the sum of squares for its
group, so the sum can only fall again.

## E04

The guard keeps the previous center when its group is empty.
The lesson code does exactly this instead of dividing by zero.

## E05

Raw: d(A,B)^2 = 5, d(A,C)^2 = 5.84. Nearest B. Scaled y x5:
d(A,B)^2 = 101, d(A,C)^2 = 29.84. Nearest C.

## E06

Scaled points: A (0,0), B (1,10), C (2.2,5). Means (1.0667,
Scaled points: A (0,0), B (1,10), C (2.2,5). Means (1.0667,
5.0), population stds (0.8994, 4.0825). Standardized squared
distances from A: to B 7.2363, to C 7.4835. Nearest returns to
B (index 1). Computed 2026-10-06.

## E07

Squared Euclidean distance multiplies each axis difference by
its unit scale squared. A feature in millimeters outvotes one
in meters by a factor of 10^6, regardless of information.

## E08

Mean (3, 4.8). Deviations: (-2,-2.8), (-1,-1.8), (0,0.2),
(1,1.2), (2,3.2). Sum of squares x: 10, /5 = 2. Cross: 15,
/5 = 3. y: 22.8, /5 = 4.56.

## E09

Product 6.541656 * 0.018344 = 0.12 = 2*4.56 - 9. Sum 6.56 =
2 + 4.56. Both hold.

## E10

Centered: (-2, -2.8). Dot with PC1: -2*0.551163 -
2.8*0.834398 = -3.438640. Reconstruct: mean + (-3.438640)*v1
= (1.1048, 1.9307). Error vector (-0.1048, 0.0693), squared
norm 0.0158 for this point.

## E11

Measured MSE 0.009172. Dropped eigenvalue 0.018344 / d (2) =
0.009172. Equal to 4 digits. The identity holds on the toy.

## E12

Eckart-Young: the rank-k SVD truncation is the best rank-k
approximation under Frobenius norm. It rests on U02-C09.

## E13

PC1 maximizes variance in raw units. A feature with large units
has large variance by construction, so PC1 aligns with it even
when it carries no signal. The components report the ruler,
not the data.

## E14

The third sensor has noise std 3.0, variance 9. PCA must spend
a direction on that variance. Eigenvalue 3.83 is the noise
sensor's share after projection onto the orthogonal complement
of PC1.

## E15

x = W z + mu + eps, eps ~ N(0, Psi) with Psi diagonal. PCA is
the special case Psi = sigma^2 I. The difference: per-dimension
noise versus one shared noise level.

## E16

k = 1 wins on held-out. The second direction models the noisy
sensor. It helps train reconstruction but hurts new data. This
is the predicted outcome. The exercise asks the learner to run
it.

## E17

From k=1 to k=2: 14.1768 - 0.4439 = 13.7329. From k=2 to
k=5: 0.4439 - 0.2939 = 0.1500. Elbow at k = 2: the first step buys 90x the
second step's gain.

## E18

Train inertia: 7.6417, 0.3698, 0.3161, 0.1789. It falls at every
k, so "pick the minimum" would pick k = 5 (or k = n with zero).
Only held-out inertia can say when new centers stop helping.

## L01, k-means mechanics

(a) Minimum: alternate nearest-center assignment and
mean-update to minimize the within-cluster sum of squares.
Strong: names the NP-hardness and the local-minimum caveat.
Red flag: "it converges to the optimum." Rubric: local vs
global distinguished. Remediation: the bad-start numbers.

(b)-(h): reproduce 0.1700 -> 0.133333. Each step minimizes its
subproblem. Guard keeps old center. K = 5 predicts invented
splits with falling inertia. Rising J means the assignment and
update disagree (stale centers). The attack uses the 145.17
path. The experiment counts success over 20 starts per method.

## L02, distance and scaling

(a) Minimum: the nearest neighbor depends on per-axis units.
Strong: quantifies the squared-scale vote change. Red flag:
"Euclidean distance is unit-free." Rubric: flip reproduced.
Remediation: C02.

(b)-(h): 1, 2, 1 sequence. Factor 25 from 5^2. Zero-variance
guard skips constant columns. X100 predicts nearest C (index
2) by the same arithmetic. NaN means division by zero std. 
"standardize always" attack uses binary columns. Transfer to
lesson 06 C11.

## L03, PCA

(a) Minimum: eigendecomposition of the covariance, keep the
top directions. Strong: Lagrange derivation of the eigen
equation. Red flag: "components are meaningful by
construction." Rubric: derivation present.

(b)-(h): eigenvalues and 0.9972 reproduced. SVD must agree to
~1e-12. Noise variance 10 predicts PC1 rotating to the noise
axis. PC1 at the mean means centering was skipped. The attack
is the noise-feature toy. Transfer to U02-C09.

## L04, reconstruction

(a) Minimum: per-entry MSE equals dropped eigenvalues over d.
Strong: cites Eckart-Young. Red flag: "PCA minimizes all
errors." Rubric: identity verified numerically.

(b)-(h): 0.009172 = 0.018344/2. Dropped directions carry the
dropped eigenvalues. K = 0 predicts 3.28 (measured by the
learner). Absolute error predicts PCA not optimal. Worse than
mean means a centering or projection bug. The attack is chords
on a circle. The k = 0 experiment design fixes seed and
protocol.

## L05, factor bridge

(a) Minimum: x = Wz + mu + eps with per-dimension noise.
Strong: names Psi diagonal vs sigma^2 I. Red flag: "factor
analysis is just PCA." Rubric: noise difference stated.

(b)-(h): 1.2785 and eigenvalues reproduced. 3.83 explained as
the noise sensor's direction. Equal noise predicts eigenvalues
~[signal + 0.01, 0.01, 0.01]-ish (learner measures). Degenerate
loadings point to lesson 04c C11. The attack is the noisy-sensor
toy. Transfer to 04c C06 EM.

## L06, held-out evaluation

(a) Minimum: inertia of held-out points under train centers.
Strong: states the elbow logic and the leak warning. Red flag:
"the elbow proves k." Rubric: elbow vs proof distinguished.

(b)-(h): four pairs reproduced. Elbow at 2 from the 13.73 vs
0.15 drops. 10 seeds predict >= 8 elbows at 2. 6 held-out
points predict noisy elbows. Zero held-out inertia means the
centers saw the held-out data. The attack: elbow shows
diminishing returns, not truth. Transfer to U05-C08.
