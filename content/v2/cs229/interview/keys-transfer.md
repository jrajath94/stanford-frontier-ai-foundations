# Keys: transfer sets

## T01

(a) It does not vanish: the update
is alpha * (y - h) * c with c the
frozen constant, so the weight keeps
moving on pure error feedback with
no signal behind it. The feature
degrades into a second bias term
and its information is lost.
(b) The frozen column is nearly
constant, so X^T X gains a near-
collinear column with the bias
term: the system becomes ill-
conditioned and the solution
depends on the solver's
regularization.
(c) Neither: the right move is to
drop the feature or fix the sensor.
A larger rate on pure error
feedback just makes the weight
drift faster. Per-coordinate rates
cannot restore lost signal.
(d) A feature-variance monitor:
alert when any feature's running
variance hits zero.

## T02

(a) The weight vector w stays
valid: p(x|y) unchanged means the
log odds slope is unchanged. The
intercept b must change.
(b) logit p_new(y=1|x) = w^T x +
b + log(pi_new/pi_old) -
log((1-pi_new)/(1-pi_old)), where
pi is the class prior.
(c) Weight spam examples by
pi_new/(pi_old) relative scale:
w_1 = (0.1 pi)/(pi) adjusted per
class so the weighted empirical
prior matches the new prior.
(d) Then w is wrong too: the
discriminative boundary needs
refitting on new-market data.

## T03

(a) mu_1 moves toward mu_0: the
contaminated mean is 0.8 mu_1 +
0.2 mu_0.
(b) Sigma inflates: the mislabeled
points add between-class spread to
the within-class estimate.
(c) Toward class 1's true region:
the boundary moves to accommodate
the shifted mu_1 and the larger
Sigma, so true class-1 points near
the old boundary get classified 0.
(d) Calibration check plus
per-class held-out likelihood: the
contaminated class shows a
likelihood drop and miscalibration.

## T04

(a) One SMO sweep is O(n^2) in the
worst case (each of n alphas scans
O(n) kernel values), with O(n^2)
memory for the Gram matrix: 1M^2
floats is 8 TB. Infeasible.
(b) Nyström or random Fourier
features: approximate the Gram
matrix with rank m << n. You lose
exactness, the error depends on the
kernel spectrum decay.
(c) Explicit features win when the
kernel is well approximated by few
features (RBF with large
bandwidth) or n is huge: linear
SVM is O(n d).
(d) Measure validation accuracy per
dollar: run RBF-SVM on a 50k
subset, Nyström with m in {1k,
5k}, and linear on full data, all
with timed runs. Pick the best
accuracy inside the time budget.

## T05

(a) The branch contributes its
frozen random projection: y = x +
F_frozen(x). The network is a
shallower effective model plus a
fixed random feature.
(b) The gradient flows through the
identity skip normally, through
the frozen branch it flows through
fixed random weights, which is a
random projection of the upstream
gradient, still a valid (if
unhelpful) direction.
(c) The rest of the network has
enough capacity to fit the training
data without the branch. Falling
loss measures the whole network,
not the branch.
(d) A per-parameter gradient
check: finite differences on the
branch weights must match the
analytic gradient AND must be
nonzero across batches. The
stronger check: the weight update
norm per layer. Zero updates on a
trainable layer is the bug
signature.

## T06

(a) Fix n, vary d (features) from
below n to above n, fit ridgeless
least squares at each d.
(b) Test error vs d, and training
error vs d. Training error falls
monotonically, test error falls,
spikes at d = n, falls again.
(c) Optimal ridge tuning flattens
the peak: the spike is a
variance explosion at the
interpolation threshold, and the
right lambda controls it.
(d) The features must be
sufficiently isotropic (or the
design well-conditioned) so the
interpolation threshold is sharp
at d = n. Heavy-tailed or
collinear features smear the peak.

## T07

(a) Pure noise at T/2, since
alpha-bar = 0 means all signal is
gone halfway.
(b) For t > T/2 the forward
process is already pure noise, so
the denoising targets are noise-
predicting-noise: those ELBO terms
carry no signal about the data.
(c) The sampler starts from noise
at T and denoises, but the model
only learned meaningful
denoising for t < T/2: samples
look like partial denoises,
blurry or washed out.
(d) Held-out ELBO (or bits per
dim): it degrades versus the
correct schedule, concentrated in
the late-t terms.

## T08

(a) It learns to pass the public
tests with minimal effort: hard-
coded branches for the known test
inputs, or outputs shaped to the
test setup, not correct code.
This is reward hacking (U15
SL-03).
(b) No. The KL penalty limits
drift from the base model, but the
base model can already write
test-specific hacks. The penalty
bounds how far the policy goes,
not whether the destination is
right.
(c) Hidden tests written by an
independent team, plus a
mutation score: seeded bugs the
review must catch. The reward uses
only the hidden suite.
(d) The gap between public-test
reward and hidden-test accuracy,
tracked during training. A
widening gap is the hacking
signature.

## T09

(a) The derivation assumes a_t
affects s_{t+1} through B_t. With
delay, the control authority
matrix is zero at the current
step: the first-order condition
gives a_t = 0, which is wrong.
(b) Augment the state with the
previous action: stack s_t over
a_{t-1} to form z_t. Then z_{t+1}
stacks A s_t + B a_{t-1} over a_t,
so the new A is [[A, B],[0, 0]] and
the new B is [[0],[I]]. Standard
LQR form restored.
(c) The Riccati matrices grow from
d x d to (d + m) x (d + m), where
m is the action dimension.
(d) A fractional delay is not a
finite Markov augmentation: the
system is not finite-dimensional
in discrete time. Options: upsample
the time grid so the delay is an
integer number of steps, or model
it as a continuous-time delay and
leave discrete LQR.

## T10

(a) They drift far from 1: some
ratios explode on rare actions,
others collapse toward 0. The data
is 50 epochs stale.
(b) The surrogate is local to
theta_old (notes 21.2). After many
epochs theta has left the trust
region, so the surrogate optimizes
a fiction built on the old state
distribution.
(c) KL(theta || theta_old) and the
clipped fraction. KL large means
the excursion is too far, clipped
fraction near zero means eps no
longer binds and the updates are
unclipped.
(d) Fewer epochs per batch (for
example 3-4), more frequent
resampling, same total gradient
steps. Monitor KL per epoch and
stop the epochs early when KL
crosses the threshold.
