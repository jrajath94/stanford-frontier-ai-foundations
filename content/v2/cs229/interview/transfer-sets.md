# Transfer sets: changed-scenario transfer across cs229

Date: 2026-10-06. Keys in
keys-transfer.md. Each set changes
one constraint on a mechanism taught
in the course and asks the learner
to transfer. Provenance: original
practice. No employer attribution.

## T01: LMS with a broken sensor (U02)

You fit LMS on streaming data. One
feature's sensor freezes at its last
value halfway through training.

(a) What happens to the gradient
contribution of that feature?
(b) The normal equations on the full
dataset: how does the frozen column
affect the solution?
(c) Can per-coordinate learning
rates fix this? Explain.
(d) Name the check that would have
caught this before training.

## T02: Logistic regression under label shift (U03)

A spam classifier trained with
logistic regression moves to a new
market where spam is 10x rarer, but
p(x|y) is unchanged.

(a) Which parameters of the model
are still valid, and which one must
change?
(b) Derive the correction from
Bayes rule.
(c) The team instead reweights the
training loss. Write the weight for
each class.
(d) What breaks if p(x|y) also
changed?

## T03: GDA with a liar (U04)

Your GDA training data for class 1
contains 20% mislabeled class-0
points. Covariances are shared.

(a) How does the estimated mu_1
move?
(b) How does the shared Sigma
change?
(c) The decision boundary shifts.
In which direction?
(d) Which U04 diagnostic detects
this before deployment?

## T04: Kernel SVM on a budget (U05/U06)

You have 1M training points and may
spend O(n^2) once but never O(n^3).
The RBF kernel SVM needs the full
Gram matrix.

(a) What is the exact asymptotic
cost of one SMO sweep here?
(b) Propose a principled
approximation that keeps the kernel,
and name what you lose.
(c) The alternative is explicit
features. When does that win?
(d) State the decision rule you
would actually use to choose.

## T05: Backprop with a dead branch (U07)

A residual network trains, but one
residual branch's weights freeze at
init (a bug masks its gradient).

(a) What does the forward pass
compute?
(b) What does the backward pass
deliver to earlier layers through
that branch?
(c) Training loss still falls. Why
does this not prove the branch is
fine?
(d) Name the gradient check that
catches it.

## T06: Double descent on purpose (U08)

You must demonstrate double descent
to a skeptic with a real dataset
and linear regression.

(a) Describe the experimental
design: what varies on the x-axis?
(b) What two curves do you plot?
(c) The skeptic says "just use
ridge." What does optimal
regularization do to the peak?
(d) What assumption about the data
does the demo need?

## T07: Diffusion with a wrong schedule (U12)

Your diffusion model trains with a
noise schedule whose alpha-bar hits
0 at t = T/2 instead of T.

(a) What does x_{T/2} look like?
(b) What happens to the ELBO terms
for t > T/2?
(c) Sampling still runs. Describe
the samples.
(d) Which held-out metric catches
this first?

## T08: RLVR with a helpful verifier (U15)

Your code verifier was written by
the same team training the model,
and it accepts any output that
passes the public tests.

(a) What will the policy learn to
optimize?
(b) The KL penalty is in place.
Does it save you? Why or why not?
(c) Redesign the verifier with two
independent checks.
(d) What metric, tracked during
training gives the warning?

## T09: LQR with a delay (U16)

Your robot's actions take effect one
time step late: s_{t+1} depends on
a_{t-1}, not a_t.

(a) Why does the standard LQR
derivation break?
(b) Propose the minimal state
augmentation that restores the LQR
form, and write the new A matrix
structure.
(c) What happens to the Riccati
dimension?
(d) The delay is actually 0.5
steps. What now?

## T10: PPO on stale data (U16)

Your PPO implementation samples one
large batch and runs 50 epochs on
it with eps = 0.2.

(a) What happens to the likelihood
ratios by epoch 50?
(b) The clipped surrogate keeps
improving. Why is that
misleading?
(c) Name the two numbers that
would have told you to stop.
(d) Redesign the update schedule
for a fixed sampling budget.
