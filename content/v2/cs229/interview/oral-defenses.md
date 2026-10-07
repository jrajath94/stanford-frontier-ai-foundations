# Oral defenses: deep ladders across cs229

Date: 2026-10-06. Keys in
keys-oral.md. Each ladder has 8
follow-ups: define, toy, derive,
implement, compare, debug,
critique, design. Provenance:
original practice. No employer
attribution.

## O01: Least squares (U02)

F1: Define the LMS objective and
the batch gradient update.
F2: Toy: 3 points, 1 feature, 2
gradient steps by hand from zero.
F3: Derive the normal equations.
F4: Implement both and check they
agree on the toy.
F5: Compare gradient descent with
the normal equations on cost and
stability.
F6: Debug: the loss oscillates and
never falls. Name two causes.
F7: Critique the Gaussian-noise
justification for least squares.
F8: Design the experiment that
decides between LMS and the normal
equations for n = 10^7, d = 100.

## O02: Logistic regression (U03)

F1: Define the sigmoid and the
Bernoulli likelihood.
F2: Toy: 2 points, compute the
gradient of the log likelihood by
hand.
F3: Derive the gradient: show it
has the form (y - h) x.
F4: Implement stable logistic loss
with the log-sum-exp trick and
test it on extreme logits.
F5: Compare logistic regression
with GDA on assumptions and data
efficiency.
F6: Debug: weights blow up on
separable data. Explain and fix.
F7: Critique "logistic regression
outputs probabilities."
F8: Design a calibration study for
the trained classifier.

## O03: SVM duality (U06)

F1: Define functional and
geometric margin.
F2: Toy: 2 points in 1d, compute
the max-margin separator by hand.
F3: Derive the dual from the
Lagrangian.
F4: Implement SMO for the toy and
check KKT conditions.
F5: Compare the primal and dual on
scaling with n and d.
F6: Debug: all alphas hit the C
bound. Diagnose.
F7: Critique the "support vectors
are the informative points" story.
F8: Design the model-selection
experiment for C and the RBF
bandwidth.

## O04: Backpropagation (U07)

F1: Define the backward contract
of a module.
F2: Toy: a 2-layer scalar network,
forward and backward by hand.
F3: Derive the chain-rule update
for a linear layer in matrix
form.
F4: Implement it and check with
finite differences.
F5: Compare reverse-mode with
forward-mode autodiff on cost.
F6: Debug: gradients are exactly
zero below layer 3. Name two
causes.
F7: Critique "deeper is better"
for this mechanism.
F8: Design the experiment that
isolates a vanishing-gradient
failure from an optimization
failure.

## O05: Bias and variance (U08)

F1: State the bias/variance
decomposition.
F2: Toy: k-NN on 1d data, sketch
bias and variance vs k.
F3: Derive the decomposition from
E[(y - h)^2].
F4: Implement the decomposition
empirically with bootstrap
resampling.
F5: Compare the classical U-curve
with double descent.
F6: Debug: test error rises while
train error falls. Two possible
causes, two different fixes.
F7: Critique "more data always
helps."
F8: Design the study that
attributes a model's error to
bias, variance, or noise.

## O06: EM (U10)

F1: Define the EM lower bound via
Jensen.
F2: Toy: 1d two-Gaussian mixture,
one E step and one M step by hand.
F3: Derive why the likelihood
never decreases.
F4: Implement EM for the toy and
check monotonicity.
F5: Compare EM with k-means on
assumptions and output.
F6: Debug: a variance collapses
to zero and likelihood explodes.
Explain and fix.
F7: Critique "EM finds the MLE."
F8: Design the initialization
study for a 10-component mixture.

## O07: Diffusion objective (U12)

F1: Define the forward Gaussian
chain.
F2: Toy: 1d, 2 steps, compute the
noisy sample distribution by hand.
F3: Derive the denoising objective
from the ELBO.
F4: Implement one training step
and check the loss is finite.
F5: Compare epsilon-prediction
with x0-prediction.
F6: Debug: samples are pure
noise. Name two causes.
F7: Critique "the ELBO trains
likelihood."
F8: Design the held-out
evaluation that is not the
training ELBO.

## O08: Attention (U14)

F1: Write single-head attention.
F2: Toy: the T = 3 hand
computation from U14.
F3: Derive the matrix form from
the per-position rule.
F4: Implement masked attention and
test the zero triangle.
F5: Compare MHA, GQA, and MQA on
memory and quality.
F6: Debug: u_t changes when x_{t+1}
is perturbed. Diagnose.
F7: Critique the sqrt(d_h)
scaling heuristic.
F8: Design the KV-cache
provisioning for a 128k-context
serving system.

## O09: Bellman and value iteration (U15)

F1: Define V^pi.
F2: Toy: the two-state V(A) = 9
computation.
F3: Derive the Bellman equation
by splitting the first step.
F4: Implement value iteration and
show the 0.9 error ratio.
F5: Compare value and policy
iteration.
F6: Debug: value iteration
diverges. Two causes.
F7: Critique the known-model
assumption.
F8: Design the model-estimation
experiment for a new simulator.

## O10: PPO (U16)

F1: Define the likelihood ratio
r_t(theta).
F2: Toy: compute the clipped
surrogate for (Ahat, r) = (1,
1.3) and (-1, 0.5).
F3: Derive the log-derivative
identity behind REINFORCE.
F4: Implement the bandit gradient
estimator and check
unbiasedness.
F5: Compare PPO with vanilla
policy gradient on data reuse.
F6: Debug: the surrogate improves
but returns fall. Diagnose.
F7: Critique "PPO is off-policy."
F8: Design the epoch schedule and
the two guard metrics.
