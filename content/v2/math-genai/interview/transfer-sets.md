# Unfamiliar transfer sets, math-genai U01-U10

Date: 2026-10-06. Questions only. Keys in
interview/keys-transfer.md. Closed-book. Do not read
the keys first. Each set changes one constraint of a
lesson toy and asks for a prediction plus a
computation. Interview provenance: role-derived
practice, not employer material.

## T-U01, generative modeling basics

T1. A 3-outcome toy. Data P = [0.5, 0.3, 0.2].
Model Q = [1/3, 1/3, 1/3]. (a) Compute H(P),
CE(P, Q), and KL(P || Q) in bits. (b) Changed
constraint: outcome 3 becomes impossible in the
data, so P' = [0.625, 0.375, 0]. Recompute H(P')
and KL(P' || Q). (c) What is KL(Q || P')? State
the support rule it teaches.

T2. Four coin flips: [1, 1, 0, 1]. (a) Give the
MLE of the heads probability. (b) Changed
constraint: the flips are [1, 1, 1, 1]. Give the
MLE. What is the log-likelihood the model
assigns to a future tail? (c) Name the standard
fix and compute its estimate.

## T-U02, variational divergence minimization

T1. X takes values {0, 2} with equal probability,
f(x) = x^2. (a) Compute E[X], f(E[X]), E[f(X)],
and the Jensen gap. (b) Changed constraint: the
values shift to {1, 3}. Recompute all four.
(c) Which quantity stayed fixed and why? State
the general fact.

T2. P = [0.5, 0.3, 0.2], Q = [1/3, 1/3, 1/3].
(a) Compute the optimal discriminator D*(x) =
p(x)/(p(x) + q(x)) per outcome. (b) Changed
constraint: the generator moves to Q' = [0.6,
0.2, 0.2]. Recompute D*. (c) Which outcomes
cross 0.5, and what update direction does each
crossing imply for the generator?

## T-U03, GAN foundations

T1. Same P, Q, D* as T-U02. (a) Compute the
minimax value V = E_P[ln D*] + E_Q[ln(1 - D*)].
Verify it against 2*JS - 2*ln 2. (b) Changed
constraint: the generator collapses to Q' =
[1, 0, 0] while D stays stale. Compute V and the
generator's non-saturating objective E_Q'[ln D*].
(c) The discriminator retrains to the new D*.
Recompute the generator's non-saturating
objective. State the lesson about stale
discriminators.

T2. D is fixed at D* from T-U02. (a) The
generator moves mass from outcome 3 to outcome
1. Compute the gradient of E_Q[ln D*] with
respect to that move. (b) Changed constraint:
the generator may now also move mass freely.
Where does the mass go, and what does that imply
for sample diversity under an optimal
discriminator?

## T-U04, Wasserstein training

T1. P is a point mass at 0. Q_eps puts mass
1 - eps at 0 and eps at 10. (a) Compute W1 and
the Jensen-Shannon divergence (nats) at eps =
0.01 and eps = 0.5. (b) Changed constraint: eps
= 0.5 was the lesson-scale case. the new case is
eps = 0.01. Which divergence still gives the
generator a usable gradient, and why?

T2. A critic f(x) = 3x on the line. P is a point
mass at 0, Q at 2. (a) State the true W1.
(b) Changed constraint: the critic is one linear
layer with weights clipped to [-0.5, 0.5].
Compute the critic's distance estimate and the
gap to the true W1. (c) Name the mechanism
behind the gap. What does the gradient-penalty
alternative constrain instead of the weight box?

## T-U05, variational autoencoders

T1. Posterior q = N(0.5, 0.5^2), prior N(0, 1).
(a) Compute KL(q || prior) in nats.
(b) Changed constraint: the posterior tightens
to sigma = 0.1. Recompute the KL. (c) Describe
posterior collapse in these numbers. Which
regime does the optimizer prefer, and why is
that a problem?

T2. A new latent dimension buys 0.05 nats of
reconstruction and costs 0.04 nats of KL.
(a) At beta = 1, does the beta-VAE keep the
dimension? (b) Changed constraint: beta = 1.6,
then beta = 4. Decide keep or drop at each.
(c) The lesson measured beta* = 1.6007 on its
toy. State what that number means.

## T-U06, discrete latents and VQ-VAE

T1. Codebook e1 = (1, 0), e2 = (0, 1), e3 =
(-1, 0). Encoder output z = (0.8, 0.6).
(a) Compute the squared distances and name the
winning code. Compute the commitment loss and
the codebook loss. (b) Changed constraint: z
moves to (0.1, 0.9). Recompute. (c) What does
the straight-through estimator pass backward,
and what happens to e3 over training?

T2. A batch of 8 tokens uses codes with counts
[4, 3, 0, 1]. (a) Which code is dead, and what
does dead mean for its gradient? (b) Changed
constraint: the codebook grows to 8 codes with
the batch fixed at 8 and random assignment.
Compute the probability a given code is unused
and the expected dead count. (c) Name one fix
and what it does to the stale code.

## T-U07, DDPM derivation

T1. The lesson uses linear betas from 1e-4 to
0.02 with alpha_bar_50 = 0.77718. (a) Changed
constraint: betas are constant 0.01 for T =
100. Compute alpha_bar_50. (b) Compute the ELBO
loss weight w(50) = beta_50 / (2 * ab_50 *
(1 - ab_50)) under the constant schedule and
compare with the lesson's 0.02873. (c) Which
schedule destroys signal faster by t = 50, and
what does the lower weight imply for training
emphasis?

T2. The lesson's posterior at t = 50 has mean
0.03956209 * x_0 + 0.96013574 * x_t.
(a) With x_t = 1.2 and x0_hat = 2.0, compute the
reverse mean under x0 prediction. (b) Changed
constraint: the net predicts x0 instead of eps.
Which coefficient multiplies the net output,
and why does that make eps prediction the
stable default at t = 50?

## T-U08, diffusion variants

T1. The lesson's toy U-Net costs 156672 FLOPs
per forward pass. (a) A serving budget allows
1 MFLOP per sample. List the feasible (S, g)
configs among (5, 1), (5, 2), (10, 1) and pick
one with a reason. (b) Changed constraint: S
drops from 10 to 5. Propose two 5-point
schedules and predict which degrades less,
using the capstone's per-jump-error finding.
State the assumption behind the prediction.

T2. The lesson computed sigma = 0.39249806 for
the 100 -> 90 jump at eta = 1. (a) Compute sigma
at eta = 0.5 and write the noise term it adds to
the jump. (b) Changed constraint: eta moves
from 0 to 0.5. What does the random seed
control now that it did not control at eta = 0?

## T-U09, score models and autoregressive LMs

T1. The lesson's DSM toy: x = 1.5, x_tilde =
1.7, sigma^2 = 0.04, target = -5.0.
(a) Changed constraint: sigma^2 = 0.16.
Compute the target at x_tilde = 1.7 and at
x_tilde = 1.9. (b) For s_hat = -1.0, compute
the squared-error loss at x_tilde = 1.7.
(c) In a multi-sigma average, which noise level
dominates the loss and why?

T2. Q = K = [[1, 1], [1, 0], [0, 1]]. (a)
Compute the scaled scores QK^T / sqrt(2) and
the causally masked softmax rows. (b) Changed
constraint: the mask is removed. Which row
changes most, and what train/serve mismatch
does that create?

## T-U10, LLM inference and alignment

T1. The lesson's KV cache holds 12.00 MiB at n
= 512. (a) Compute the bytes per token, the
cache at n = 1024, and the cache for a batch of
8 at n = 1024. (b) Changed constraint: n =
4096. Compute the per-sequence cache. (c) State
the roofline consequence of this linear growth
for decode.

T2. Weights w = [0.317, -0.733, 1.127,
-0.183]. (a) Quantize to INT8 with scale s =
0.02: give q, the dequantized values, and the
max absolute error. Check the s/2 bound.
(b) Changed constraint: s = 0.05. Recompute.
(c) State the tradeoff the scale controls.
