# Crash course: math-genmodels in about forty minutes

Date: 2026-10-06. Baseline: October 6, 2026.

Eight units, about five minutes each. Every worked number below
comes from the named lesson section. The crash course points, the
lessons prove. Honesty banner from README.md applies: this is a
provisional independent bridge, not asserted official syllabus.

## U01, Probability, measures, and density estimation (lessons/u01)

Probability masses sit on points and sum to 1. Densities sit
under curves and integrate to 1. The empirical distribution puts
mass 1/n on each sample and approaches the truth as n grows.
Likelihood scores fixed data under moving parameters. Entropy is
expected surprise. KL divergence is directed: KL(p||q) punishes
q for missing p's mass.

Worked number (U01): 12 draws from (0.5, 0.333, 0.167). Loglik
under the true law -12.137 versus -13.183 under a wrong law.
The data pick the truth by 1.046 nats.

Memory aid: mass sums, density integrates. The likelihood is a
function of the parameters, not the data.

Self-test: Q: Why is p(x) = 0.3989 at the mode not a
probability? A: A density value is not a mass. Only integrals
of it are probabilities.

## U02, Latent models and variational inference (lessons/u02)

The marginal p(x) = sum over z of p(x, z) is intractable when z
is rich. Bayes inverts to the posterior. Jensen on the concave
log gives the ELBO: log p(x) >= E_q[log p(x,z) - log q(z)].
The gap is KL(q||posterior): zero exactly when q matches the
posterior. Forward KL covers, reverse KL seeks. The
reparameterization trick moves noise outside the parameters so
gradients flow.

Worked number (U02): log p(R) = -0.5108, ELBO -0.5390, gap
0.0282. The bound sits below the truth by exactly the KL.

Memory aid: the ELBO is a floor. The gap is the distance from
your q to the truth.

Self-test: Q: When is the ELBO tight? A: When q equals the
posterior. Then the gap KL is zero.

## U03, Autoencoders and representation objectives (lessons/u03)

A deterministic autoencoder compresses and rebuilds. A VAE
trains the ELBO: reconstruction minus the KL to the prior. The
Gaussian KL has a closed form. Posterior collapse kills the
latents: the decoder ignores z. Beta scales the rate price on
the rate-distortion frontier. VQ replaces the continuous
latent with a codebook and pays the straight-through bias.

Worked number (U03): recon -2.1516, KL 0.9013, ELBO -3.0529,
2.2022 bits/dim. The bound, labeled as a bound.

Memory aid: rate buys distortion. Dead latents have zero KL.

Self-test: Q: KL reads 0.0003 per dim but reconstructions are
sharp. What happened? A: Collapse. The decoder memorized and
ignores z.

## U04, Change of variables and normalizing flows (lessons/u04)

An invertible map plus the Jacobian determinant gives exact
density: p_x(x) = p_z(f^-1(x)) x |det J|. Coupling layers make
the Jacobian triangular, so the determinant is the diagonal
product at O(D) cost. Forward samples, inverse scores. MAF is
cheap at density eval, IAF at sampling. One affine layer of a
Gaussian stays Gaussian. Accumulate log-determinants or
overflow.

Worked number (U04): log p(y) = -2.4629 - 0.2231 = -2.6860
nats at the toy point. Exact, not a bound.

Memory aid: the minus sign. Forgetting it reports -2.2398.

Self-test: Q: Why triangular? A: The determinant becomes the
diagonal product. O(D^3) becomes O(D).

## U05, Adversarial and ratio-based estimation (lessons/u05)

An implicit law samples without a density. The discriminator
estimates the density ratio through D* = p/(p+q). At the
optimum the game minimizes the Jensen-Shannon divergence:
max_D V = -log 4 + 2 JSD. JSD saturates on disjoint laws and
the gradient dies. The Wasserstein dual keeps a linear signal
but needs the 1-Lipschitz constraint. The gradient penalty
enforces it only on interpolations: a heuristic, not a proof.
Mode collapse is a recall failure the loss cannot see.

Worked number (U05): D*(0) = 0.6225, JSD = 0.1114, V* =
-1.1635. Disjoint: JSD = 0.6931 = log 2, W1 = 5.0.

Memory aid: the critic is a ratio meter. Perfect accuracy
means saturation, not convergence.

Self-test: Q: The discriminator hits 100 percent. Good news?
A: No. JSD is saturated at log 2 and the generator gets zero
gradient.

## U06, Scores, diffusion, and stochastic dynamics (lessons/u06)

The score is grad log p: it points uphill with no normalizer.
Score matching fits it without the trace via denoising.
Tweedie turns a noisy score into a clean estimate. The DDPM
chain destroys signal on a schedule. The loss predicts the
noise. DDIM strides the chain deterministically. Langevin
adds calibrated noise to uphill steps. Guidance steers the
score. Solvers trade steps for accuracy. Boundary scores blow
up: likelihoods need a named noise floor.

Worked number (U06): Tweedie 1.2 -> 0.0 exactly. DDIM step
1.70 -> 1.7630. DDPM loss 0.005 on the toy batch.

Memory aid: the score needs no partition function. The noise
schedule is the curriculum.

Self-test: Q: Why does denoising avoid the trace term? A:
The model is never differentiated. Tweedie links noise
prediction to the score directly.

## U07, Autoregressive models and modern bridges (lessons/u07)

The chain rule factors the joint exactly: order is the model.
Teacher forcing trains on true prefixes. Sampling sees its
own outputs (exposure bias). The causal mask is the arrow of
time. Attention is a weighted past, costed at O(L^2 d) with a
KV cache. Temperature reshapes the draw without retraining.
Masked models have no valid joint: pseudo-perplexity sums to
1.0639, not 1.0. Flow matching regresses the velocity of a
chosen path directly.

Worked number (U07): log p(abc) = -2.9957, perplexity 2.7144.
Attention head row 2: weights (0.3302, 0.6698), output
(3.3395, 4.3395).

Memory aid: exact likelihood, serial sampling. Compare
perplexity only within one tokenization.

Self-test: Q: Why is the AR likelihood exact? A: The chain
rule is an identity. No bound, no sampling.

## U08, Evaluation, inference, and experimental synthesis (lessons/u08)

Held-out likelihood is the honest score. Bounds are floors:
never rank a bound against an exact number silently. Precision
is quality, recall is coverage: report both. FID sees only
moments: FID 0 with the wrong shape is the counterexample.
Memorization hides from likelihood: run the NN test.
Match compute before comparing ideas. Synthetic truth is the
laboratory. Report seed uncertainty. Pre-register the failure
criterion: falsify, do not confirm.

Worked number (U08): held-out A -10.1073 versus B -70.949.
Precision 0.4708/0.9555, recall 1.0/0.5. NN 0.0066 versus
0.4666.

Memory aid: one metric is one blind spot with confidence.

Self-test: Q: FID is 0. Done? A: No. Match the moments and
miss the shape: FID 0, truth bimodal, model unimodal.

## The whole course in eight lines

U01: mass sums, density integrates, likelihood scores.
U02: the ELBO floors the marginal, the gap is a KL.
U03: rate buys distortion, dead latents cost nothing.
U04: invert, correct the volume, read the exact number.
U05: the game minimizes JSD at the optimum, which never
holds. Watch the geometry.
U06: learn the uphill direction at every noise scale, then
walk back.
U07: order the factors, pay the quadratic context, sample
serially.
U08: no single number is honest. Falsify.
