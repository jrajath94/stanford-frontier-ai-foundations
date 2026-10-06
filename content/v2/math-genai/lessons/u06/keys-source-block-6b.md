# Answer keys, source block 06b

Date: 2026-10-06. Ground truth: compute_run5a.py.

## E01

The beta-VAE weights the KL term by beta, trading
reconstruction against latent use with one scalar.
Toy: beta = 2 gives -1.0874 nats, beta = 8 gives
-2.1468 nats.

## E02

Beta-VAE prices latent information softly first. VQ-VAE enforces it with a hard discrete bottleneck
second. The playlist orders the easy knob before the
structural change. The pairing is a playlist fact. Shared lecture content is not inspected.

## E03

A VQ-VAE replaces the Gaussian latent with a learned
codebook lookup. The three VQ losses replace the
ELBO's KL term. The straight-through estimator
replaces reparameterization.

## E04

Recon 0.05, codebook 0.05, commitment 0.0125. The
codebook loss moves the codebook entries.

## E05

(1) codebook table: replaces the Gaussian prior
params. (2) quantizer: replaces the sampler. (3) STE
wrapper: replaces reparameterization. (4) three
losses: replace the ELBO. (5) EMA update: replaces
the KL closed form. (6) usage counter: new (dead-code
diagnostic). (7) prior fitting: new (stage 2). The
encoder/decoder nets, batches, and optimizers are
shared.

## E06

No. No tutorial code was inspected. Gamma = 0.99 is
an authored choice from the lesson. Citing W6T13 for
it violates the inspection boundary.

## L01

VQ-VAE: an autoencoder whose latent is a codebook
index, trained with three losses and straight-through
gradients. Toy: distances [0.05, 1.45, 3.65, 2.25],
winner e_1. Derivation: the argmin is piecewise
constant, so its gradient is 0 almost everywhere.
Implement: quantize plus the z_e + sg[z_q - z_e]
wrapper. Compare: the titles promise the VQ-VAE idea
and its implementation. No transcript inspected, so
the lecture's K, losses, and gradient treatment are
unknown. Debug: frozen encoder weights mean the STE
wrapper is absent. The plain argmin output carries
no gradient. Add the wrapper. Critique: "variational"
is strained: there is no q(z|x) approximating a
posterior, only a deterministic quantization. The
name is historical. Design: an inspected transcript
or slide deck showing the VQ-VAE losses would promote
the row.
