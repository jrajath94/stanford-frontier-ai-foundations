# Interview bank: U03 (questions)

Unit: math-genmodels-U03. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice questions. Not actual employer
questions. Keys in interview/keys/interview-u03.md.

## Breadth (6)

Q1. A deterministic autoencoder reaches near-zero reconstruction
error. Why might its latent codes still be useless?
Q2. What turns an autoencoder into a variational autoencoder? Name
the new moving parts.
Q3. Write the Gaussian KL closed form. When is it zero?
Q4. Define posterior collapse. How do you detect it on a trained
model?
Q5. What does the beta in beta-VAE control? What breaks if you tune
it on the beta objective?
Q6. A VAE reports 2.2 bits/dim. Is that number exact? What would
make it exact?

## Deep ladders (2 x 5)

L1 (the VAE bound):
1. Write the VAE ELBO and map each symbol to encoder, decoder, or
   prior.
2. Compute the reconstruction term, KL term, and ELBO for the toy
   numbers.
3. Derive the Gaussian KL closed form from the KL integral.
4. Implement the ELBO in numpy and check it against the closed
   form.
5. Compare the VAE against a deterministic autoencoder with a
   post-hoc prior on training cost, sampling, and the honesty of
   the likelihood number.

L2 (discrete latents and gradients):
1. Define the codebook posterior for a datapoint and four codes.
2. Compute it on the toy.
3. Explain why argmin quantization has no usable gradient.
4. Define the straight-through estimator and compute its bias on
   the threshold toy.
5. Compare straight-through against Gumbel-Softmax on bias,
   variance, and tuning knobs. When would you switch?

## Analytical exercises (2)

A1. Show that the beta-VAE objective is a lower bound on log p(x)
for beta >= 1 but not for beta < 1. Give the counterexample.
A2. A VAE has a 2-D diagonal Gaussian posterior. Derive the total
KL as a sum of per-dim terms and explain why this decomposition is
the collapse diagnostic.

## Implementation and debug (1)

D1. A VAE's second latent dimension shows KL 0.000 across the
validation batch while reconstruction is good. The proposal is to
delete the KL term. Diagnose the real problem and give the cheaper
fix first.

## Changed-constraint scenarios (2)

T1. The decoder must run on 8-bit integer hardware. The Gaussian
likelihood with float variance no longer fits. Redesign the
likelihood and the loss, and state what happens to exactness of
the bound.
T2. The product team wants sharper samples and accepts worse
likelihood. Name the two knobs from this unit that trade blur for
sharpness, predict the direction of each, and state what metric
you would watch instead of BPD.

## Research critique (1)

R1. "Our discrete-latent model beats the Gaussian VAE on BPD, so
discrete codes are better latents." Give two reasons this
comparison can mislead, and design the experiment that would
settle it.
