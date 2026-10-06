# Interview bank, U05 variational autoencoders

Date: 2026-10-06. Questions only. Keys in
interview/keys-u05.md. Closed-book. Do not read the keys
first. The running toy: x = [1.2, 0.4], z in R^1,
p(z) = N(0, 1), encoder mu = 0.4, sigma^2 = 0.5,
decoder N(W z, 0.25 I), W = [[1.0], [0.5]].

## Breadth (6)

B1. Name the three VAE parts. State what each one
takes as input and returns as output.
B2. Derive the ELBO from log p(x) in five lines.
Name the step where Jensen is used.
B3. Compute KL(N(0.4, 0.5) || N(0, 1)) from the
closed form. Give nats and bits.
B4. Write the reparameterization z = mu + sigma
eps. Explain why gradients reach mu through it.
B5. Fill the collapse ledger: healthy vs collapsed
ELBO on the toy. State which wins.
B6. Define amortized inference. Contrast it with
the EM E-step in one sentence each.

## Deep ladder D1, the ELBO (5 follow-ups)

D1.1. Define the ELBO in one sentence, no jargon.
D1.2. Toy: compute the 1-sample ELBO from recon
-0.7342 nats and KL 0.1766 nats.
D1.3. Derive the five-line proof from log p(x).
D1.4. Implement/debug: a colleague's 1-sample ELBO
exceeds their grid-computed log p(x) on one draw.
Is the bound violated? What do you tell them?
D1.5. Changed constraint: the prior becomes a
mixture of two Gaussians. Which closed form breaks
and what replaces it?

## Deep ladder D2, collapse and beta (5 follow-ups)

D2.1. Define posterior collapse in one sentence.
D2.2. Toy: fill the collapse ledger with the four
numbers (recon and KL for both rows).
D2.3. Derive why the ELBO prefers q = prior when
the decoder is perfect.
D2.4. Implement/debug: KL per dim reads 0.0001 on
all 32 dims but the ELBO looks healthy. Diagnose
and name the confirming test.
D2.5. Research critique: "A higher ELBO means a
better generative model." Attack with the ledger.

## Analytical/quantitative (2)

Q1. At beta = 4 the toy KL price is 0.7063 nats.
The healthy recon is -0.7342 nats, collapsed recon
-0.4516 nats. Compute both beta-ELBOs and state
which model survives.
Q2. Solve for beta* where the collapsed ELBO
equals the healthy beta-ELBO on the toy. Show the
arithmetic.

## Implementation/debug (1)

T1. This code intends the 1-sample ELBO:

```python
import numpy as np
def vae_loss(x, mu, logvar, xhat, s):
    recon = -0.5 * ((x - xhat) ** 2).sum() / s ** 2
    kl = 0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar)
    return -(recon - kl)
```

It trains, the loss falls, but the KL term stays
exactly 0.0 from the first step. Name the likeliest
cause in the encoder code (not in this function)
and the one-line fix.

## Changed-constraint scenarios (2)

S1. The decoder std s becomes a learned
per-dimension vector. Name the gaming failure and
the constraint that stops it.
S2. The latent dim grows from 1 to 64 with a fixed
decoder. Predict what happens to the collapse
pressure and name the diagnostic you would watch.

## Research-critique (1)

R1. "Reparameterization gradients are always
better than score-function gradients." Attack the
claim: name a latent type where it fails, state
what the toy SE comparison actually proves, and
design the experiment that would change your mind.
