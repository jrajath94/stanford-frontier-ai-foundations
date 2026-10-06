# Interview bank, U07 DDPM derivation and parameterizations

Date: 2026-10-06. Questions only. Keys in
interview/keys-u07.md. Closed-book. Do not read the keys
first. The running toy: T = 100, linear betas 1e-4 to
0.02, x_0 = 2.0, eps = 0.5, eps_hat = 0.4.

## Breadth (6)

B1. Write q(x_t | x_{t-1}) and q(x_t | x_0).
State what alpha_bar_t means in one sentence.
B2. Give beta_50, alpha_bar_50, and SNR_50 on the
toy schedule.
B3. Write the posterior q(x_49 | x_50, x_0): the
variance and the two coefficients.
B4. From eps_hat = 0.4 at t = 50, compute x0_hat
and mu_rev on the toy.
B5. State the ELBO decomposition: the three kinds
of terms. Give the toy L_T in nats.
B6. Write one DDPM training step as five
operations.

## Deep ladder D1, forward to reverse (5 follow-ups)

D1.1. Define the forward process in one sentence.
D1.2. Toy: compute x_50 from x_0 = 2.0, eps = 0.5.
Show both terms.
D1.3. Derive q(x_t | x_0) by unrolling two steps.
D1.4. Implement/debug: a colleague's chain
simulation shows variance growing linearly with
t. Name the missing factor.
D1.5. Changed constraint: beta_t = 0.5 for all t.
Compute alpha_bar_10 and explain the consequence.

## Deep ladder D2, the objective (5 follow-ups)

D2.1. Define the simplified training loss in one
sentence.
D2.2. Toy: compute the five pair losses and their
mean.
D2.3. Derive why each middle KL reduces to a
squared mean difference.
D2.4. Implement/debug: training loss falls but
samples worsen. Name the likeliest weighting cause
and the diagnostic.
D2.5. Research critique: "The simplified loss is
justified because it optimizes a reweighted
ELBO." Attack it.

## Analytical/quantitative (2)

Q1. The middle-KL sum over t = 2..100 is 0.0629
nats. L_T is 0.7713 nats. State the ratio and what
it implies about where the toy ELBO's mass sits.
Q2. Epsilon error 0.1 at t = 50 becomes x0 error
0.0535. Derive the conversion factor from the
schedule numbers.

## Implementation/debug (1)

T1. This code intends one DDPM training step:

```python
import numpy as np
def ddpm_step(x0, ab, T, net, rng):
    t = int(rng.integers(0, T))
    eps = float(rng.normal())
    xt = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps
    eps_hat = net(xt, t)
    return (eps - eps_hat) ** 2
```

The network was built for 1-indexed timesteps
1..T. Name the indexing bug and state whether the
noise level or the timestep label is wrong.

## Changed-constraint scenarios (2)

S1. T = 1000 with the same beta endpoints. State
what happens to alpha_bar_T, to the per-sample
serving cost, and to the SNR = 1 crossing as a
fraction of T.
S2. The data variance is 4, not 1. Name what
breaks in the variance-preservation argument and
give two fixes.

## Research-critique (1)

R1. "DDPM training is just denoising
autoencoders at multiple noise levels." Defend or
attack: name what the ELBO decomposition adds
beyond stacked denoising autoencoders, and state
the experiment that would separate the two views.
