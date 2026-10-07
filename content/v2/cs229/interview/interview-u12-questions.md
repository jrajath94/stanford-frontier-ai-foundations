# Interview bank, U12 diffusion models

Date: 2026-10-06. Questions only. Keys in interview/keys-u12.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Write the forward transition of a diffusion model and name the
two quantities that control one step.
B2. State the closed form of x_t given x_0 and explain in one
sentence why it exists.
B3. What is the reverse parameterization p_theta(x_{t-1} | x_t),
and why is it a reasonable family?
B4. The ELBO splits into named terms. List them and say which one
is dropped during training and why.
B5. Write the simplified training objective and the data each
training step consumes.
B6. State the sampling loop from noise to a new sample.

## Deep ladder D1, forward process (5 follow-ups)

D1.1. Define the forward chain: transition density, the beta
schedule, and alpha-bar_t.
D1.2. Toy: T = 100, beta linear from 1e-4 to 0.02, x_0 = 3.0.
Give alpha-bar_50 and the mean and std of x_50.
D1.3. Derive: why is x_t given x_0 Gaussian, and why is the
accumulated noise variance exactly 1 - alpha-bar_t?
D1.4. Implement/debug: a teammate samples x_T directly with the
closed form but the histogram of 50,000 draws has variance 40.
Name the most likely bug.
D1.5. Changed constraint: the data are 0-255 integer pixels, not
normalized continuous values. Which assumptions of SL-05 break
first, and what must change before likelihood numbers mean
anything?

## Deep ladder D2, ELBO and sampling (5 follow-ups)

D2.1. Write the ELBO with the full path x_1:T as the latent
variable and the forward chain as the auxiliary distribution.
D2.2. Toy: epsilon = 0.8417, x_0 = 3.0, t = 50 on the same
schedule. Compute x_50 and the unweighted loss of a predictor
that outputs 0.9417.
D2.3. Derive: from the posterior mean (14.19) to the noise form
(14.22). State the substitution and the algebraic move.
D2.4. Implement/debug: loss is near zero on training data but
generated samples are noisy garbage. Two candidate causes: (a)
the network memorizes epsilon per (x_t, t) pair seen in
training, (b) sampling starts from a fixed x_T = 0 vector.
Which is more consistent with the evidence, and what is the
one-line fix for the real bug?
D2.5. Research critique: "The unweighted loss is a worse
objective because it is not the ELBO." Attack or defend with
evidence from the lesson: name what each objective optimizes
and which evaluation each one wins.

## Analytical/quantitative (2)

Q1. Prove beta-tilde_t < beta_t for every t, and give its
numeric value at t = 50 on the lesson schedule.
Q2. Derive the score relation grad_x log q(x_t | x_0) =
-epsilon-hat_t / sqrt(1 - alpha-bar_t) starting from the
Gaussian log density of (14.5).

## Implementation/debug (1)

T1. This code intends the full reverse sampling loop from the notes:

```python
import numpy as np
xt = rng.normal(size=d)          # start from white noise
for t in range(T, 0, -1):
    eps_pred = net(xt, t)
    alpha_bar_t = abar[t-1]
    alpha_t = alpha[t-1]
    beta_t = beta[t-1]
    mu = (xt - beta_t / np.sqrt(1 - alpha_bar_t) * eps_pred) / np.sqrt(alpha_t)
    xt = mu + np.sqrt(beta_t) * rng.normal(size=d)
```

The samples come out far too noisy, worst near t = 1. Identify the two
bugs (one in the noise scale, one in the final step), fix them, and state
the check that proves the fixed loop matches the notes.

## Changed-constraint scenarios (2)

S1. You must generate in 20 reverse steps instead of 1000, with
the same total corruption (alpha-bar_20 = alpha-bar_1000 of the
old schedule). Which SL-05 assumption breaks, and what family of
fixes keeps the Gaussian reverse kernel honest?
S2. Your data are discrete tokens, not continuous images. The
Gaussian forward chain no longer applies. Name two standard
replacements for the forward process on discrete data and the
one property the replacement must keep for the ELBO argument
to survive.

## Research critique (1)

R1. "Our diffusion model reaches a better (lower) training loss
than the baseline but its FID is worse. The baseline must be
overfitting." Critique: is the conclusion sound? Separate the
training objective, the likelihood evaluator, and the
perceptual evaluator, and design the one experiment that
decides the question.
