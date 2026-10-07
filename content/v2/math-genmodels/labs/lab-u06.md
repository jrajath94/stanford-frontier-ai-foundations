# Lab U06: scores and diffusion

Unit: math-genmodels-U06. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u06.md. Do not open the keys
until the code runs.

## E1: score field audit

For p = N(0,1): compute s(x) on the grid x in [-3, 3]. Confirm
s(1) = -1, s(0) = 0, s(-2) = 2. Plot the density with score
arrows at 11 points. Then fit s_theta(x) = -a x by minimizing
the empirical score-matching objective on 5,000 samples (seed
11) over a in (0.5, 1.0, 2.0). Report the winner and its value.

## E2: Tweedie check

Point mass at 0, sigma = 0.5. For observed x in (1.2, -0.8,
2.5): compute the noisy score and the Tweedie estimate.
Confirm all three return 0.0. Then use two point masses at -2
and 2 with sigma = 1.5, observe x = 0, and report the Tweedie
estimate and the true posterior. Write the one-sentence lesson.

## E3: chain trajectory

Betas (0.1, 0.2, 0.3, 0.4), x_0 = 2.0. Sample x_t for t = 1..4
two ways: (a) the closed form with one eps per t, (b) stepping
through the chain with fresh noise each step. Use seed 11.
Report both trajectories and confirm they differ (different
noise realizations) but both satisfy the marginal law.

## E4: reverse step

Compute the DDPM reverse kernel for t = 2 -> 1 with x_0 = 2.0,
x_2 = 1.7. Report the mean 1.8983 and variance 0.0714. Then
compute the DDIM eta = 0 step from x_2 = 1.7 with predicted
noise 0.5. Report 1.7630. Run the DDIM step twice and confirm
identical output.

## E5: sampler comparison

Integrate dx/dt = -x from 1 to t = 1 with Euler at 10, 20, 40
steps and Heun at 10 steps. Report all four errors against
e^-1. Confirm Euler error quarters when steps double. State
the cheapest solver under error 0.001.

## E6: boundary score

Smoothed uniform(0,1), sigma = 0.05. Compute the score at x =
0.01, 0.1, 0.5 analytically via the Phi formula. Confirm 13.5,
a moderate value, and 0.0. Then shrink sigma to 0.01 and report
the new boundary score. Write the one-sentence lesson for
likelihood reporting.
