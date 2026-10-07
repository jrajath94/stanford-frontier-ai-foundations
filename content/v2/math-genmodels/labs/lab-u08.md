# Lab U08: honest evaluation

Unit: math-genmodels-U08. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u08.md. Do not open the keys
until the code runs.

## E1: held-out scoring

Truth 0.5 N(-2,0.25) + 0.5 N(2,0.25). Models A = N(0,4.25), B =
N(2,0.25). Held-out points (-2.1, 1.9, -1.8, 2.2, 0.1). Compute
both total logliks. Confirm -10.1073 versus -70.949. Identify
the single point that contributes most to the gap.

## E2: bound versus exact

Point y = 1.0, model N(0,4). Compute the exact loglik
(-1.7371). With q = N(0.5,1), compute the gap KL(q||N(0,1)) =
0.125 and the bound -1.8621. Then invent a second VAE with
bound -1.70 and state exactly what you can and cannot conclude
versus the flow.

## E3: precision and recall

Draw 4,000 samples from A and B (seed 13). Manifold: within 1
of either mode. Compute all four numbers. Confirm 0.4708,
1.0, 0.9555, 0.5. Then shrink the radius to 0.5 and report how
each number moves. State which model the radius change favors.

## E4: the FID trap

Compute the 1-D FID between the truth's moments and model A.
Confirm 0. Build a discrete 3-point law with mean 0 and
variance 4.25. Confirm its FID is also 0. Plot all three
densities/histograms and write the one-sentence lesson.

## E5: memorization screen

50 training points (seed 13). Memorizer: resample training
points plus N(0, 0.01) noise. Honest: 400 samples from model
A. Compute mean NN distances. Confirm 0.0066 versus 0.4666.
Plot both histograms. Then raise the memorizer noise to 0.5
and report when the screen stops flagging it.

## E6: seed uncertainty

Model A precision over seeds 100-104, n = 2000 each. Report
mean and std. Confirm 0.4835 +/- 0.0115. A rival reports 0.490
on one seed. State whether the rival is better, with the
arithmetic.
