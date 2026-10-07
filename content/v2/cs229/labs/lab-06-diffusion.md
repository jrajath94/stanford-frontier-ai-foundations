# Lab 06: Diffusion forward chain and the denoising identities

Date: 2026-10-06. Unit: cs229-U12.
Work the tasks, then check keys-lab-06.md. Run all code.

Setup: numpy only. Seeds as stated per task. No other installs.
Schedule everywhere: T = 100, beta linear from 1e-4 to 0.02.

## Task 1: forward sampling, two ways

Take x_0 = 3.0 in one dimension. Compute alpha-bar_t for
t = 50. Report the theoretical mean and standard
deviation of x_50. Then (a) sample x_50 by the
single-step recursion (14.3) starting from x_0,
and (b) sample x_50 by the closed form (14.5),
both with seed 7 and 20,000 draws. Report both
sample means and standard deviations. State
whether they agree with theory and with each
other.

## Task 2: the posterior-mean identities

With the same schedule and x_0 = 3.0: compute
beta-tilde_50 and verify beta-tilde_50 <
beta_50. For x_t values on a grid of 61 points
from -4 to 8, compute mu-tilde_50 by (14.19)
and by (14.22). Report the maximum absolute
disagreement. Then pick x_t = 3.0420 and report
mu-tilde_50. State what the agreement proves.

## Task 3: the score relation

With the same schedule, x_0 = 3.0, t = 50:
(a) compute grad_x log q(x_t | x_0) at x_t =
3.0420 analytically, (b) compute
-epsilon-hat / sqrt(1 - alpha-bar_t) with
epsilon-hat recovered from x_t and x_0,
(c) approximate the score by central finite
differences with h = 1e-5. Report all three
numbers and the pairwise gaps. State which
object the noise predictor is really learning.

## Deliverable

A short log: the three result blocks with numbers. No
essay. The numbers must match keys-lab-06.md within the
stated tolerance.
