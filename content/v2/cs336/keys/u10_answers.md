# U10 answer key

All numeric claims from `../visuals/compute_u10.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

log L = [0.1, -0.24, -0.58], log C = [44, 45, 46]. Slope =
(-0.58-0.1)/(46-44) = -0.34. a = 0.34.

## A1

(a) 6ND: 2 forward, 4 backward per param per token. (b) 5.88e23
FLOPs, D/N = 20.0. (c) FLOPs use active params (7B), memory uses
total params (70B).

## A2

(a) E is the data entropy floor, loss -> E as compute -> inf.
(b) Excess 0.104 vs floor 1.69, band [1.737, 1.739]. (c) No: compare
excess over each mix's own floor, not raw loss.

## A3

(a) log(L-E) = log A - a log(C/C0), slope negated. (b) a = 0.340,
A_hat = 0.499, RMS 0.0031. (c) 6 points over half an order: the fit
runs but the CI is wide, report a with the bootstrap band and refuse
strong claims.

## A4

(a) Loss vs N at fixed C, the minimum is the optimal split. (b)
Fitted 0.449 vs analytic 0.500, the 16-point grid plus noise moves
it. (c) N_opt = C^0.449 scaled from a known point, state the grid
risk: the true optimum may sit between grid points.

## A5

(a) 20 tokens/param minimizes training loss per FLOP in the toy.
(b) N = 7.00e10, D = 1.40e12, ratio 20.0. (c) D capped at 2e11:
allocate N from C = 6*N*2e11 -> N = 8.3e9, defend: the data boundary
binds, the unconstrained optimum is unreachable.

## A6

(a) lr ~ 1/width for hidden layers in the toy. (b) 7.5e-05.
(c) lr = 3e-4 * 256/4096 = 1.9e-05, residual risk: depth changed
too, and the width rule says nothing about depth: watch for
divergence, be ready to retune.

## A7

(a) Residuals diagnose form errors. RMS vs noise level decides.
(b) 0.0031 vs injected 0.02 on the excess: consistent, law fits.
(c) RMS 0.05 >> noise 0.01: the law is wrong or E is wrong, refit
with E free.

## A8

(a) The band of honest predictions from resampled fits. (b) CI
[0.337, 0.343], band [1.737, 1.739]. (c) Band from the CI
endpoints at 10x, planning risk: the band assumes the law holds
out of range.

## A9

(a) Compute-optimal: min loss per train FLOP. Inference-optimal:
min loss + serving cost. (b) Gap 0.36 nats, B trains 10x cheaper
and serves 10x cheaper per token. (c) Q = 1e15: inference cost
dominates. B, unless the 2.4 quality bar binds: A loss 2.290
passes, B loss 2.646 fails: A ships despite the cost.

## A10

(a) Prediction, band, n, decision rule, date. (b) Band [0.30,
0.38], observed 0.340: held. (c) A valid registration, e.g.:
"KV per token in [120, 140] KB, pass if the computed value lands
inside."

## A11

(a) 0.357 nats per 10x data at fixed 7B. (b) The data term
flattens as D^{-a}. (c) 3T/1B = 3000 tok/param available, sane
stopping point is where marginal nats per FLOP fall below the
alternative use of the compute: state the rule, not a number.

## A12

(a) Form change, noise regime, range. (b) Fitted 0.449 vs 0.500,
deviation to 0.277: grid + noise artifacts. (c) Ask: what is the
fitted range, what is the noise model, what happens at the
boundary: refuse the 100x claim without staged evidence.
