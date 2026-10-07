# U10 interview bank , questions

Closed-book. Keys in `u10_key.md`. All numbers are synthetic toys
from `visuals/compute_u10.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. State the 6ND rule and the 2x/4x split.
B2. What is the loss floor E, and why can scale never remove it?
B3. How do you read a power-law exponent off a log-log fit?
B4. What does an isoFLOP curve hold fixed, and what does its minimum
mean?
B5. What is preregistration, and what are its five fields?
B6. Name the three scale-transfer failure modes.

## Deep ladders (2 x 5)

L1. Power-law fitting.
- L1.1 Define the log-space transform.
- L1.2 Toy: fit the 24 synthetic points, report a=0.340.
- L1.3 Derive the normal equations for the slope.
- L1.4 Implement the fit, state the noiseless recovery check.
- L1.5 Compare log-linear with nonlinear fit, debug the wrong-E
  case, critique the iid noise assumption, propose the
  E-perturbation experiment.

L2. Compute-optimal allocation.
- L2.1 Define N_opt(C) and D/N.
- L2.2 Toy: allocate C=5.88e23, report N=7.00e10, D=1.40e12.
- L2.3 Derive the 0.5 exponent from the symmetric joint toy law.
- L2.4 Implement `allocate`, state the C-reproduction check.
- L2.5 Compare with overtraining, debug the data-cap case,
  critique the symmetric toy, propose the constrained-allocation
  experiment.

## Analytical exercises (2)

E1. A team fits a=0.34 on C in [1e20, 1e21] and predicts loss at
C=1e23. (a) How many orders of magnitude is the extrapolation?
(b) Name two reasons the band is wider than the bootstrap says.
(c) What staged evidence would you demand before a 100x bet?
E2. Config X: 30B at 20 tok/param. Config Y: 7B at 200 tok/param.
(a) Compute both training FLOPs. (b) The toy losses are 2.45 and
2.65. The product serves 1e14 tokens. Which config is
inference-optimal and why? (c) Name the assumption that could
flip the answer.

## Implementation/debug task (1)

D1. This fit returns a negative exponent. Find the bug.

```
x = np.log(C)
y = np.log(L - E)
M = np.vstack([np.ones_like(x), x]).T
coef = np.linalg.lstsq(M, y, rcond=None)[0]
a = coef[1]  # bug here
```

## Changed-constraint scenarios (2)

S1. The corpus is capped at 5e11 tokens, C=5.88e23. Allocate and
defend the choice against the unconstrained optimum.
S2. Only 6 loss points exist, spanning half an order of magnitude.
The team reports a=0.34 with no interval. Repair the report.

## Research critique (1)

R1. A paper claims a new scaling law from 8 runs over 1 order of
magnitude, no residuals, no E discussion, no preregistration.
List three gaps and the check for each.
