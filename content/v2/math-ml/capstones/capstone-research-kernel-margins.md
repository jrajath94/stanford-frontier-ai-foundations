# Capstone R1, research: hinge vs logistic under label noise

Track: research replication plus scoped extension. Date: 2026-10-06.
Status: design complete. Small-scale executed runs are labeled
EXECUTED. Everything else is PROPOSED. A generated plan is not
evidence.

## Question

On linearly separable data with symmetric label noise, does the
hinge loss resist noise better than the logistic loss at fixed
model class (linear) and fixed optimization budget?

## Falsifiable hypothesis

H1: at 15 percent label flips, a linear SVM (hinge, C = 1,
subgradient, 200 steps) reaches higher train accuracy than
logistic regression (200 gradient steps, eta = 0.5) on the same
data, and the gap shrinks to zero at 0 percent flips.
Falsification: if the accuracies tie within 0.01 at 15 percent
flips, H1 is rejected.

## Literature (to read, not claimed as read)

- Vapnik, statistical learning theory: margin bounds.
- Shalev-Shwartz et al., Pegasos: primal SVM optimization.
- Huber: outlier-resistant loss functions and influence.

## Data

EXECUTED: the lesson's 60-point Gaussian toy, seed 7, 9 of 60
labels flipped. PROPOSED: 20 fresh seeds, n = 600 per run, flip
rates {0, 0.05, 0.15, 0.25}.

## Baselines

Logistic regression (gradient descent) and linear SVM
(subgradient), both linear, no intercept tricks: identical model
class, loss as the only difference.

## Matched budgets

Both get 200 passes over the data, same initialization (zeros),
step sizes tuned once on seed 7 and frozen. Compute is trivial. 
the budget that matters is the step count.

## Metrics

Primary: train accuracy gap (SVM minus logistic). Secondary:
held-out accuracy gap on 600 fresh points. Diagnostic: weight
norms ||w|| for both.

## Controls

Same data, same seeds, same step budget. Flip masks fixed per
seed across both models.

## Ablations

Loss only (the point of the study). Then C in {0.1, 1, 10} for
the SVM. Eta in {0.1, 0.5} for logistic, reported as
sensitivity, not tuned per seed.

## Seed variation

EXECUTED: 1 seed (seed 7). PROPOSED: 20 seeds. Report mean and
std of the gap.

## Uncertainty

EXECUTED: none quantified (n = 1 seed). PROPOSED: standard
error over 20 seeds. The gap must clear 2 SE to claim.

## Executed results (2026-10-06, numpy 1.26.4, float64)

Seed 7, 9 flips: logistic train accuracy 0.7167, linear SVM
0.7833. Gap +0.0667 in favor of the hinge. This EXECUTED run
is consistent with H1 and does not confirm it.

## Failure criteria

H1 is rejected if the 20-seed mean gap is within 0.01 of zero,
or if the sign flips on the majority of seeds. The study stops
early if logistic diverges on any seed (NaN weights): that
becomes a finding about stability, reported as a negative
result for the protocol, not silently dropped.

## Reproducibility

Script: compute_run5.py (C12 block), seed 7, numpy 1.26.4.
The 20-seed extension reruns the same block with seeds 7-26.

## Negative results

To be recorded: any seed where the gap reverses, any C where
the SVM collapses, any eta where logistic diverges. Negative
results ship in the writeup.

## Limitations

Linear models only. Synthetic Gaussian data. Train accuracy as
the primary metric (a deliberate weakness, flagged). No claim
about deep nets or real label noise.

## Ethical considerations

None beyond honesty: no human data, no deployment claims. The
writeup must not present the single-seed EXECUTED run as a
general finding.

## Extension (scoped)

If H1 survives 20 seeds, extend to the RBF kernel at fixed
gamma with matched Gram budgets, testing whether the noise
resistance survives nonlinearity. One paragraph of design. 
execution is a separate proposal.
